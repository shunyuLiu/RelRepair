import re
import time
import json
import openai
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gen_patch_prompt import sf_construct_prompt

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


def extract_test_method(testcase_lst):
    method_info_lst = []

    bracket_cnt = 0
    bracket_flag = False
    for line in testcase_lst:
        method_info_lst.append(line)
        bracket_cnt += line.count('{')
        if bracket_cnt:
            bracket_flag = True
        bracket_cnt -= line.count('}')
        if bracket_cnt == 0 and bracket_flag:
            return method_info_lst
    return None


def extract_all_patch_codes(orig_patch, dataset, bug_name):
    patch_code_lst = []
    code_patch_pattern = r'```(?:java\n)?(.*?)\n```'
    extracted_lst = re.findall(code_patch_pattern, orig_patch, re.DOTALL)
    function_name = ' ' + dataset[bug_name]['method_signature']['method_name'] + '('
    function_return_type = dataset[bug_name]['method_signature']['return_type']

    if extracted_lst:
        for patch_code in extracted_lst:
            if function_name in patch_code and function_return_type in patch_code:
                patch_code_lst.append(patch_code)
        if len(patch_code_lst) > 0:
            return patch_code_lst

    orig_patch_lines = orig_patch.split('\n')
    len_orig_patch_lines = len(orig_patch_lines)
    for idx in range(len_orig_patch_lines - 1, -1, -1):
        curr_rline = orig_patch_lines[idx]
        if function_name not in curr_rline or function_return_type not in curr_rline:
            continue
        patch_code = extract_test_method(orig_patch_lines[idx:])
        if patch_code:
            patch_code_lst.append('\n'.join(patch_code))
            break
    return patch_code_lst


def extract_patch(dataset, raw_patch_result):
    extracted_patch_result = {}
    for bug_name in raw_patch_result.keys():
        extracted_patch_result[bug_name] = {'prompt': raw_patch_result[bug_name]['prompt'], 'patches': [],
                                            'raw_patches': []}
        for raw_patch in raw_patch_result[bug_name]['patches']:
            extracted_patch_result[bug_name]['raw_patches'].append(raw_patch)
            extracted_patches = extract_all_patch_codes(raw_patch, dataset, bug_name)
            for extracted_patch in extracted_patches:
                if extracted_patch.startswith(':'):
                    extracted_patch = extracted_patch[1:]
                elif extracted_patch.startswith('@@ Response:'):
                    extracted_patch = extracted_patch[12:]
                extracted_patch_result[bug_name]['patches'].append(extracted_patch)
    return extracted_patch_result


def api_gpt3_5_response(prompt, n):
    response = openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        # model="gpt-4o",
        model="gpt-3.5-turbo-1106",
        n=n,
        temperature=0.8)
    return response

def api_deepseek_response(prompt, n):
    responses = []
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for _ in range(n):
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("This is decoded context", decoded)
        # Remove prompt from the beginning if needed
        response_text = decoded.replace(prompt, "").strip()
        print("This is response context", response_text)
        responses.append(response_text)

    return responses


def query_model(prompt, bug_name):
    delay = 10
    batch_size = 1
    patches = {}
    curr_patch = {}
    curr_patch['prompt'] = []
    curr_patch['patches'] = []
    suggest_patch = []
    try:
        # response = api_gpt3_5_response(prompt, batch_size)
        response = api_deepseek_response(prompt, batch_size)
        for output in response:
            fixed_result = output.strip('`').strip()
            suggest_patch.append(fixed_result)

        curr_patch_cnt = len(curr_patch['patches']) + len(suggest_patch)
        print(
            f'### [APR]:bug_name:{bug_name:25}  |  curr_patch_cnt:{curr_patch_cnt:>3}  |  batch_size:{batch_size:2}  |  patches_cnt:{len(patches):3} ###')

    except Exception as e:
        print(f"Exception in api_query_model {e}")
        if "Please reduce " in str(e):
            raise ValueError("Over Length")
        time.sleep(delay)
    curr_patch['patches'].extend(suggest_patch)
    patches[bug_name] = curr_patch
    return patches


class AprInfo():
    def __init__(self, dataset_path, out_path, target_bug):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        if target_bug is not None:
            self.dataset = {target_bug: self.dataset[target_bug]}
        self.out_path = out_path
        self.target_bug = target_bug
        return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='dataset path')
    parser.add_argument('-o', type=str, required=True, help='patch_result path')
    parser.add_argument('-bug', type=str, required=False, help='bug')

    return parser.parse_args()


def main():
    apr_result = {}
    apr_info = AprInfo(
        args.d,
        args.o,
        args.bug
    )
    with open("C:/Users/uqsliu32/Desktop/Code/Srepair/dataset/single_function_location_repair.json", 'r') as f:
        dataset2 = json.load(f)

    prompt = sf_construct_prompt(apr_info.dataset, apr_info.target_bug, dataset2)
    # print(prompt)
    # # prompt = f"<s>[INST] <<SYS>>\nYou are an Automatic Program Repair Tool.\n<</SYS>>\n\n{prompt} [/INST]"
    solutions = query_model(prompt, apr_info.target_bug)
    apr_result = extract_patch(apr_info.dataset, solutions)
    with open(apr_info.out_path, 'w') as f:
        json.dump(apr_result, f, indent=2)
    return


if __name__ == '__main__':
    args = parse_arguments()
    main()
