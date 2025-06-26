import hashlib
import faiss
import numpy as np
import javalang
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class Function:
    def __init__(self, function_name, comments, function_code, function_inputs, unique_id):
        self.function_name = function_name
        self.comments = comments
        self.function_code = function_code
        self.function_inputs = function_inputs
        self.unique_id = unique_id


def extract_functions_and_variables(java_code, tag):
    tree = javalang.parse.parse(java_code)
    functions = []
    function_counter = {}

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            if tag == "file" or (node.modifiers and 'public' in node.modifiers):
                for member in node.body:
                    if isinstance(member, javalang.tree.MethodDeclaration):
                        if tag == "variable" and 'private' in member.modifiers:
                            continue

                        function_name = member.name
                        function_inputs = [param.type.name + " " + param.name for param in member.parameters]
                        method_code = extract_method_code(java_code.splitlines(), member)

                        if method_code:  # Ensure method_code was successfully extracted
                            preceding_comments = extract_comments_before_method(java_code.splitlines(),
                                                                                member.position.line - 1)
                            inline_comments = extract_inline_comments(method_code)
                            all_comments = preceding_comments + "\n" + inline_comments if inline_comments else preceding_comments

                            method_code_without_comments = remove_comments_from_code(method_code)

                            if function_name not in function_counter:
                                function_counter[function_name] = 1
                            else:
                                function_counter[function_name] += 1
                            unique_id = f"{function_name}_{function_counter[function_name]}"

                            function_obj = Function(function_name, all_comments.strip(), method_code_without_comments,
                                                    function_inputs, unique_id)
                            functions.append(function_obj)

    return functions


def extract_method_code(code_lines, method_node):
    """
    Extracts the method code from the Java source given the method's AST node.
    """
    start_line = method_node.position.line - 1  # Convert to zero-indexed
    open_braces = 0
    in_method_body = False
    method_lines = []

    for i in range(start_line, len(code_lines)):
        line = code_lines[i]

        # Track braces to find the full function body
        open_braces += line.count('{')
        open_braces -= line.count('}')

        # Start adding lines once the method body starts
        if '{' in line and not in_method_body:
            in_method_body = True

        if in_method_body:
            method_lines.append(line)

        # If all braces are closed, end of method is found
        if in_method_body and open_braces == 0:
            break

    # Only return method lines if we have found a valid method body
    if method_lines:
        return "\n".join(method_lines)
    else:
        return None


def extract_comments_before_method(code_lines, start_line):
    comments = []
    for i in range(start_line - 1, -1, -1):
        line = code_lines[i].strip()
        if not line.startswith("//") and not line.startswith("/*") and not line.startswith("*"):
            break
        comments.insert(0, code_lines[i])
    return "\n".join(comments) + "\n" if comments else ""


def extract_inline_comments(method_code):
    inline_comments = re.findall(r'//.*?$|/\*.*?\*/', method_code, re.DOTALL | re.MULTILINE)
    return "\n".join(inline_comments)


def remove_comments_from_code(code):
    code_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)
    code_without_comments = "\n".join([line for line in code_without_comments.splitlines() if line.strip()])
    return code_without_comments


def save_to_jsonl(functions, filename):
    with open(filename, 'w') as f:
        for function in functions:
            function_dict = {
                "unique_id": function.unique_id,
                "function_name": function.function_name,
                "comments": function.comments,
                "function_inputs": function.function_inputs,
                "function_code": function.function_code
            }
            f.write(json.dumps(function_dict) + '\n')


def encode_functions(functions, document_model):
    if not functions:
        print("No functions to encode.")
        return None  # Return None if there are no functions

    embeddings = []
    for func in functions:
        # Combine function name, inputs, and comments
        combined_text = func.function_name + " " + " ".join(func.function_inputs) + " " + func.comments

        # Encode combined text using Sentence-BERT
        embedding = document_model.encode(combined_text)

        embeddings.append(embedding)

    return np.vstack(embeddings) if embeddings else None


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def main():
    # Step 1: Extract functions from Java file
    print("Extracting the functions ...")
    java_file_path = '/yourPatch'
    with open(java_file_path, 'r') as file:
        java_code = file.read()
    tag = "file"
    functions = extract_functions_and_variables(java_code, tag)

    if not functions:
        print("No functions were extracted. Please check the Java code or tag.")
        return

    save_to_jsonl(functions, '/yourPatch')

    # Step 2: Load embedding models
    print("Loading the embedding model...")
    document_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 3: Encode functions
    print("Encoding functions...")
    function_embeddings = encode_functions(functions, document_model)

    # Step 4: Normalize embeddings and create FAISS index
    print("Creating FAISS index...")
    faiss_index = create_faiss_index(function_embeddings)

    if faiss_index is None:
        print("Failed to create FAISS index. Exiting.")
        return

    # Step 5: Encode root cause
    root_cause = """   RootCause """
    root_cause_embedding = document_model.encode(root_cause)

    # Normalize root cause embedding for cosine similarity
    root_cause_embedding = normalize(root_cause_embedding.reshape(1, -1), axis=1, norm='l2').astype('float32')

    # Step 6: Use FAISS to find the most similar functions
    print("Finding similar functions using FAISS...")
    distances, similar_indices = faiss_index.search(root_cause_embedding, len(functions))

    # Step 7: Save the ranking to a text file
    ranking = sorted(zip(similar_indices[0], distances[0]), key=lambda x: x[1])

    output_file_path = "/yourPath"
    with open(output_file_path, 'w') as output_file:
        for idx, score in ranking:
            function = functions[idx]
            output_file.write(
                f"Function ID: {function.unique_id}, Name: {function.function_name}, Similarity Score: {score}\n")

    print(f"Function ranking saved to {output_file_path}")


if __name__ == '__main__':
    main()
