import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Initialize CodeBERT and Sentence-BERT models
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model_comments = SentenceTransformer("all-MiniLM-L6-v2")


import javalang
import json
import re


class Function:
    def __init__(self, function_name, comments, function_code):
        self.function_name = function_name
        self.comments = comments
        self.function_code = function_code


def extract_comments(java_code):
    """
    Extracts all comments in the Java code.
    """
    comments = re.findall(r'//.*?$|/\*.*?\*/', java_code, re.DOTALL | re.MULTILINE)
    return comments


def extract_functions_and_variables(java_code):
    """
    Extracts function name, comments, and function body (without comments) from the Java code.
    """
    tree = javalang.parse.parse(java_code)
    functions = []

    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            function_name = node.name
            method_code = extract_method_code(java_code, node)

            # Extract comments both before and inside the method
            preceding_comments = extract_comments_before_method(java_code.splitlines(), node.position.line - 1)
            inline_comments = extract_inline_comments(method_code)

            # Combine preceding and inline comments
            all_comments = preceding_comments + "\n" + inline_comments if inline_comments else preceding_comments

            # Remove all comments from the method code
            method_code_without_comments = remove_comments_from_code(method_code)

            # Create a Function object and append to the list
            function_obj = Function(function_name, all_comments.strip(), method_code_without_comments)
            functions.append(function_obj)

    return functions


def extract_method_code(java_code, method_node):
    """
    Extracts the method code from the Java source given the method's AST node.
    """
    code_lines = java_code.splitlines()
    start_line = method_node.position.line - 1  # Convert to zero-indexed
    open_braces = 0
    in_method_body = False
    method_lines = []

    for i in range(start_line, len(code_lines)):
        line = code_lines[i]
        # Check for opening and closing braces
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

    return "\n".join(method_lines)


def extract_comments_before_method(code_lines, start_line):
    """
    Extracts comments located before the method declaration.
    """
    comments = []
    for i in range(start_line - 1, -1, -1):
        line = code_lines[i].strip()
        if not line.startswith("//") and not line.startswith("/*") and not line.startswith("*"):
            break
        comments.insert(0, code_lines[i])
    return "\n".join(comments) + "\n" if comments else ""


def extract_inline_comments(method_code):
    """
    Extracts inline comments from the method code.
    """
    inline_comments = re.findall(r'//.*?$|/\*.*?\*/', method_code, re.DOTALL | re.MULTILINE)
    return "\n".join(inline_comments)


def find_method_end(java_code, start_line):
    """
    Finds the end of a method by counting braces.
    """
    code_lines = java_code.splitlines()
    open_braces = 0
    for i in range(start_line, len(code_lines)):
        line = code_lines[i]
        open_braces += line.count('{')
        open_braces -= line.count('}')
        if open_braces == 0:
            return i
    return len(code_lines) - 1


def remove_comments_from_code(code):
    """
    Removes all comments (single-line and multi-line) from the given method code.
    """
    # Remove both inline (//) and block (/* */) comments
    code_without_comments = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)
    # Remove any empty lines caused by removing comments
    code_without_comments = "\n".join([line for line in code_without_comments.splitlines() if line.strip()])
    return code_without_comments


def save_to_jsonl(functions, filename):
    """
    Saves function data (name, comments, and code without comments) to a JSONL file.
    """
    with open(filename, 'w') as f:
        for function in functions:
            function_dict = {
                "function_name": function.function_name,
                "comments": function.comments,
                "function_code": function.function_code  # Store the function body without comments
            }
            f.write(json.dumps(function_dict) + '\n')



# Function to get embeddings from CodeBERT (for function code)
def get_code_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use the [CLS] token
    return embedding


# Function to get embeddings from Sentence-BERT (for comments)
def get_comment_embedding(text):
    return model_comments.encode(text)


# Function to calculate cosine similarity
def cosine_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]


# Function to update the weights a and b based on the similarity difference
def update_weights(a, b, sim_func, sim_comment, target_sim, lr=0.01):
    # Compute weighted similarity
    current_sim = a * sim_func + b * sim_comment

    # Calculate gradients (difference between current similarity and target similarity)
    grad_a = (target_sim - current_sim) * sim_func
    grad_b = (target_sim - current_sim) * sim_comment

    # Debugging: Print the gradients
    # print(f"Gradient a: {grad_a:.6f}, Gradient b: {grad_b:.6f}")

    # Update weights with gradient descent step
    a = a + lr * grad_a
    b = b + lr * grad_b

    # Normalize a and b to sum to 1
    a = max(0, min(1, a))
    b = 1 - a

    return a, b, current_sim


# Read functions from JSONL file
def read_functions_from_jsonl(file_path):
    functions_data = []
    with open(file_path, 'r') as f:
        for line in f:
            functions_data.append(json.loads(line))
    return functions_data


# Function to find the target function by its name
def find_target_function(target_function_name, functions_data):
    for function in functions_data:
        if function['function_name'] == target_function_name:
            return function
    return None


# Function to calculate weighted similarity and update weights iteratively
def calculate_weighted_similarity(target_function, functions_data, a, b, lr, epochs, target_similarity=0.9):
    target_code_embedding = get_code_embedding(target_function['function_code'])
    target_comment_embedding = get_comment_embedding(target_function['comments'])

    most_similar_func_idx = -1
    highest_similarity = -1  # Track the highest similarity

    for epoch in range(epochs):
        for i, function in enumerate(functions_data):
            if function['function_name'] == target_function['function_name']:
                continue  # Skip the target function itself

            # Get the function code and comment embeddings
            function_code_embedding = get_code_embedding(function['function_code'])
            function_comment_embedding = get_comment_embedding(function['comments'])

            # Calculate similarity for function bodies and comments
            sim_func = cosine_sim(target_code_embedding, function_code_embedding)
            sim_comment = cosine_sim(target_comment_embedding, function_comment_embedding)

            # Update weights based on similarity
            a, b, current_similarity = update_weights(a, b, sim_func, sim_comment, target_similarity, lr=lr)

            # Track the function with the highest similarity
            if current_similarity > highest_similarity:
                highest_similarity = current_similarity
                most_similar_func_idx = i

        print(f"Epoch {epoch + 1}: Updated weights -> a: {a:.2f}, b: {b:.2f}")

    # Return the most similar function and updated weights
    return a, b, most_similar_func_idx


# Get the top 10 similar functions after weight updates
def get_top_10_similar_functions(target_function, functions_data, a, b, lr, epochs):
    # First, run the weight update process
    final_a, final_b, most_similar_func_idx = calculate_weighted_similarity(target_function, functions_data, a=a, b=b,
                                                                            lr=lr, epochs=epochs)

    # Now calculate the similarities for all functions with the final weights
    target_code_embedding = get_code_embedding(target_function['function_code'])
    target_comment_embedding = get_comment_embedding(target_function['comments'])

    similarities = []
    for function in functions_data:
        if function['function_name'] == target_function['function_name']:
            continue  # Skip the target function itself

        function_code_embedding = get_code_embedding(function['function_code'])
        function_comment_embedding = get_comment_embedding(function['comments'])

        # Compute similarities for function code and comments
        sim_func = cosine_sim(target_code_embedding, function_code_embedding)
        sim_comment = cosine_sim(target_comment_embedding, function_comment_embedding)

        # Compute weighted similarity using the final weights
        weighted_sim = final_a * sim_func + final_b * sim_comment

        # Store the similarity along with the function details
        similarities.append({
            'function_name': function['function_name'],
            'function_code': function['function_code'],
            'comments': function['comments'],
            'similarity': weighted_sim
        })

    # Sort by similarity in descending order and get the top 10
    top_10_functions = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:]
    return top_10_functions


# Main function to process everything
def main():
    # Define the target function name (you will provide the target function's name here)
    target_function_name = "Buggy Function Name"
    java_file_path = 'javaFilePatch'

    # Read the Java file
    with open(java_file_path, 'r') as file:
        java_code = file.read()

    # Extract function details (function name, comments, function body without comments)
    functions = extract_functions_and_variables(java_code)

    file_path = 'function Path'

    # Save to JSONL file
    save_to_jsonl(functions, file_path)
    functions_data = read_functions_from_jsonl(file_path)

    # Find the target function by name
    target_function = find_target_function(target_function_name, functions_data)
    if target_function is None:
        print(f"Target function '{target_function_name}' not found in the JSONL file.")
        return

    # Get the top 10 most similar functions with dynamic weight updates
    top_10_functions = get_top_10_similar_functions(target_function, functions_data, a=0.5, b=0.5, lr=0.01, epochs=7)

    # Output the top 10 functions
    output_file_path = "/Path"

    # Write the results to the text file
    with open(output_file_path, 'w') as f:
        for idx, func in enumerate(top_10_functions, 1):
            f.write(f"Rank {idx}:\n")
            f.write(f"Function Name: {func['function_name']}\n")
            f.write(f"Similarity: {func['similarity']:.4f}\n")
            f.write(f"Function Code:\n{func['function_code']}\n")
            f.write(f"Comments:\n{func['comments']}\n\n")

    print(f"Results have been written to {output_file_path}")


if __name__ == "__main__":
    main()
