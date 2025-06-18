import ast
import json
import sys
import tqdm
from model import Model
from collections import defaultdict
import string

import spacy
nlp = spacy.load("en_core_web_sm")

def split_sent(text):
    sents = []
    for t in text.split("\n"):
        for x in nlp(t).sents:
            sents.append(x.text.strip())
    return sents

# Define the valid functions and their argument constraints.
# For paraphrase and compression: exactly 1 positional argument.
# For fusion: at least 2 positional arguments.
VALID_FUNCTIONS = {
    "paraphrase": {"min_args": 1, "max_args": 1},
    "compression": {"min_args": 1, "max_args": 1},
    "fusion": {"min_args": 2, "max_args": None},  # None means no maximum limit.
}

function2prompts = {
    "paraphrase": open("prompts/paraphrase.txt").read(),
    "compression": open("prompts/compression.txt").read(),
    "fusion": open("prompts/fusion.txt").read(),
}

function2prompts_with_question = {
    "paraphrase": open("prompts/paraphrase_with_question.txt").read(),
    "compression": open("prompts/compression_with_question.txt").read(),
    "fusion": open("prompts/fusion_with_question.txt").read(),
}

def run_module(model, module_name, sentences, instruction=None, question=None, max_tokens=100):
    if question is None:
        prompt = function2prompts[module_name]
    else:
        prompt = function2prompts_with_question[module_name]
    # if instruction is None, remove instruction part, otherwise include instruction
    if instruction is None:
        prompt = prompt.replace("**Instruction**:\n[[INSTRUCTION]]\n", "")
    else:
        prompt = prompt.replace("[[INSTRUCTION]]", instruction)
    if question is not None:
        prompt = prompt.replace("[[QUESTION]]", question)
    sentence_str = ""
    for sent in sentences:
        sentence_str += sent.replace("\n", "").strip() + "\n"
    prompt = prompt.replace("[[SENTENCE]]", sentence_str)
    # print(prompt)
    output = model.run(prompt, max_tokens=max_tokens)
    return output

def is_irrelevant(text):
    if text is None:
        return True
    text = text.strip()
    for t in text.split("\n"):
        if t.startswith("- "):
            t = t[2:]
        t = t.lower()
        # remove punctuations
        t = t.translate(str.maketrans('', '', string.punctuation))
        if t == "irrelevant" or t == "not relevant":
            return True
    return False

def parse_node_dfs(node, results):
    """
    Recursively parses an AST node and appends any function call tuple 
    to the results list in a depth-first order (inner-most calls first).
    Returns the parsed value of the node.
    
    If the node is a function call, it returns a tuple of the form:
      (function_name, [parsed_arguments], instruction)
    """
    if isinstance(node, ast.Call):
        # Process all positional arguments first.
        parsed_args = []
        for arg in node.args:
            parsed_arg = parse_node_dfs(arg, results)
            parsed_args.append(parsed_arg)
        
        # Process keyword arguments (only "instruction" is allowed).
        instruction = None
        for keyword in node.keywords:
            if keyword.arg == "instruction":
                instruction = parse_node_dfs(keyword.value, results)
            else:
                raise ValueError(f"Unexpected keyword argument: {keyword.arg}")
        
        # Ensure the function name is a simple identifier.
        if not isinstance(node.func, ast.Name):
            raise ValueError("Function name must be a simple identifier.")
        func_name = node.func.id
        
        # Validate the function name.
        if func_name not in VALID_FUNCTIONS:
            raise ValueError(f"Invalid function name: {func_name}")
        
        # Validate the number of positional arguments.
        func_spec = VALID_FUNCTIONS[func_name]
        if len(parsed_args) < func_spec["min_args"]:
            raise ValueError(f"{func_name} requires at least {func_spec['min_args']} positional argument(s).")
        if func_spec["max_args"] is not None and len(parsed_args) > func_spec["max_args"]:
            raise ValueError(f"{func_name} accepts at most {func_spec['max_args']} positional argument(s).")
        
        # Create the tuple for the current function call.
        call_tuple = (func_name, parsed_args, instruction)
        # Append the current call after processing its inner calls.
        results.append(call_tuple)
        return call_tuple

    elif isinstance(node, ast.Name):
        return node.id

    elif isinstance(node, ast.Constant):
        return node.value

    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")

def parse_function_calls(call_string):
    """
    Parse a string containing one or more function calls into a list of tuples.
    Each tuple is of the form (function_name, [arguments], instruction) and 
    the inner-most function calls are returned first (depth-first order).
    """
    # first remove extra characters
    call_string = call_string.replace("python","").replace("`","").strip()
    try:
        tree = ast.parse(call_string, mode="exec")
    except Exception as e:
        # try to fix the commen error of incorrect nesting
        call_string = call_string.replace(")), instruction", "), instruction")
        try:
            tree = ast.parse(call_string, mode="exec")
        except Exception as e_:
            # try to fix multple paranthesis
            call_string = call_string.replace("))", ")")
            try:
                tree = ast.parse(call_string, mode="exec")
            except Exception as e__:
                print("Error parsing string:", e, e_, e__, call_string)
                return []

    results = []
    # Process each expression in the module.
    for node in tree.body:
        if isinstance(node, ast.Expr):
            try:
                parse_node_dfs(node.value, results)
            except Exception as e:
                print("Error:", e)
        else:
            print("Error: Not an expression:", ast.dump(node))
    return results