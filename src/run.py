import json
import argparse
from model import Model
from tqdm import tqdm
import time
import re
from program import Program
from util import parse_function_calls, run_module, is_irrelevant, split_sent
from collections import defaultdict
import string
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--num_docs", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_sent", type=int, default=1)
    parser.add_argument("--add_citation", action="store_true", default=False)
    parser.add_argument("--module_with_question", action="store_true", default=False)
    parser.add_argument("--fewshot_examples", type=str, default=None)
    parser.add_argument("--max_fewshot", type=int, default=1)
    args = parser.parse_args()

    data = json.load(open(args.data))
    if os.path.exists(args.output):
        print("Output file already exists. Loading existing data.")
        data = json.load(open(args.output))

    model = Model(args.model)

    fewshot_examples = json.load(open(args.fewshot_examples)) if args.fewshot_examples is not None else None

    prompt = open(args.prompt).read().strip()

    for k, dat in tqdm(enumerate(data), total=len(data)):
        if "output" in dat and dat["output"] is not None:
            continue

        context = dat["context"]
        if "summaries_individual" in dat:
            context = dat["summaries_individual"]
        
        # if using summ in the prev step inherent the citations
        operations_individual = dat["operations_individual"] if "operations_individual" in dat else None
        
        if type(context) is list:
            context = context[:args.num_docs]
        else:
            context = [context]
        
        sent_id = 0
        context_str = ""
        citation, id2sent = defaultdict(lambda:defaultdict(list)), dict()
        # use the generated sentences if available to prevent sent_tokenization mismatch
        if operations_individual is not None:
            for i, operations in enumerate(operations_individual): # loop each document
                doc_lines = []
                if operations is None or len(operations) == 0:
                    continue
                
                keys = sorted([int(k) for k in operations.keys()])
                for j in keys: # loop each output sentence
                    if len(operations[str(j)]) == 0:
                        continue
                    
                    operation = operations[str(j)][-1] # last one
                    for doc_id, sent_ids in operation["citations"].items():
                        citation[operation["output"]][doc_id].extend(sent_ids)
                    doc_lines.append(f"S{sent_id+1}: {operation['output']}")
                    id2sent[f"S{sent_id+1}"] = operation["output"]
                    sent_id += 1
                doc_lines = "\n".join(doc_lines)
                if doc_lines:
                    context_str += f"Passage {i+1}:\n{doc_lines}\n\n"
        else:
            for i, ctxt in enumerate(context):
                doc_lines = []

                if ctxt is None or ctxt == "":
                    continue
                for j, sent in enumerate(split_sent(ctxt)):
                    sent = sent.replace("\n", "")
                    if sent.startswith("- "):
                        sent = sent[2:]
                    doc_lines.append(f"S{sent_id+1}: {sent}")
                    citation[sent][i+1].append(j)
                    id2sent[f"S{sent_id+1}"] = sent
                    sent_id += 1
                doc_lines = "\n".join(doc_lines)
                if doc_lines:
                    context_str += f"Passage {i+1}:\n{doc_lines}\n\n"
        
        context = context_str

        # use the first document if all irrelevant
        if context == "":
            doc_lines = []
            next_aviailable_context = None
            for ctxt in dat["context"]:
                if ctxt != "":
                    next_aviailable_context = ctxt
                    break
            if next_aviailable_context is None:
                continue
            for j, sent in enumerate(split_sent(next_aviailable_context)):
                sent = sent.replace("\n", "")
                if sent.startswith("- "):
                    sent = sent[2:]
                doc_lines.append(f"S{sent_id+1}: {sent}")
                citation[sent][1].append(j)
                id2sent[f"S{sent_id+1}"] = sent
                sent_id += 1
            doc_lines = "\n".join(doc_lines)
            context_str += f"Passage 1:\n{doc_lines}\n\n"
            context = context_str


        program = Program(citation)

        if fewshot_examples is not None:
            # prepare fewshot examples
            fewshot_str = ""
            fewshot_i = 0

            for kk, vv in fewshot_examples.items():
                if int(kk) != int(k): # use the ones that are different ids
                    fewshot_str += f"Example {fewshot_i+1}:\n"
                    fewshot_str += """Question:
{question}

Document:
{context}

Output:
{output}""".format(question=vv["question"], context=vv["context"], output=vv["output"])
                    fewshot_str += "\n\n"
                    fewshot_i += 1
                    if fewshot_i >= args.max_fewshot:
                        break
            cur_prompt = prompt.format(context=context, question=dat["question"], examples=fewshot_str)
        else:
            cur_prompt = prompt.format(context=context, question=dat["question"])

        res = model.run(cur_prompt)
        
        # print(cur_prompt)
        # print("---"*5)
        # print(res)
        # print("==="*10)
        
        if res is None:
            res = ""
        
        dat["program_raw"] = res
        
        res = res.split("\n")

        for i, r in enumerate(res):
            r = r.lstrip("- ")
            r = r.replace("**", "")
            parsed_calls = parse_function_calls(r)

            generated_result = []
            for name, sentence_ids, instruction in parsed_calls:
                # change sentences to the actual sentences
                sentences = [id2sent[str(sent)] for sent in sentence_ids if str(sent) in id2sent]
                if len(sentences) != len(sentence_ids):
                    print("ERROR: sentence not found")
                    continue
                # print(model, name, sentences, instruction)
                
                output = run_module(model, name, sentences, instruction=instruction, question=dat["question"] if args.module_with_question else None,  max_tokens=1000)
                if output is None:
                    continue
                id2sent[str((name, sentence_ids, instruction))] = output
                program.add_operation(i, name, sentences, output, instruction=instruction)
        
        program_sentences = program.to_sentences(add_citation=args.add_citation)
        # print(program.to_sentences(add_sentence_citation=True))
        # print(" ".join(program_sentences))
        program_sentences = [s for s in program_sentences if s is not None]
        dat["citation"] = program.citation
        dat["operations"] = program.operations
        dat["output"] = " ".join(program_sentences)

        program_sentences = program.to_sentences(add_sentence_citation=args.add_citation)
        dat["output_sentence_level"] = " ".join(program_sentences)
        
        with open(args.output, "w") as f:
            json.dump(data, f, indent=4)
        # break


if __name__ == "__main__":
    main()