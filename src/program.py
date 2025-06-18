from typing import List, Dict
from collections import defaultdict

class Program:
    def __init__(self, citation: Dict[str, Dict[int, List[int]]], operations = None):
        if citation is not None:
            self.citation = citation
        else:
            self.citation = defaultdict(lambda: defaultdict(list))

        if operations is not None:
            self.operations = operations
        else:
            self.operations = defaultdict(list)

    def add_operation(self, sentence_index: int, name: str, sentences: List[str], output: str, instruction: str=None, candidates: List[str]=None):
        citations = defaultdict(list)
        for sent in sentences:
            if sent in self.citation:
                citation = self.citation[sent]
                for k,v in citation.items():
                    citations[k] += v
            else:
                print(f"Warning: {sent} not found in citation")
        citations = {k: list(set(v)) for k,v in citations.items()}
        
        self.operations[int(sentence_index)].append(
            {
                "name": name,
                "sentences": sentences,
                "instruction": instruction,
                "output": output,
                "citations": citations,
                "candidates": candidates,
            }
        )

        self.citation[output] = citations
    
    def to_sentences(self, add_citation=False, add_sentence_citation=False, sentid2globalid=None) -> str:
        sentences = []
        keys = sorted(list(self.operations.keys()))
        for i in keys:
            sent = self.operations[i]
            if len(sent) == 0:
                continue
            cur_sent = sent[-1]["output"]
            if add_citation or add_sentence_citation:
                ends_with_period = False
                if cur_sent.endswith("."):
                    ends_with_period = True
                    cur_sent = cur_sent[:-1]
                if add_citation:
                    for doc_citation, sent_citations in sent[-1]["citations"].items():
                        cur_sent += f"[{doc_citation}]"
                elif add_sentence_citation:
                    citations = set()
                    for doc_citation, sent_citations in sent[-1]["citations"].items():
                        for sent_citation in sent_citations:
                            if sentid2globalid is not None:
                                if (int(doc_citation), sent_citation) in sentid2globalid:
                                    citations.add(sentid2globalid[(int(doc_citation), sent_citation)])
                                else:
                                    print(f"Warning: {doc_citation} {sent_citation} not found in sentid")
                            else:
                                citations.add(sent_citation)
                    citations = sorted(list(citations))
                    for si in citations:
                        cur_sent += f"[{si}]"
                if ends_with_period:
                    cur_sent += "."
            sentences.append(cur_sent)
        
        return sentences

    def __repr__(self) -> str:
        return f"ProgramSentence(calls={self.operations})"