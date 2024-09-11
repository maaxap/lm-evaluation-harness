UNER_PROMPT = """You are an NER model trained to label sequences using BIO tags. The entities you need to recognize are "LOC" for locations, "ORG" for organizations, "PER" for persons, and "OTH" for other entities. The input will be a list of words, and your task is to output a list with the corresponding tags. Use "B-" for the beginning of an entity, "I-" for inside an entity, and "O" for outside any entity. The output has to be a valid JSON list without any additional prefixes. Strictly follow the output format.

Examples:

Input: ["Apple", "is", "based", "in", "California", "."]
Output: ["B-ORG", "O", "O", "O", "B-LOC", "O"]

Input: ["Barack", "Obama", "was", "born", "in", "Hawaii", "."]
Output: ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O"]

Task: Label the following sequence.

Input: {tokens}
Output:"""


def doc_to_text(doc):
    tokens = doc["tokens"].split()
    text = UNER_PROMPT.format(tokens=tokens)
    return text


def doc_to_target(doc):
    labels = doc["labels"].split()
    return [labels]
