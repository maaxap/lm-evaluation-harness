from collections import defaultdict


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


# Source of the code below can be found here:
# https://github.com/sighsmile/conlleval

def _split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)


def _is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = _split_tag(prev_tag)
    prefix2, chunk_type2 = _split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def _is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = _split_tag(prev_tag)
    prefix2, chunk_type2 = _split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def _calc_metrics(tp, p, t):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, fb1


def _count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = _split_tag(true_tag)
        _, pred_type = _split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = _is_chunk_end(prev_true_tag, true_tag)
            pred_end = _is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = _is_chunk_start(prev_true_tag, true_tag)
        pred_start = _is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def _get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    # compute overall precision, recall and FB1 (default values are 0.0)
    _, _, f1 = _calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    acc = sum_correct_counts / sum_true_counts
    acc_non0 = nonO_correct_counts / nonO_true_counts if nonO_true_counts != 0 else 0.0
    res = (acc, acc_non0, f1)

    return res


def conlleval_f1(items):
    golds, preds = items

    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, _ = _count_chunks(golds, preds)
    _, _, f1 = _get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts)

    return f1


def conlleval_acc(items):
    golds, preds = items

    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, _ = _count_chunks(golds, preds)
    acc, _, _ = _get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts)

    return acc


def conlleval_acc_non0(items):
    golds, preds = items

    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, _ = _count_chunks(golds, preds)
    _, acc_non0, _ = _get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts)

    return acc_non0
