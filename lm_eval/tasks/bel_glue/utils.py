import re
import string
from collections import defaultdict

import numpy as np


UNER_EN_PROMPT = """Identify entities in Belarusian text. When asked about "чалавек", "арганізацыя", "месца", or "іншая іменаваная сутнасць", find the matching entity in the given text. Always provide an answer. The answer must have a single entity. Output it the same as it is mentioned in the text.

What entity of type {label} is found in the following text?
{text}

Entity of type {label}:"""


UNER_BE_PROMPT = """Твая задача – знаходзіць іменаваныя сутнасці ў тэксце на беларускай мове. Калі пытаюць пра іменаваную сутнасць тыпу "чалавек", "арганізацыя", "месца" або "іншая іменаваная сутнасць", трэба знайсці адпаведную іменаваную сутнасць у дадзеным тэксце і вывесці як адказ. Заўсёды давай адказ. Адказ павінен утрымліваць адзіную сутнасць. Падавай яе так, як згадана ў тэксце.

Якая іменаваная сутнасць тыпу {label} ёсць ў наступным тэксце?
{text}

Іменаваная сутнасць тыпу {label}:"""


def doc_to_text_en(doc):
    text = doc["text"]
    label = doc["label"]
    text = UNER_EN_PROMPT.format(text=text, label=label)
    return text

def doc_to_text_be(doc):
    text = doc["text"]
    label = doc["label"]
    text = UNER_BE_PROMPT.format(text=text, label=label)
    return text


def doc_to_target(doc):
    # TODO: dump the answers with json.dumps before loading to HF
    # return json.loads(doc["answers"])
    answers = eval(doc["answers"])
    return list(set(answers))


def intersection(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    intersection = set(predictions) & set(references)
    score = int(len(intersection) > 0)

    return {"intersection": score}


def at_least_one_match(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]

    scores = [int(pred in gold) for gold, pred in zip(golds, preds)]

    return sum(scores) / len(scores)


# TO DELETE EVERYTHING BELOW

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
