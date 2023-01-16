import rouge
import nltk
nltk.download('stopwords')
import neptune.new as neptune
import numpy as np
from bleu import list_bleu
from sklearn.metrics import ndcg_score
import json

from constants import PROJECT, API_TOKEN


def calculate_scores(y, data_usage_type, T5_config):
    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["data_usage_type"] = data_usage_type
    run["T5_query_rewriter"] = T5_config
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
    calculate_and_log_ndcg3_score(y, run)


def calculate_rouge_score(y, run):
    evaluator = rouge.Rouge(metrics=['rouge-l', ], max_n=4, length_limit=1000, length_limit_type='words', apply_avg='Avg', alpha=0.5, weight_factor=1.2, stemming=True)
    for i in range(12):
        scores = evaluator.get_scores([query['rewritten_query'] for query in y if query["turn"] == i + 1], [query['manual_rewritten_query'] for query in y if query["turn"] == i + 1])
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            run["rouge_" + str(i + 1)] = format(100 * results["f"], ".2f")

    scores = evaluator.get_scores([query['rewritten_query'] for query in y], [query['manual_rewritten_query'] for query in y])
    run["rouge_sum"] = format((100 *  scores["rouge-l"]["f"]), ".2f")


def calculate_bleu_score(y, run):
    for i in range(12):
        hypothesis = [query["rewritten_query"] for query in y if query["turn"] == i + 1]
        references = [query["manual_rewritten_query"] for query in y if query["turn"] == i + 1]
        bleu_score = list_bleu([references], hypothesis, verbose=True, detok=True)
        run["bleu_" + str(i + 1)] = format(bleu_score, ".2f")

    hypothesis = [query["rewritten_query"] for query in y]
    references = [query["manual_rewritten_query"] for query in y]
    bleu_score = list_bleu([references], hypothesis, verbose=True, detok=True)
    run["bleu_sum"] = format(bleu_score, ".2f")


def calculate_and_log_ndcg3_score(messy_y, run):
    clean_y = []
    for turn in messy_y:
        clean_y.append({
            "turn": turn["turn"],
            "correct_id": turn["manual_returned_paragraph_id"],
            ##"returned_paragraphs": [{"id": x[1], "score": x[0]} for x in turn["returned_paragraphs"]]
            "returned_paragraphs": [{"id": json.loads(x[1])["id"], "score": x[0]} for x in turn["returned_paragraphs"]]
        })

    try:
        for line in clean_y:
            for pair in line["returned_paragraphs"]:
                if len(pair["id"]) == 40:
                    pair["id"] = "CAR_" + pair["id"]
                else:
                    pair["id"] = "MARCO_" + pair["id"]
    except:
        raise Exception("Something is wrong with the returned ID's in ndcg@3 calculation")

    for i in range(12):
        temporary_y = [line for line in clean_y if line["turn"] == i + 1]
        run["ndcg_" + str(i + 1)] = calculate_ndcg_score(temporary_y)
    run["ndcg3_sum"] = calculate_ndcg_score(clean_y)


def calculate_ndcg_score(temp_y):
    ndcg3_sum = 0
    for line in temp_y:
        correct_id = line["correct_id"]
        returned_paragraphs = line["returned_paragraphs"]

        y_true = np.asarray([[0 for i in range(1000)]])
        for i in range(len(returned_paragraphs)):
            if returned_paragraphs[i]["id"] == correct_id:
                y_true[0][i] = 3
        y_scores = [[pair["score"] for pair in returned_paragraphs]]
        ndcg3_sum += ndcg_score(y_true, y_scores, k=1000)
    return ndcg3_sum / len(temp_y)


def calculate_counts(y, run, word_count=[], token_count=[]):
    for i in range(12):
        temporary_y = [{"word_count": x["word_count"], "token_count": x["token_count"]} for x in y if x["turn"] == i + 1]
        word_count.append(format(sum([row.get("word_count") for row in temporary_y]) / len(temporary_y), ".2f"))
        token_count.append(format(sum([row.get("token_count") for row in temporary_y]) / len(temporary_y), ".2f"))
    run["average_word_count"] = word_count
    run["average_token_count"] = token_count


