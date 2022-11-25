import rouge
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import ndcg_score
import numpy as np
import json


def calculate_rouge_score(y, run):
    precisions, recalls, fmeasures = [], [], []
    evaluator = rouge.Rouge(metrics=['rouge-l', ], max_n=4, limit_length=True, length_limit=100, length_limit_type='words', apply_avg='Avg', alpha=0.5, weight_factor=1.2, stemming=True)

    for i in range(1, 12):
        temporary_y = []
        for x in y:
            if x["turn"] == i + 1:
                temporary_y.append({"rewritten_query": x["rewritten_query"], "manual_rewritten_query": x["manual_rewritten_utterance"]})

        scores = evaluator.get_scores([pair['rewritten_query'] for pair in temporary_y], [pair['manual_rewritten_query'] for pair in temporary_y])
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            precision = 100 * results['p']
            recall = 100 * results['r']
            fmeasure = 100 * results['f']
            print("Examples of queries in turns " + str(i + 1) + " :" + str(len(temporary_y)))
            print('\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', precision, 'R', recall, 'F1', fmeasure))
            precisions.append(precision)
            recalls.append(recall)
            fmeasures.append(fmeasure)
            run["r_s_" + str(i + 1) + " f_measure"] = format(fmeasure, ".2f")

    run["sum_r_s f_measure"] = format(sum(fmeasures) / 12, ".2f")
    return precisions, recalls, fmeasures


def calculate_bleu_score(y, run):
    for i in range(1, 12):
        temporary_y = []
        for x in y:
            if x["turn"] == i + 1:
                temporary_y.append({"rewritten_query": x["rewritten_query"], "manual_rewritten_query": x["manual_rewritten_utterance"]})

        hypothesis = [word_tokenize(query["rewritten_query"]) for query in temporary_y]
        references = [word_tokenize(query["manual_rewritten_query"]) for query in temporary_y]

        bleu_score = sum([sentence_bleu([single_ref], single_hypo) for single_ref, single_hypo in zip(references, hypothesis)]) / len(hypothesis)

        print("The BLEU score for turn " + str(i + 1) + " is : " + str(bleu_score))
        run["b_s_" + str(i + 1)] = format(bleu_score * 100, ".2f")

    hypothesis = [word_tokenize(query["rewritten_query"]) for query in y if query["turn"] > 1]
    references = [word_tokenize(query["manual_rewritten_query"]) for query in y if query["turn"] > 1]

    bleu_score = sum([sentence_bleu([single_ref], single_hypo) for single_ref, single_hypo in zip(references, hypothesis)]) / len(hypothesis)
    run["sum_b_s"] = format(bleu_score * 100, ".2f")

    print("The BLEU score for all turns combined is : " + str(bleu_score))


def calculate_ndcg3_score(y, run):
    y = clean_data_for_ndcg(y)

    for i in range(13):
        temporary_y = []
        for line in y:
            if line["turn"] == i + 1:
                temporary_y.append(line)

        ndcg = calculate_ndcg(temporary_y)
        print("The NDCG score for turn " + str(i + 1) + " is : " + str(ndcg))
        run["b_s_" + str(i + 1)] = ndcg

    ndcg = calculate_ndcg(y)
    print("The average NDCG score is " + str(ndcg))
    run["ndcg3_score"] = ndcg


def clean_data_for_ndcg(y):
    new_y = []
    for i in range(len(y)):
        new_y.append({
            "turn": y[i]["turn"],
            "correct_id": y[i]["manual_canonical_result_id"],
            "returned_paragraphs": [{"id": json.loads(x[1])["id"], "score": x[0]} for x in y[i]["returned_paragraphs"]]
        })
    for line in new_y:
        for pair in line["returned_paragraphs"]:
            if len(pair["id"]) <= 30:
                pair["id"] = "MARCO_" + pair["id"]
            elif len(pair["id"]) == 40:
                pair["id"] = "CAR_" + pair["id"]
            else:
                raise Exception("Something is wrong with the returned ID's in ndcg@3 calculation")
    return new_y


def calculate_ndcg(y):
    ndcg3_sum = 0
    for line in y:
        correct_id = line["correct_id"]
        returned_paragraphs = line["returned_paragraphs"]

        y_true = np.asarray([[0 for i in range(1000)]])
        for i in range(len(returned_paragraphs)):
            if returned_paragraphs[i]["id"] == correct_id:
                y_true[0][i] = 3
        y_scores = [[pair["score"] for pair in returned_paragraphs]]
        ndcg3_sum += ndcg_score(y_true, y_scores, k=1000)
    return ndcg3_sum / len(y)


def calculate_counts(y, run):
    average_turn_word_count, average_turn_token_count = [], []
    for i in range(13):
        temporary_y = []
        for x in y:
            if x["turn"] == i + 1:
                temporary_y.append({"word_count": x["word_count"], "token_count": x["token_count"]})

        average_turn_word_count.append(sum([row.get("word_count") for row in temporary_y]) / len(temporary_y))
        average_turn_token_count.append(sum([row.get("token_count") for row in temporary_y]) / len(temporary_y))

    print("The average word_count is : " + str(average_turn_word_count))
    print("The average token_count is : " + str(average_turn_token_count))
    run["average_word_count"] = average_turn_word_count
    run["average_token_count"] = average_turn_token_count
    return True
