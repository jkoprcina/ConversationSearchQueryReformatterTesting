import rouge
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import neptune.new as neptune

PROJECT = "red-lion/Query-rewriting"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2MxYjc0MC1mODhiLTQ2ZDctOGVmNC1lOGE1ZTkyM2YzMDEifQ=="


def calculate_scores(y, sentence_extractor, canonical):
    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["sent_extra"] = sentence_extractor
    run["canonical"] = canonical
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)


def calculate_rouge_score(y, run):
    precisions, recalls, fmeasures = [], [], []
    evaluator = rouge.Rouge(metrics=['rouge-l', ], max_n=4, limit_length=True, length_limit=100,
                            length_limit_type='words', apply_avg='Avg', alpha=0.5, weight_factor=1.2, stemming=True)

    for i in range(2, 11):
        temporary_y = []
        for x in y:
            if x["turn"] == i:
                temporary_y.append({"reformatted_query": x["reformatted_query"],
                                    "automatic_reformatted_query": x["automatic_reformatted_query"]})

        scores = evaluator.get_scores([pair['reformatted_query'] for pair in temporary_y],
                                      [pair['automatic_reformatted_query'] for pair in temporary_y])
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            precision = 100 * results['p']
            recall = 100 * results['r']
            fmeasure = 100 * results['f']
            print("Examples of queries in turns " + str(i) + " :" + str(len(temporary_y)))
            print('\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', precision, 'R', recall, 'F1',
                                                                        fmeasure))
            precisions.append(precision)
            recalls.append(recall)
            fmeasures.append(fmeasure)
            run["rouge_score_" + str(i) + " f_measure"] = str(fmeasure)

    run["sum_of_rouge_scores f_measure"] = str(sum(fmeasures) / 9)
    return precisions, recalls, fmeasures


def calculate_bleu_score(y, run):
    for i in range(2, 11):
        temporary_y = []
        for x in y:
            if x["turn"] == i:
                temporary_y.append({"reformatted_query": x["reformatted_query"],
                                    "automatic_reformatted_query": x["automatic_reformatted_query"]})

        hypothesis = [word_tokenize(query["reformatted_query"]) for query in temporary_y]
        references = [word_tokenize(query["automatic_reformatted_query"]) for query in temporary_y]

        bleu_score = sum([sentence_bleu([single_ref], single_hypo)
                          for single_ref, single_hypo in zip(references, hypothesis)]) / len(hypothesis)

        print("The BLEU score for turn " + str(i) + " is : " + str(bleu_score))
        run["blue_score_" + str(i)] = bleu_score

    hypothesis = [word_tokenize(query["reformatted_query"]) for query in y]
    references = [word_tokenize(query["automatic_reformatted_query"]) for query in y]

    bleu_score = sum([sentence_bleu([single_ref], single_hypo)
                      for single_ref, single_hypo in zip(references, hypothesis)]) / len(hypothesis)
    run["sum_of_blue_scores"] = bleu_score

    print("The BLEU score for all turns combined is : " + str(bleu_score))
