import matplotlib.pyplot as plt
import rouge


def calculate_rouge_score(y):
    precisions, recalls, fmeasures = [], [], []
    evaluator = rouge.Rouge(metrics=['rouge-l', ], max_n=4, limit_length=True, length_limit=100,
                            length_limit_type='words', apply_avg='Avg', alpha=0.5, weight_factor=1.2, stemming=True)

    for i in range(1, 11):
        temporary_y = []
        for x in y:
            if x["turn"] == i:
                temporary_y.append({"reformatted_query": x["reformatted_query"],
                                    "automatically_reformatted_query": x["automatically_reformatted_query"]})

        scores = evaluator.get_scores([pair['reformatted_query'] for pair in temporary_y],
                                      [pair['automatically_reformatted_query'] for pair in temporary_y])
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
    return precisions, recalls, fmeasures


def print_rouge_score(p, r, fm):
    x = range(1, 11)
    plt.figure(figsize=(12, 6))
    plt.plot(x, p, label="precision")
    plt.plot(x, r, label="recall")
    plt.plot(x, fm, label="fmeasure")
    plt.legend()
    plt.show()

    print("Average stats are:")
    print("Precision: " + str(sum(p) / 10))
    print("Recall: " + str(sum(r) / 10))
    print("Fmeasure: " + str(sum(fm) / 10))