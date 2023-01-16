from score_calculators import *


def testing_metrics(evaluation_data_json, y=[]):
    for topic in evaluation_data_json:
        for turn in topic["turn"]:
            y.append({
                "turn": turn["number"],
                "rewritten_query": turn["manual_rewritten_utterance"],
                "manual_rewritten_query": turn["manual_rewritten_utterance"],

                "manual_returned_paragraph_id": turn["manual_canonical_result_id"],
                "returned_paragraphs": [(10, turn["manual_canonical_result_id"])] + [(-10, str(i)) for i in range(999)],
            })

    calculate_scores(y,"testing_metrics","testing_metrics")
