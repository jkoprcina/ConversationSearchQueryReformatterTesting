import neptune.new as neptune

from constants import API_TOKEN, PROJECT
from score_calculators import calculate_rouge_score, calculate_bleu_score, calculate_ndcg3_score


def manual_run(ms_marco_df, car_df, evaluation_data_json):
    y = []
    for topic in evaluation_data_json:
        for turn in topic["turn"]:
            if turn["automatic_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_df.loc[ms_marco_df['id'] ==
                                            str(turn["automatic_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_df.loc[car_df['id'] ==
                                       str(turn["automatic_canonical_result_id"][4:])]["paragraph"].to_string()

            if turn["number"] > 1:
                y.append({
                    "turn": turn["number"],
                    "returned_paragraph": paragraph,
                    "automatic_returned_paragraph": paragraph,
                    "reformatted_query": turn["automatic_rewritten_utterance"],
                    "automatic_reformatted_query": turn["automatic_rewritten_utterance"]
                })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["run_type"] = "manual"
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)