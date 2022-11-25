import neptune.new as neptune
from constants import API_TOKEN, PROJECT
from score_calculators import *


def testing_metrics(ms_marco_df, car_df, evaluation_data_json):
    y = []
    for topic in evaluation_data_json:
        for turn in topic["turn"]:
            if turn["manual_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_df.loc[ms_marco_df['id'] == str(turn["manual_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_df.loc[car_df['id'] == str(turn["manual_canonical_result_id"][4:])]["paragraph"].to_string()

            if turn["number"] > 1:
                y.append({
                    "turn": turn["number"],

                    "returned_paragraph_id": [json.loads(x[1])["id"] for x in reranked_paragraphs],  #####working on this
                    "manual_returned_paragraph_id": turn["manual_canonical_result_id"],
                    "returned_paragraph": paragraph,

                    "reformatted_query": turn["manual_rewritten_utterance"],
                    "automatic_reformatted_query": turn["manual_rewritten_utterance"]
                })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["run_type"] = "manual"
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
    calculate_ndcg3_score(y, run)
