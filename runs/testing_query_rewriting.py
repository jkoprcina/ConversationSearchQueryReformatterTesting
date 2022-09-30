import neptune.new as neptune

from constants import API_TOKEN, PROJECT
from conversation import Conversation
from choosing_query_rewriting import query_rewriting
from score_calculators import calculate_rouge_score, calculate_bleu_score


def testing_query_rewriting(ms_marco_df, car_df, evaluation_data_json, T5, data_usage_type, summarizer, T5_config):
    y = []
    for topic in evaluation_data_json:
        conv = Conversation()
        for turn in topic["turn"]:
            conv.add_query(turn["raw_utterance"])

            if turn["number"] > 1:
                query_rewriting(data_usage_type, conv, T5, summarizer, T5_config)
            else:
                conv.add_to_rewritten_queries(turn["raw_utterance"])

            if turn["automatic_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_df.loc[ms_marco_df['id'] == str(turn["automatic_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_df.loc[car_df['id'] == str(turn["automatic_canonical_result_id"][4:])]["paragraph"].to_string()

            conv.add_paragraph(paragraph)

            y.append({
                "turn": turn["number"],
                "reformatted_query": conv.rewritten_queries[-1],
                "automatic_reformatted_query": turn["automatic_rewritten_utterance"]
            })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["data_usage_type"] = data_usage_type
    run["run_type"] = "testing_query_rewriting"
    run["T5_query_rewriter"] = T5_config
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
