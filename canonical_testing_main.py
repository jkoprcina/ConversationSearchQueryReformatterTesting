from rouge_score import *
from sentence_extractors import *


def canonical_main_loop(ms_marco_collection_df, car_collection_df, evaluation_data_json, T5, sentence_extractor):
    y = []
    for topic in evaluation_data_json:
        turn_number = 0
        context = []
        paragraph = ""
        for turn in topic["turn"]:
            turn_number += 1
            query = turn["raw_utterance"]

            if turn["number"] > 1:
                sentence = h2oloo_find_sentence(paragraph, query)
                query = T5.predict(query + " " + sentence + " " + ' '.join(context))[0]

            if turn["automatic_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_collection_df.loc[ms_marco_collection_df['id'] == str(turn["automatic_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_collection_df.loc[car_collection_df['id'] == str(turn["automatic_canonical_result_id"][4:])]["paragraph"].to_string()

            context.append(str(query))
            y.append({"turn": turn["number"],
                      "reformatted_query": query,
                      "automatically_reformatted_query": turn["automatic_rewritten_utterance"]})

    precisions, recalls, fmeasures = calculate_rouge_score(y)
    print_rouge_score(precisions, recalls, fmeasures)
