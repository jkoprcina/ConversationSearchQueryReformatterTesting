from sentence_extractors import h2oloo_find_sentence
from score import calculate_scores


def canonical_main_loop(ms_marco_df, car_df, evaluation_data_json, T5, paragraph_usage_option):
    y = []
    for topic in evaluation_data_json:
        turn_number = 0
        context = []
        paragraph = ""
        for turn in topic["turn"]:
            turn_number += 1
            query = turn["raw_utterance"]

            if turn["number"] > 1:
                if paragraph_usage_option == "h2oloo":
                    sentence = h2oloo_find_sentence(paragraph, query)
                elif paragraph_usage_option == "full paragraph":
                    sentence = paragraph
                else:
                    sentence = ""
                query = T5.predict(query + " " + sentence + " " + ' '.join(context))[0]

            if turn["automatic_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_df.loc[ms_marco_df['id'] ==
                                            str(turn["automatic_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_df.loc[car_df['id'] ==
                                       str(turn["automatic_canonical_result_id"][4:])]["paragraph"].to_string()

            context.append(str(query))
            if turn["number"] > 1:
                y.append({"turn": turn["number"],
                          "reformatted_query": query,
                          "automatic_reformatted_query": turn["automatic_rewritten_utterance"]})

    calculate_scores(y, paragraph_usage_option, True)
