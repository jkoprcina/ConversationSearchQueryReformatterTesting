from sentence_extractors import h2oloo_find_sentence
from pygaggle.pygaggle.rerank.base import Query, hits_to_texts
from score import calculate_scores


def main_loop(evaluation_data_json, BM25, MonoBERT, T5, paragraph_usage_option):
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

            hits = BM25.search(Query(query).text)
            texts = hits_to_texts(hits)

            reranked = MonoBERT.rerank(Query(query), texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]
            paragraph = reranked_paragraphs[0][1]
            context.append(str(query))
            if turn["number"] > 1:
                y.append({"turn": turn["number"],
                          "reformatted_query": query,
                          "automatic_reformatted_query": turn["automatic_rewritten_utterance"]})

    calculate_scores(y, paragraph_usage_option, False)
