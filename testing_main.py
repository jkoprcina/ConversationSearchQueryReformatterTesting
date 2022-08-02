from sentence_extractors import *
from pygaggle.pygaggle.rerank.base import Query, hits_to_texts


def main_loop(evaluation_data_json, BM25, MonoBERT, T5, sentence_extractor):
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

            hits = BM25.search(Query(query).text)
            texts = hits_to_texts(hits)

            reranked = MonoBERT.rerank(Query(query), texts)
            reranked.sort(key=lambda x:x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]
            paragraph = reranked_paragraphs[0][1]

            context.append(str(query))
            print(turn["number"])
            print("Automatic query: " + str(turn["automatic_rewritten_utterance"]))
            print("Our query: " + str(query))
