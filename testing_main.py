from rouge_score import *
from sentence_extractors import *
from pygaggle.rerank.base import Text


def main_loop(corpus, evaluation_data_json, BM25, MonoBERT, T5, sentence_extractor):
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

            tokenized_query = query.split(" ")
            top_10_returned_paragraphs = BM25.get_top_n(tokenized_query, corpus, n=10)

            top_10_returned_paragraph_text = [Text(p) for p in top_10_returned_paragraphs]
            top_10_reranked_paragraphs = MonoBERT.rerank(Text(query), top_10_returned_paragraph_text)
            reranked_paragraphs = [[p.score, p.text] for p in top_10_reranked_paragraphs]
            paragraph = reranked_paragraphs[0][1]

            context.append(str(query))
            y.append({"turn": turn["number"],
                      "reformatted_query": query,
                      "automatically_reformatted_query": turn["automatic_rewritten_utterance"]})

    precisions, recalls, fmeasures = calculate_rouge_score(y)
    print_rouge_score(precisions, recalls, fmeasures)
