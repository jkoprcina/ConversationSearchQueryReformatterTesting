from pygaggle.rerank.base import Query, hits_to_texts
import json
import neptune.new as neptune

from constants import API_TOKEN, PROJECT
from conversation import Conversation
from choosing_query_rewriting import query_rewriting
from score_calculators import *
from sentence_extractors import text_useful_word_count


def canonical(ms_marco_df, car_df, evaluation_data_json, BM25, MonoBERT, T5, data_usage_type, summarizer, T5_config):
    y = []
    for topic in evaluation_data_json:
        conv = Conversation()
        for turn in topic["turn"]:
            conv.add_query(turn["raw_utterance"])

            if turn["number"] > 1:
                data = query_rewriting(data_usage_type, conv, T5, summarizer, T5_config)
            else:
                data = turn["raw_utterance"]
                conv.add_to_rewritten_queries(turn["raw_utterance"])

            if turn["manual_canonical_result_id"][0:5] == "MARCO":
                paragraph = ms_marco_df.loc[ms_marco_df['id'] == str(turn["manual_canonical_result_id"][6:])]["paragraph"].to_string()
            else:
                paragraph = car_df.loc[car_df['id'] == str(turn["manual_canonical_result_id"][4:])]["paragraph"].to_string()
            conv.add_paragraph(paragraph)

            hits = BM25.search(Query(query_to_use).text, k=1000)
            texts = hits_to_texts(hits)

            reranked = MonoBERT.rerank(Query(query_to_use), texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]

            y.append({
                "turn": turn["number"],

                "returned_paragraph_id": [json.loads(x[1])["id"] for x in reranked_paragraphs],
                "manual_returned_paragraph_id": turn["manual_canonical_result_id"],
                "returned_paragraphs": reranked_paragraphs,

                "rewritten_query": conv.rewritten_queries[-1],
                "manual_rewritten_query": turn["manual_rewritten_utterance"],

                "token_count": len(data),
                "word_count": text_useful_word_count(data),
            })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["data_usage_type"] = data_usage_type
    run["run_type"] = "testing_query_rewriting"
    run["T5_query_rewriter"] = T5_config
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
    calculate_ndcg3_score(y, run)
    calculate_counts(y, run)
