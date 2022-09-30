from pygaggle.rerank.base import Query, hits_to_texts
import json
import neptune.new as neptune

from constants import API_TOKEN, PROJECT
from conversation import Conversation
from choosing_query_rewriting import query_rewriting
from score_calculators import calculate_rouge_score, calculate_bleu_score, calculate_ndcg3_score


def testing_everything(evaluation_data_json, BM25, MonoBERT, T5, data_usage_type, summarizer, T5_config):
    y = []
    for topic in evaluation_data_json:
        conv = Conversation()
        for turn in topic["turn"]:
            conv.add_query(turn["raw_utterance"])
            query_to_use = turn["raw_utterance"]

            if turn["number"] > 1:
                query_rewriting(data_usage_type, conv, T5, summarizer, T5_config)
                query_to_use = conv.rewritten_queries[-1]
            else:
                conv.add_to_rewritten_queries(query_to_use)

            hits = BM25.search(Query(query_to_use).text, k=1000)
            texts = hits_to_texts(hits)

            reranked = MonoBERT.rerank(Query(query_to_use), texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]

            reranked_paragraph = json.loads(reranked_paragraphs[0][1])
            conv.add_paragraph(reranked_paragraph["contents"])

            y.append({
                "turn": turn["number"],

                "returned_paragraph_id": [json.loads(x[1])["id"] for x in reranked_paragraphs],
                "automatically_returned_paragraph_id": turn["automatic_canonical_result_id"],
                "returned_paragraphs": reranked_paragraphs,

                "reformatted_query": conv.rewritten_queries[-1],
                "automatic_reformatted_query": turn["automatic_rewritten_utterance"]
            })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["data_usage_type"] = data_usage_type
    run["run_type"] = "testing_everything"
    run["T5_query_rewriter"] = T5_config
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
    calculate_ndcg3_score(y, run)
