from pygaggle.rerank.base import Query, hits_to_texts
import neptune.new as neptune
from score_calculators import *


def testing_retrieval_reranking(evaluation_data_json, BM25, MonoBERT):
    y = []
    for topic in evaluation_data_json:
        for turn in topic["turn"]:

            hits = BM25.search(Query(turn["manual_rewritten_utterance"]).text, k=1000)
            texts = hits_to_texts(hits)

            reranked = MonoBERT.rerank(Query(turn["manual_rewritten_utterance"]), texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]

            y.append({
                "turn": turn["number"],

                "returned_paragraph_id": [json.loads(x[1])["id"] for x in reranked_paragraphs],
                "manual_returned_paragraph_id": turn["manual_canonical_result_id"],
                "returned_paragraphs": reranked_paragraphs,
            })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["run_type"] = "testing_retrieval_reranking"
    calculate_and_log_ndcg3_score(y, run)
