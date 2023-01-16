from pygaggle.rerank.base import Query, hits_to_texts
import neptune.new as neptune

from conversation import Conversation
from choosing_query_rewriting import query_rewriting
from score_calculators import *
from sentence_extractors import text_useful_word_count


def automatic(evaluation_data_json, BM25, MonoBERT, T5, data_usage_type, summarizer, T5_config):
    y = []
    for topic in evaluation_data_json:
        conv = Conversation()
        for turn in topic["turn"]:
            conv.add_query(turn["raw_utterance"])
            query_to_use = turn["raw_utterance"]

            if turn["number"] > 1:
                data = query_rewriting(data_usage_type, conv, T5, summarizer)
                query_to_use = conv.rewritten_queries[-1]
            else:
                conv.add_rewritten_query(query_to_use)
                data = query_to_use

            hits = BM25.search(Query(query_to_use).text, k=1000)
            texts = hits_to_texts(hits)
            reranked = MonoBERT.rerank(Query(query_to_use), texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked_paragraphs = [[p.score, p.text] for p in reranked]

            reranked_paragraph = json.loads(reranked_paragraphs[0][1])
            conv.add_paragraph(reranked_paragraph["contents"])
            conv.add_golden_query(turn["manual_rewritten_utterance"])

            y.append({
                "turn": turn["number"],

                "manual_returned_paragraph_id": turn["manual_canonical_result_id"],
                "returned_paragraphs": reranked_paragraphs,

                "rewritten_query": conv.rewritten_queries[-1],
                "manual_rewritten_query": turn["manual_rewritten_utterance"],

                "token_count": len(data),
                "word_count": text_useful_word_count(data),
            })

    run = neptune.init(project=PROJECT, api_token=API_TOKEN)
    run["data_usage_type"] = data_usage_type
    run["run_type"] = "automatic"
    run["T5_query_rewriter"] = T5_config
    calculate_rouge_score(y, run)
    calculate_bleu_score(y, run)
    calculate_and_log_ndcg3_score(y, run)
    calculate_counts(y, run)
