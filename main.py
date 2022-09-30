from pygaggle.rerank.transformer import MonoBERT
from pyserini.search.lucene import LuceneSearcher
from simplet5 import SimpleT5
from transformers import pipeline

from get_data import get_evaluation_data, get_canonical_df
from runs.manual_run import manual_run
from runs.testing_everything import testing_everything
from runs.testing_query_rewriting import testing_query_rewriting
from runs.testing_retrieval_reranking import testing_retrieval_reranking
from constants import *

if __name__ == '__main__':
    evaluation_data_json = get_evaluation_data(EVALUATION_DATA_FILE_LOCATION)
    canonical_ms_marco_df = get_canonical_df(CANONICAL_MS_MARCO_DATA_FILE_LOCATION)
    canonical_car_df = get_canonical_df(CANONICAL_CAR_DATA_FILE_LOCATION)

    searcher = LuceneSearcher(SEARCHER_DATA_FILE_LOCATION)
    reranker = MonoBERT()
    our_T5 = SimpleT5()
    our_T5.load_model("t5", PERSONAL_T5_REWRITER_CONFIG_LOCATION, use_gpu=True)
    base_T5 = SimpleT5()
    base_T5.load_model("t5", BASE_T5_REWRITER_CONFIG_LOCATION, use_gpu=True)
    summarizeT5 = pipeline("summarization", model=T5_SUMMARIZER_CONFIG_LOCATION)

    manual_run(canonical_ms_marco_df, canonical_car_df, evaluation_data_json)
    testing_retrieval_reranking(evaluation_data_json, searcher, reranker)

    for data_usage_type in DATA_USAGE_TYPES:
        testing_everything(evaluation_data_json, searcher, reranker, our_T5, data_usage_type, summarizeT5, "personal")
        testing_query_rewriting(canonical_ms_marco_df, canonical_car_df, evaluation_data_json, our_T5,
                                data_usage_type, summarizeT5, "personal")

        testing_everything(evaluation_data_json, searcher, reranker, base_T5, data_usage_type, summarizeT5, "base")
        testing_query_rewriting(canonical_ms_marco_df, canonical_car_df, evaluation_data_json, base_T5,
                                data_usage_type, summarizeT5, "base")
