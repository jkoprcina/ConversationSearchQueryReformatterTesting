EVALUATION_DATA_FILE_LOCATION = 'data/2020_manual_evaluation_topics_v1_with_paragraphs.json'
CANONICAL_MS_MARCO_DATA_FILE_LOCATION = 'data/canonical_ms_marco'
CANONICAL_CAR_DATA_FILE_LOCATION = 'data/canonical_car'
SEARCHER_DATA_FILE_LOCATION = 'data/indexes/lucene-index-msmarco-passage'
PERSONAL_T5_REWRITER_CONFIG_LOCATION = 'data/our_simplet5_qr'
BASE_T5_REWRITER_CONFIG_LOCATION = 'data/base_simplet5_qr'
T5_SUMMARIZER_CONFIG_LOCATION = 'data/t5summarize'
DATA_USAGE_TYPES = [
    'all_raw_queries',
    'all_raw_queries_one_similar_sentence',
    'all_raw_queries_all_similar_sentences',
    'all_raw_queries_one_cosine_sentence',
    'all_raw_queries_all_cosine_similar_sentences',

    'all_rewritten_queries',
    'all_rewritten_queries_one_similar_sentence',
    'all_rewritten_queries_all_similar_sentences',
    'all_rewritten_queries_one_cosine_sentence',
    'all_rewritten_queries_all_cosine_similar_sentences',

    'all_gold_standard_queries',
    'all_gold_standard_queries_one_similar_sentence',
    'all_gold_standard_queries_all_similar_sentences',
    'all_gold_standard_queries_one_cosine_sentence',
    'all_gold_standard_queries_all_cosine_similar_sentences',

    'one_short_summarized_paragraph',
    'one_mid_summarized_paragraph',
    'all_paragraphs_summarized_short_then_combined',
    'all_paragraphs_summarized_mid_then_combined',

    'all_paragraphs_combined_then_summarized_short',
    'all_paragraphs_combined_then_summarized_mid',
    'all_paragraphs_combined_then_summarized_long',

    'all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting',
    'all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting',
]
PROJECT = "red-lion/query-rewriting-clean"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2MxYjc0MC1mODhiLTQ2ZDctOGVmNC1lOGE1ZTkyM2YzMDEifQ=="
