from base_query_rewriters import *
from personal_query_rewriters import *


def query_rewriting(data_usage_type, conv, T5, summarizer, T5_config):
    if T5_config == "base":
        return base_query_rewriting(data_usage_type, conv, T5, summarizer)
    elif T5_config == "personal":
        return personal_query_rewriting(data_usage_type, conv, T5, summarizer)
    else:
        raise Exception("ERROR: T5 CONFIGURATION DOES NOT EXIST")


def base_query_rewriting(data_usage_type, conv, T5, summarizer):
    if data_usage_type == "all_raw_queries":
        return base_all_raw_queries_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries":
        return base_all_rewritten_queries_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_all_paragraphs":
        return base_all_raw_queries_all_paragraphs_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_all_similar_sentences":
        return base_all_raw_queries_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_one_similar_sentences":
        return base_all_raw_queries_one_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_one_similar_sentence":
        return base_all_rewritten_queries_one_similar_sentence_rewriting(T5, conv)

    elif data_usage_type == "all_raw_queries_no_paragraph":
        return base_all_raw_queries_no_paragraph_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_no_paragraph":
        return base_all_rewritten_queries_no_paragraph_rewriting(T5, conv)
    elif data_usage_type == "one_cosine_similar_sentence":
        return base_one_cosine_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_similar_sentences":
        return base_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_cosine_similar_sentences":
        return base_all_cosine_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "one_full_paragraph":
        return base_one_full_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_short_summarized_paragraph":
        return base_one_short_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "one_mid_summarized_paragraph":
        return base_one_mid_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_short_then_combined":
        return base_all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_mid_then_combined":
        return base_all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_short":
        return base_all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_mid":
        return base_all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_long":
        return base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting":
        return base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting":
        return base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    else:
        raise Exception("ERROR: SENTENCE EXTRACTION OPTION SHOULDN'T EXIST")


def personal_query_rewriting(data_usage_type, conv, T5, summarizer):
    if data_usage_type == "all_raw_queries":
        return personal_all_raw_queries_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries":
        return personal_all_rewritten_queries_rewriting(T5, conv)
    if data_usage_type == "all_raw_queries_all_paragraphs":
        return personal_all_raw_queries_all_paragraphs_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_all_similar_sentences":
        return personal_all_raw_queries_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_one_similar_sentences":
        return personal_all_raw_queries_one_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_one_similar_sentence":
        return personal_all_rewritten_queries_one_similar_sentence_rewriting(T5, conv)

    elif data_usage_type == "all_raw_queries_no_paragraph":
        return personal_all_raw_queries_no_paragraph_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_no_paragraph":
        return personal_all_rewritten_queries_no_paragraph_rewriting(T5, conv)
    elif data_usage_type == "one_cosine_similar_sentence":
        return personal_one_cosine_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_similar_sentences":
        return personal_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_cosine_similar_sentences":
        return personal_all_cosine_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "one_full_paragraph":
        return personal_one_full_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_short_summarized_paragraph":
        return personal_one_short_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "one_mid_summarized_paragraph":
        return personal_one_mid_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_short_then_combined":
        return personal_all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_mid_then_combined":
        return personal_all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_short":
        return personal_all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_mid":
        return personal_all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_long":
        return personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting":
        return personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting":
        return personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    else:
        raise Exception("ERROR: SENTENCE EXTRACTION OPTION SHOULDN'T EXIST")
