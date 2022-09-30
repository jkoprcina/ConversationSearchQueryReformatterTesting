from base_query_rewriters import *
from personal_query_rewriters import *


def query_rewriting(data_usage_type, conv, T5, summarizer, T5_config):
    if T5_config == "base":
        base_query_rewriting(data_usage_type, conv, T5, summarizer)
    elif T5_config == "personal":
        personal_query_rewriting(data_usage_type, conv, T5, summarizer)
    else:
        raise Exception("ERROR: T5 CONFIGURATION DOES NOT EXIST")


def base_query_rewriting(data_usage_type, conv, T5, summarizer):
    if data_usage_type == "previous_query":
        base_previous_query_rewriting(T5, conv)
    elif data_usage_type == "no_paragraph":
        base_no_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_similar_sentence":
        base_one_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_similar_sentences":
        base_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "one_full_paragraph":
        base_one_full_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_short_summarized_paragraph":
        base_one_short_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "one_mid_summarized_paragraph":
        base_one_mid_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_short_then_combined":
        base_all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_mid_then_combined":
        base_all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_short":
        base_all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_mid":
        base_all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_long":
        base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting":
        base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting":
        base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    else:
        raise Exception("ERROR: SENTENCE EXTRACTION OPTION SHOULDN'T EXIST")


def personal_query_rewriting(data_usage_type, conv, T5, summarizer):
    if data_usage_type == "previous_query":
        personal_previous_query_rewriting(T5, conv)
    elif data_usage_type == "no_paragraph":
        personal_no_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_similar_sentence":
        personal_one_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_similar_sentences":
        personal_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "one_full_paragraph":
        personal_one_full_paragraph_rewriting(T5, conv)

    elif data_usage_type == "one_short_summarized_paragraph":
        personal_one_short_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "one_mid_summarized_paragraph":
        personal_one_mid_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_short_then_combined":
        personal_all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_mid_then_combined":
        personal_all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_short":
        personal_all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_mid":
        personal_all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_long":
        personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting":
        personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting":
        personal_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    else:
        raise Exception("ERROR: SENTENCE EXTRACTION OPTION SHOULDN'T EXIST")
