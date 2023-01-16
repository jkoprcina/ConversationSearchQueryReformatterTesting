from query_rewriters import *


def query_rewriting(data_usage_type, conv, T5, summarizer):
    if data_usage_type == "all_raw_queries":
        return all_raw_queries_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_one_similar_sentence":
        return all_raw_queries_one_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_all_similar_sentences":
        return all_raw_queries_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_one_cosine_sentence":
        return all_raw_queries_one_cosine_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_raw_queries_all_cosine_similar_sentences":
        return all_raw_queries_all_cosine_similar_sentences_rewriting(T5, conv)

    elif data_usage_type == "all_rewritten_queries":
        return all_rewritten_queries_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_one_similar_sentence":
        return all_rewritten_queries_one_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_all_similar_sentences":
        return all_rewritten_queries_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_one_cosine_sentence":
        return all_rewritten_queries_one_cosine_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_rewritten_queries_all_cosine_similar_sentences":
        return all_rewritten_queries_all_cosine_similar_sentences_rewriting(T5, conv)

    elif data_usage_type == "all_gold_standard_queries":
        return all_gold_standard_queries_rewriting(T5, conv)
    elif data_usage_type == "all_gold_standard_queries_one_similar_sentence":
        return all_gold_standard_queries_one_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_gold_standard_queries_all_similar_sentences":
        return all_gold_standard_queries_all_similar_sentences_rewriting(T5, conv)
    elif data_usage_type == "all_gold_standard_queries_one_cosine_sentence":
        return all_gold_standard_queries_one_cosine_similar_sentence_rewriting(T5, conv)
    elif data_usage_type == "all_gold_standard_queries_all_cosine_similar_sentences":
        return all_gold_standard_queries_all_cosine_similar_sentences_rewriting(T5, conv)

    elif data_usage_type == "one_short_summarized_paragraph":
        return one_short_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "one_mid_summarized_paragraph":
        return one_mid_summarized_paragraph_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_short_then_combined":
        return all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_summarized_mid_then_combined":
        return all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_short":
        return all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_mid":
        return all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_long":
        return all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)

    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting":
        return all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    elif data_usage_type == "all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting":
        return all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv)
    else:
        raise Exception("ERROR: SENTENCE EXTRACTION OPTION SHOULDN'T EXIST")
