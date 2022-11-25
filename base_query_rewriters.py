def base_all_raw_queries_rewriting(T5, conv):
    data = "|||".join(conv.queries)
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_rewritten_queries_rewriting(T5, conv):
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_raw_queries_all_paragraphs_rewriting(T5, conv):
    data = "|||".join(conv.queries[:-1]) + "|||" + "|||".join(conv.paragraphs) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_raw_queries_all_paragraphs_rewriting(T5, conv):
    data = "|||".join(conv.queries[:-1]) + "|||" + "|||".join(conv.paragraphs) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_raw_queries_all_similar_sentences_rewriting(T5, conv):
    conv.add_similar_sentence()
    data = "|||".join(conv.queries[:-1]) + "|||" + "|||".join(conv.sentences) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_raw_queries_one_similar_sentences_rewriting(T5, conv):
    conv.add_similar_sentence()
    data = "|||".join(conv.queries[:-1]) + "|||" + conv.sentences[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_rewritten_queries_one_similar_sentence_rewriting(T5, conv):
    conv.add_similar_sentence()
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.sentences[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_raw_queries_no_paragraph_rewriting(T5, conv):
    data = "|||".join(conv.queries) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_rewritten_queries_no_paragraph_rewriting(T5, conv):
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_one_cosine_similar_sentence_rewriting(T5, conv):
    conv.add_cosine_similar_sentence()
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.sentences[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_similar_sentences_rewriting(T5, conv):
    conv.add_similar_sentence()
    data = "|||".join(conv.rewritten_queries) + "|||" + "|||".join(conv.sentences) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_cosine_similar_sentence_rewriting(T5, conv):
    conv.add_cosine_similar_sentence()
    data = "|||".join(conv.rewritten_queries) + "|||" + "|||".join(conv.sentences) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_one_full_paragraph_rewriting(T5, conv):
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.paragraphs[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_one_short_summarized_paragraph_rewriting(T5, summarizer, conv):
    conv.add_summarization(summarizer, 10, 2)
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.summarizations[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_one_mid_summarized_paragraph_rewriting(T5, summarizer, conv):
    conv.add_summarization(summarizer, 30, 5)
    data = "|||".join(conv.rewritten_queries) + "|||" + conv.summarizations[-1] + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_summarized_short_then_combined_rewriting(T5, summarizer, conv):
    conv.add_summarization(summarizer, 10, 2)
    data = "|||".join(conv.rewritten_queries) + "|||" + " ".join(conv.summarizations) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_summarized_mid_then_combined_rewriting(T5, summarizer, conv):
    conv.add_summarization(summarizer, 30, 5)
    data = "|||".join(conv.rewritten_queries) + "|||" + " ".join(conv.summarizations) + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_combined_then_summarized_short_rewriting(T5, summarizer, conv):
    summarization = summarizer(
        " ".join(conv.paragraphs), max_length=12, min_length=5, do_sample=False)[0].get("summary_text")
    data = "|||".join(conv.rewritten_queries) + "|||" + summarization + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_combined_then_summarized_mid_rewriting(T5, summarizer, conv):
    summarization = summarizer(
        " ".join(conv.paragraphs), max_length=20, min_length=5, do_sample=False)[0].get("summary_text")
    data = "|||".join(conv.rewritten_queries) + "|||" + summarization + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_combined_then_summarized_long_rewriting(T5, summarizer, conv):
    summarization = summarizer(
        " ".join(conv.paragraphs), max_length=50, min_length=10, do_sample=False)[0].get("summary_text")
    data = "|||".join(conv.rewritten_queries) + "|||" + summarization + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_combined_then_summarized_scaling_by_3_to_10_rewriting(T5, summarizer, conv):
    max_length = len(conv.queries) * 10
    min_length = len(conv.queries) * 3
    summarization = summarizer(
        " ".join(conv.paragraphs), max_length=max_length, min_length=min_length, do_sample=False)[0].get("summary_text")
    data = "|||".join(conv.rewritten_queries) + "|||" + summarization + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data


def base_all_paragraphs_combined_then_summarized_scaling_by_2_to_5_rewriting(T5, summarizer, conv):
    max_length = len(conv.queries) * 5
    min_length = len(conv.queries) * 2
    summarization = summarizer(
        " ".join(conv.paragraphs), max_length=max_length, min_length=min_length, do_sample=False)[0].get("summary_text")
    data = "|||".join(conv.rewritten_queries) + "|||" + summarization + "|||" + conv.queries[-1]
    rewritten_query = T5.predict(data)[0]
    conv.rewritten_queries.append(rewritten_query)
    return data
