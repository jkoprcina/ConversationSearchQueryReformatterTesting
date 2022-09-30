from sentence_extractors import h2oloo_find_sentence


class Conversation:
    def __init__(self):
        self.queries = []
        self.sentences = []
        self.paragraphs = []
        self.summarizations = []
        self.rewritten_queries = []

    def add_query(self, query):
        self.queries.append(query)

    def add_paragraph(self, paragraph):
        self.paragraphs.append(paragraph)

    def add_similar_sentence(self):
        sentence = h2oloo_find_sentence(self.paragraphs[-1], self.queries[-1])
        self.sentences.append(sentence)

    def add_summarization(self, summarizer, max_length, min_length):
        self.summarizations.append(summarizer(
            self.paragraphs[-1], max_length=max_length, min_length=min_length, do_sample=False)[0].get("summary_text"))

    def add_to_rewritten_queries(self, query):
        self.rewritten_queries.append(query)
