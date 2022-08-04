import nltk
import spacy

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
stemmer = nltk.stem.porter.PorterStemmer()


def h2oloo_find_sentence(paragraph, query):
    query = nlp(query)
    paragraph = nlp(paragraph)

    cleaned_query = []
    cleaned_paragraph = []

    for token in query:
        if token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "ADJ":
            cleaned_query.append(stemmer.stem(str(token)))

    for sentence in paragraph.sents:
        sentence_list = []
        for token in sentence:
            if token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "ADJ":
                sentence_list.append(stemmer.stem(str(token)))
        cleaned_paragraph.append({"tokenized_sentence": sentence_list, "sentence": str(sentence), "value": 0})

    for sentence in cleaned_paragraph:
        for sentence_word in sentence["tokenized_sentence"]:
            for query_word in cleaned_query:
                if query_word == sentence_word:
                    sentence["value"] += 1

    sentence_to_return = ""
    value = 0
    for sentence in cleaned_paragraph:
        if sentence["value"] > value:
            sentence_to_return = sentence["sentence"]

    return sentence_to_return
