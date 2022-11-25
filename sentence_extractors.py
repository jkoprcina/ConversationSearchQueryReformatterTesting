import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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


def cosine_find_sentence(paragraph, query):
    paragraph = nlp(paragraph)
    sw = stopwords.words('english')
    query = word_tokenize(query.lower())
    query_set = {w for w in query if not w in sw}
    remember_results = []

    for sentence in paragraph.sents:
        l1, l2 = [], []
        sentence_list = word_tokenize(sentence.text.lower())
        sentence_set = {w for w in sentence_list if not w in sw}

        rvector = query_set.union(sentence_set)
        [l1.append(1) if w in query_set else l1.append(0) for w in rvector]
        [l2.append(1) if w in sentence_set else l2.append(0) for w in rvector]
        c = 0

        for i in range(len(rvector)):
            c += l1[i] * l2[i]
        cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
        remember_results.append({"result": cosine, "sentence": sentence})

    print(remember_results)
    best_result = {"result": 0, "sentence": ""}
    for result in remember_results:
        if result["result"] > best_result["result"]:
            best_result = result
    return best_result["sentence"]


def text_useful_word_count(text):
    text = nlp(text)
    cleaned_text = []
    for token in text:
        if token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "ADJ":
            cleaned_text.append(str(token))
    return len(cleaned_text)
