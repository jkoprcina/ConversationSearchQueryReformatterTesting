from pygaggle.pygaggle.rerank.transformer import MonoBERT
from rank_bm25 import BM25Okapi
from simplet5 import SimpleT5

from get_data import *
from testing_main import *
from canonical_testing_main import *


if __name__ == '__main__':
    EVALUATION_DATA_FILE_NAME = "data/2020_automatic_evaluation_topics_v1.json"
    MS_MARCO_DATA_FILE_NAME = "data/ms_marco_collection.tsv"
    CAR_DATA_FILE_NAME = "data/dedup.articles-paragraphs.cbor"

    evaluation_data_json = get_evaluation_data(EVALUATION_DATA_FILE_NAME)
    ms_marco_collection_df = get_ms_marco_data(MS_MARCO_DATA_FILE_NAME)
    with open(CAR_DATA_FILE_NAME, 'rb') as f:
        car_collection_df = get_car_collection(f)

    corpus_df = pd.concat([ms_marco_collection_df, car_collection_df])

    corpus = corpus_df["paragraph"]
    tokenized_corpus = [x.split(" ") for x in corpus]
    BM25 = BM25Okapi(tokenized_corpus, k1=4.46, b=0.82)

    reranker = MonoBERT()
    T5 = SimpleT5()
    T5.load_model("t5", "data/simplet5", use_gpu=True)

    test = "How much does it cost for someone to fix it? Why did garage door opener stop working? How do you know " \
           "when your garage door opener is going bad? "
    T5.predict(test)

    main_loop(corpus, evaluation_data_json, BM25, MonoBERT, T5, "h2oloo")

    canonical_main_loop(ms_marco_collection_df, car_collection_df, evaluation_data_json, T5, "h2oloo")


