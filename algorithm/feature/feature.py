import logging

import jieba
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sent_transformer = None
try:
    from sentence_transformers import SentenceTransformer

    sent_transformer = SentenceTransformer('bert-base-chinese')
except ImportError as ie:
    logging.warning("import sentence transformer failed")
except (requests.exceptions.ProxyError, EnvironmentError) as pe:
    logging.warning("cannot connect to huggingface")
except Exception as e:
    logging.error(f"unknown error, load sent transformer failed: {e}")


def id_feature(id, sparse=False):
    id_encoder = OneHotEncoder(sparse=sparse)
    return id_encoder.fit_transform(id)


def num_feature(num):
    num_scaler = StandardScaler()
    return num_scaler.fit_transform(num)


def vector_feature(text):
    # `raise sent_transformer` used to stand here: raising the model object instead of an exception
    # is a guaranteed TypeError, and it also made the encode call below unreachable.
    if sent_transformer is None:
        raise RuntimeError(
            "sentence transformer unavailable: install sentence-transformers and make sure "
            "bert-base-chinese can be downloaded")
    return sent_transformer.encode(text)


def text_feature(text):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: jieba.lcut(x))
    return vectorizer.fit_transform(text)


def multi_value_feature(multi_value, tokenizer=None):
    if tokenizer:
        vectorizer = CountVectorizer(tokenizer=tokenizer)
    else:
        vectorizer = CountVectorizer()

    if len(multi_value.values) and type(multi_value.values[0]) is list:
        multi_value = multi_value.apply(lambda x: " ".join(x))
    return vectorizer.fit_transform(multi_value).toarray()


def bool_feature(boo_value):
    return boo_value.astype(int).values
