import inspect
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# scikit-learn renamed OneHotEncoder's `sparse` to `sparse_output` in 1.2 and removed the old name
# in 1.4. Pick whichever this install understands rather than pinning the library to one side.
_ONEHOT_SPARSE_KW = (
    "sparse_output" if "sparse_output" in inspect.signature(OneHotEncoder).parameters else "sparse"
)

_sent_transformer = None
_sent_transformer_error = None
_sent_transformer_tried = False


def _load_sent_transformer():
    """
    Resolve the sentence transformer on first use and remember the outcome.

    Building it downloads `bert-base-chinese`, and this module is imported by user_feature and
    item_feature — so constructing it at import time charged every training run and every
    rank-engine startup for a model that only `vector_feature` needs.
    """
    global _sent_transformer, _sent_transformer_error, _sent_transformer_tried
    if _sent_transformer_tried:
        return _sent_transformer
    _sent_transformer_tried = True

    try:
        from sentence_transformers import SentenceTransformer

        _sent_transformer = SentenceTransformer('bert-base-chinese')
    except ImportError as ie:
        _sent_transformer_error = f"sentence-transformers is not installed ({ie})"
    except Exception as e:
        # usually no route to huggingface from the training box
        _sent_transformer_error = f"cannot load bert-base-chinese ({e})"
    if _sent_transformer_error:
        logging.warning("load sent transformer failed: %s", _sent_transformer_error)
    return _sent_transformer


def id_feature(id, sparse=False):
    id_encoder = OneHotEncoder(**{_ONEHOT_SPARSE_KW: sparse})
    return id_encoder.fit_transform(id)


def num_feature(num):
    num_scaler = StandardScaler()
    return num_scaler.fit_transform(num)


def vector_feature(text):
    # `raise sent_transformer` used to stand here: raising the model object instead of an exception
    # is a guaranteed TypeError, and it also made the encode call below unreachable.
    transformer = _load_sent_transformer()
    if transformer is None:
        raise RuntimeError(f"sentence transformer unavailable: {_sent_transformer_error}")
    return transformer.encode(text)


def text_feature(text):
    # imported here rather than at module scope: jieba is needed by this helper alone, while this
    # module is imported by every consumer of UserFeature/ItemFeature.
    import jieba

    vectorizer = TfidfVectorizer(tokenizer=lambda x: jieba.lcut(x))
    # `text` must be a 1-D iterable of strings (a Series) — iterating a DataFrame yields its column
    # names, which would vectorize the header instead of the rows. Dense like the other helpers,
    # because callers np.hstack the results together.
    return vectorizer.fit_transform(text).toarray()


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
