from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def id_feature(id, sparse=False):
    id_encoder = OneHotEncoder(sparse=sparse)
    return id_encoder.fit_transform(id)


def num_feature(num):
    num_scaler = StandardScaler()
    return num_scaler.fit_transform(num)


def multi_value_feature(multi_value):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(multi_value).toarray()


def bool_feature(boo_value):
    return boo_value.astype(int).values
