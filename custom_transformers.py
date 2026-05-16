import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MultiTagBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sep=",", min_freq=50):
        self.sep, self.min_freq = sep, min_freq

    def _split(self, series):
        return [[t.strip() for t in str(v).split(self.sep) if t.strip()]
                for v in series]

    def fit(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(X.flatten())
        c = {}
        for tags in self._split(col):
            for t in tags:
                c[t] = c.get(t, 0) + 1
        self.classes_   = sorted(t for t, f in c.items() if f >= self.min_freq)
        self.tag_index_ = {t: i for i, t in enumerate(self.classes_)}
        return self

    def transform(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(X.flatten())
        mat = np.zeros((len(list(self._split(col))), len(self.classes_)),
                       dtype=np.float32)
        for i, tags in enumerate(self._split(col)):
            for t in tags:
                if t in self.tag_index_:
                    mat[i, self.tag_index_[t]] = 1.0
        return mat

    def get_feature_names_out(self, input_features=None):
        return np.array([f"tag_{t.replace(' ', '_')}" for t in self.classes_])