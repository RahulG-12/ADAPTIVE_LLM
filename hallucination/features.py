import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)