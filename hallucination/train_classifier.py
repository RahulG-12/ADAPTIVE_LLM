import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hallucination.features import FeatureExtractor
import joblib

def main():
    df = pd.read_csv("data/hallucination_data.csv")

    texts = df["response"]
    labels = df["label"]

    extractor = FeatureExtractor()
    X = extractor.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    joblib.dump(model, "hallucination/model.pkl")
    joblib.dump(extractor, "hallucination/vectorizer.pkl")

if __name__ == "__main__":
    main()