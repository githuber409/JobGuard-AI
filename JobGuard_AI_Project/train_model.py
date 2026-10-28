"""Optional local training script for the EMSCAD / Kaggle fake-job-posting CSV.

Usage:
    python train_model.py path/to/fake_job_postings.csv

This script trains and evaluates a text classifier. It prints metrics but does not
save personal information or upload the dataset anywhere.
"""
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main(path):
    df = pd.read_csv(path)
    if "fraudulent" not in df.columns:
        raise SystemExit("CSV must contain a 'fraudulent' label column (0=real, 1=fake).")
    fields = [c for c in ["title","company_profile","description","requirements","benefits",
                          "employment_type","required_experience","required_education","industry",
                          "function","location"] if c in df.columns]
    if not fields:
        raise SystemExit("No recognized job text columns were found.")
    X = df[fields].fillna("").astype(str).agg(" ".join, axis=1)
    X = X.map(lambda s: re.sub(r"\s+", " ", s).strip())
    y = pd.to_numeric(df["fraudulent"], errors="coerce")
    mask = X.str.len().gt(0) & y.isin([0,1])
    X, y = X[mask], y[mask].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise SystemExit("Need examples from both classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Rows used: {len(X)}")
    print("\nClassification report:\n")
    print(classification_report(y_test, pred, target_names=["Legitimate (0)", "Fake (1)"], zero_division=0))
    print("Confusion matrix (actual rows; predicted columns; order 0,1):")
    print(confusion_matrix(y_test, pred, labels=[0,1]))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_model.py path/to/fake_job_postings.csv")
    main(sys.argv[1])
