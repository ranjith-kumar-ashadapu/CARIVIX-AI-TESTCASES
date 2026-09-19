"""
Train a simple intent classifier for CARIVIX NLP routing.
Creates a small synthetic dataset, trains a TF-IDF + LogisticRegression
pipeline, evaluates it, and saves the model to models/intent_classifier.joblib.

This is a bootstrap classifier — replace with real labeled data for production.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import random

MODEL_PATH = Path("models/intent_classifier.joblib")

# Synthetic training phrases by intent
RAG_EXAMPLES = [
    "What is CARIVIX?",
    "Explain the findings in the report",
    "Who authored the sample_report.docx?",
    "Where can I find the economic indicators?",
    "Describe the methodology used in the document",
    "How does the model handle missing values?",
    "Summarize the sample_about_carivix.txt",
    "Provide a concise description of the dataset",
    "Show me the contents of the sample report",
    "What does the Traffic Analysis report say about congestion?",
]

ML_EXAMPLES = [
    "Predict next quarter GDP",
    "Estimate the probability of default",
    "Forecast sales for the next month",
    "Classify this customer as churn or not churn",
    "What is the predicted score for this applicant",
    "Run a model to predict housing prices",
    "Give me the probability the loan will default",
    "Compute the predicted demand for product X",
    "Estimate future unemployment rate",
    "Provide prediction for target variable using dataset Y",
]

COMBINED_EXAMPLES = [
    "Predict next quarter GDP and explain the supporting evidence in the documents",
    "Estimate probability and cite the report sections that support the forecast",
    "Forecast sales and provide the relevant analysis from the documents",
    "Classify and give supporting context from the report",
    "Predict and summarize related documents",
]

# Augment examples slightly
def augment_examples(examples, n_aug=20):
    out = []
    for ex in examples:
        out.append(ex)
        for i in range(n_aug):
            # small perturbations
            if random.random() < 0.5:
                out.append(ex + " please")
            else:
                out.append("Please " + ex.lower())
    return out

rag = augment_examples(RAG_EXAMPLES, n_aug=10)
ml = augment_examples(ML_EXAMPLES, n_aug=10)
combined = augment_examples(COMBINED_EXAMPLES, n_aug=6)

X = rag + ml + combined
y = ["rag" for _ in rag] + ["ml" for _ in ml] + ["combined" for _ in combined]

# Shuffle
combined_pairs = list(zip(X, y))
random.shuffle(combined_pairs)
X, y = zip(*combined_pairs)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(list(X), list(y), test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("lr", LogisticRegression(max_iter=1000, solver="lbfgs")),
])

print("Training intent classifier on synthetic data... this may take a few seconds")
pipeline.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Ensure models dir exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved intent classifier to: {MODEL_PATH}")
