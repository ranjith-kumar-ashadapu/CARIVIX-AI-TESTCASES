"""
Train an enterprise intent classifier for CARIVIX NLP routing and workflow validation.
Supports core intents:
  - GIS_VIEW: Map, boundary, district, spatial intelligence, telemetry queries
  - PREDICTION: Default risk, loan scoring, GDP/sales forecast, ML inference queries
  - DATA_METRIC: Revenue, count, statistics, economic indicator queries
  - FAQ: General knowledge, platform architecture, methodology documentation
  - UNKNOWN_INTENT: Out-of-scope queries, gibberish, noise, security fuzzing

Saves trained model to models/intent_classifier.joblib.
"""
from pathlib import Path
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "intent_classifier.joblib"

GIS_EXAMPLES = [
    "Show boundary map for Adilabad district",
    "Display rainfall map of Jagtial and Peddapalli",
    "View Telangana district boundaries",
    "Show GIS spatial map for Hyderabad",
    "Map coordinates for Karimnagar",
    "Display administrative boundary for tier 2",
    "Show risk zone polygon in Warangal",
    "View spatial analytics and density distribution",
    "Load district boundaries for Telangana state",
    "Where is the boundary for Nizamabad district?",
    "Show spatial telemetry and point clusters",
    "Render GeoJSON boundary tier 1",
    "Zoom to district coordinates on spatial map",
    "Show me the map of India national borders",
    "Filter districts by state Telangana",
]

PREDICTION_EXAMPLES = [
    "Predict next quarter GDP",
    "Estimate the probability of default",
    "Forecast sales for the next month",
    "Classify this customer as churn or not churn",
    "What is the predicted score for this applicant",
    "Run a model to predict housing prices",
    "Give me the probability the loan will default",
    "Compute the predicted demand for product X",
    "Estimate future unemployment rate",
    "Predict default probability for a 35-year-old borrower with income 50000",
    "Forecast credit risk for applicant with credit score 720",
    "Run predictive model on economic dataset",
    "Predict default risk for loan amount 25000",
    "Evaluate applicant loan default likelihood",
    "Run ML inference for credit default classification",
]

DATA_METRIC_EXAMPLES = [
    "Show rainfall metrics in Jagtial for 2025",
    "What is the total revenue for CARIVIX Tech Global",
    "Get average economic growth rate across sectors",
    "Retrieve inflation rate for 2026",
    "What is the average density index in the state",
    "Show financial report metrics for company Acme",
    "Count total registered data sources in catalog",
    "What was the annual revenue for Alpha Analytics Corp",
    "Retrieve unemployment statistics by quarter",
    "Show quarterly economic KPIs for 2024 to 2026",
    "Calculate average loan amount across applicants",
    "Fetch metric trends for regional employment",
]

FAQ_EXAMPLES = [
    "What is CARIVIX AI?",
    "Explain the methodology used in the document",
    "Who authored the project report?",
    "Where can I find the system architecture guide?",
    "Describe the data processing steps and ETL pipeline",
    "How does the platform handle missing values?",
    "Summarize the sample about CARIVIX platform",
    "What algorithms are supported by the ML module?",
    "Explain how the RAG document retrieval works",
    "What is the tech stack for CARIVIX AI?",
    "Describe the role of the Python backend service",
]

UNKNOWN_EXAMPLES = [
    "asdfghjkl qwerty zxcvbnm 12345",
    "SELECT * FROM users WHERE 1=1; DROP TABLE users;",
    "<script>alert('xss payload')</script>",
    "gibberish foo bar baz blip blop",
    "how do I bake a chocolate cake at home?",
    "tell me a funny bedtime story about dragons",
    "what is the capital of Mars?",
    "random noise words with no meaning whatsoever",
    "who won the cricket world cup in 1983?",
    "can you play music for me right now",
]

def augment_examples(examples, n_aug=12):
    out = []
    for ex in examples:
        out.append(ex)
        for _ in range(n_aug):
            prefix = random.choice(["Please ", "Can you ", "Kindly ", "I want to ", ""])
            suffix = random.choice([" please", " right now", " for me", " today", ""])
            out.append(f"{prefix}{ex.lower()}{suffix}".strip())
    return out

def train_and_save():
    gis = augment_examples(GIS_EXAMPLES, n_aug=10)
    pred = augment_examples(PREDICTION_EXAMPLES, n_aug=10)
    metric = augment_examples(DATA_METRIC_EXAMPLES, n_aug=10)
    faq = augment_examples(FAQ_EXAMPLES, n_aug=10)
    unknown = augment_examples(UNKNOWN_EXAMPLES, n_aug=10)

    X = gis + pred + metric + faq + unknown
    y = (
        ["GIS_VIEW"] * len(gis)
        + ["PREDICTION"] * len(pred)
        + ["DATA_METRIC"] * len(metric)
        + ["FAQ"] * len(faq)
        + ["UNKNOWN_INTENT"] * len(unknown)
    )

    pairs = list(zip(X, y))
    random.seed(42)
    random.shuffle(pairs)
    X_shuffled, y_shuffled = zip(*pairs)

    X_train, X_test, y_train, y_test = train_test_split(
        list(X_shuffled), list(y_shuffled), test_size=0.2, random_state=42, stratify=list(y_shuffled)
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
    ])

    print("Training intent classifier on CARIVIX enterprise domain dataset...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved intent classifier to: {MODEL_PATH}")
    return pipeline

if __name__ == "__main__":
    train_and_save()
