import os
import json
import glob
import re

import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


SENTIMENT_REVIEW_THRESHOLD = 0.70
COMPLAINT_REVIEW_THRESHOLD = 0.70
OUTPUT_FILE = "output.json"


print("=" * 70)
print("DOWNLOADING DATASET")
print("=" * 70)

path = kagglehub.dataset_download(
    "crowdflower/twitter-airline-sentiment"
)

print(f"Dataset downloaded to: {path}")


csv_files = glob.glob(
    os.path.join(path, "**", "*.csv"),
    recursive=True
)

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the downloaded Kaggle dataset."
    )

print("\nCSV files found:")

for file in csv_files:
    print(" -", file)

csv_path = csv_files[0]

print(f"\nUsing CSV: {csv_path}")


df = pd.read_csv(csv_path)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


text_column = None
sentiment_column = None
reason_column = None

for col in df.columns:

    col_lower = col.lower().strip()

    if col_lower == "text":
        text_column = col

    elif col_lower == "airline_sentiment":
        sentiment_column = col

    elif col_lower == "negativereason":
        reason_column = col


if text_column is None:
    raise ValueError("Could not find 'text' column.")

if sentiment_column is None:
    raise ValueError(
        "Could not find 'airline_sentiment' column."
    )

print("\nDetected columns:")
print("Text:", text_column)
print("Sentiment:", sentiment_column)
print("Negative reason:", reason_column)


df = df[
    [text_column, sentiment_column]
    + ([reason_column] if reason_column else [])
].copy()

df.rename(
    columns={
        text_column: "text",
        sentiment_column: "sentiment"
    },
    inplace=True
)

if reason_column:
    df.rename(
        columns={
            reason_column: "negative_reason"
        },
        inplace=True
    )
else:
    df["negative_reason"] = None


df["text"] = df["text"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].fillna("").astype(str)

df = df[df["text"].str.strip() != ""]
df = df[df["sentiment"].str.strip() != ""]


def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)


print("\n" + "=" * 70)
print("SENTIMENT DISTRIBUTION")
print("=" * 70)

print(df["sentiment"].value_counts())


print("\n" + "=" * 70)
print("TRAINING SENTIMENT MODEL")
print("=" * 70)


X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


sentiment_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


sentiment_model.fit(X_train, y_train)


sentiment_predictions = sentiment_model.predict(X_test)

sentiment_accuracy = accuracy_score(
    y_test,
    sentiment_predictions
)

print("\nSentiment Accuracy:")
print(f"{sentiment_accuracy:.4f}")

print("\nSentiment Classification Report:")

print(
    classification_report(
        y_test,
        sentiment_predictions
    )
)


print("\n" + "=" * 70)
print("TRAINING COMPLAINT CATEGORY MODEL")
print("=" * 70)


negative_df = df[
    df["sentiment"].str.lower() == "negative"
].copy()


print(
    "Number of negative tweets:",
    len(negative_df)
)


def map_complaint_category(reason):

    if pd.isna(reason):
        return "other"

    reason = str(reason).lower().strip()

    if "delay" in reason:
        return "flight_delay"

    if "cancel" in reason:
        return "flight_cancellation"

    if (
        "luggage" in reason
        or "baggage" in reason
    ):
        return "lost_baggage"

    if "customer service" in reason:
        return "customer_service"

    if "flight attendant" in reason:
        return "flight_experience"

    if (
        "booking" in reason
        or "reservation" in reason
    ):
        return "booking"

    if (
        "refund" in reason
        or "compensation" in reason
    ):
        return "refund_compensation"

    if (
        "bad flight" in reason
        or "aircraft" in reason
        or "seat" in reason
    ):
        return "flight_experience"

    return "other"


negative_df["complaint_category"] = (
    negative_df["negative_reason"]
    .apply(map_complaint_category)
)


print("\nComplaint category distribution:")

print(
    negative_df["complaint_category"]
    .value_counts()
)


X_complaint = negative_df["clean_text"]
y_complaint = negative_df["complaint_category"]


Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_complaint,
    y_complaint,
    test_size=0.20,
    random_state=42,
    stratify=y_complaint
)


complaint_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


complaint_model.fit(
    Xc_train,
    yc_train
)


complaint_predictions = complaint_model.predict(
    Xc_test
)

complaint_accuracy = accuracy_score(
    yc_test,
    complaint_predictions
)

print("\nComplaint Classification Accuracy:")
print(f"{complaint_accuracy:.4f}")

print("\nComplaint Classification Report:")

print(
    classification_report(
        yc_test,
        complaint_predictions,
        zero_division=0
    )
)


def needs_human_review(
    sentiment,
    sentiment_confidence,
    complaint_confidence=None
):

    if sentiment_confidence < SENTIMENT_REVIEW_THRESHOLD:
        return True

    if sentiment != "negative":
        return False

    if complaint_confidence is not None:

        if complaint_confidence < COMPLAINT_REVIEW_THRESHOLD:
            return True

    return False


def analyze_tweet(text):

    cleaned = clean_text(text)

    sentiment = sentiment_model.predict(
        [cleaned]
    )[0]

    probabilities = sentiment_model.predict_proba(
        [cleaned]
    )[0]

    sentiment_confidence = float(
        max(probabilities)
    )

    complaint_category = None
    complaint_confidence = None

    if sentiment.lower() == "negative":

        complaint_category = complaint_model.predict(
            [cleaned]
        )[0]

        complaint_probabilities = (
            complaint_model.predict_proba(
                [cleaned]
            )[0]
        )

        complaint_confidence = float(
            max(complaint_probabilities)
        )

    human_review = needs_human_review(
        sentiment=sentiment,
        sentiment_confidence=sentiment_confidence,
        complaint_confidence=complaint_confidence
    )

    result = {
        "text": text,
        "sentiment": sentiment,
        "sentiment_confidence": round(
            sentiment_confidence,
            4
        ),
        "complaint_category": complaint_category,
        "complaint_confidence": (
            round(complaint_confidence, 4)
            if complaint_confidence is not None
            else None
        ),
        "human_review": human_review
    }

    return result


print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)


sample_tweets = [
    "My flight was delayed for 6 hours and nobody helped me.",
    "The flight attendant was amazing and very helpful!",
    "I am extremely disappointed with the customer service.",
    "My luggage was lost and I have been waiting for hours.",
    "Great flight! Everything was perfect.",
    "I don't know what is happening with my booking."
]


sample_results = []


for tweet in sample_tweets:

    result = analyze_tweet(tweet)

    sample_results.append(result)

    print("\nTweet:")
    print(tweet)

    print("\nPrediction:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )


print("\n" + "=" * 70)
print("ANALYZING ENTIRE DATASET")
print("=" * 70)


results = []


for index, row in df.iterrows():

    result = analyze_tweet(
        row["text"]
    )

    result["original_sentiment"] = row["sentiment"]

    if (
        row["sentiment"].lower() == "negative"
        and pd.notna(row["negative_reason"])
    ):
        result["original_complaint_category"] = (
            map_complaint_category(
                row["negative_reason"]
            )
        )
    else:
        result["original_complaint_category"] = None

    results.append(result)


print(
    f"Processed {len(results)} tweets."
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

print(
    f"\nJSON output saved to: {OUTPUT_FILE}"
)

print(
    f"\nSentiment model accuracy: "
    f"{sentiment_accuracy:.4f}"
)

print(
    f"Complaint model accuracy: "
    f"{complaint_accuracy:.4f}"
)

print(
    "\nHuman review threshold:",
    SENTIMENT_REVIEW_THRESHOLD
)

print(
    "Complaint review threshold:",
    COMPLAINT_REVIEW_THRESHOLD
)