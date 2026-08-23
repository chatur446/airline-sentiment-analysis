# ✈️ Airline Sentiment & Complaint Classification

An NLP-based airline customer feedback analysis system that automatically classifies airline tweets into **Positive, Negative, or Neutral sentiment**.

For negative feedback, the system performs a second-level classification to identify the **complaint category** and determines whether the prediction requires **human review** based on model confidence.

The project uses the **Twitter Airline Sentiment Dataset** from Kaggle and implements a two-stage machine learning pipeline using **TF-IDF and Logistic Regression**.

---

## 📌 Project Overview

Airlines receive thousands of customer comments and complaints through social media. Manually analyzing every message is time-consuming and difficult to scale.

This project automates the initial analysis of airline customer feedback.

The system performs three main tasks:

1. **Sentiment Classification**
   - Positive
   - Negative
   - Neutral

2. **Complaint Classification**
   - Performed only when the sentiment is negative
   - Identifies the type of complaint

3. **Human Review Detection**
   - Uses prediction confidence to determine whether a prediction should be manually reviewed

### Overall Pipeline

```text
Customer Tweet
      │
      ▼
Text Preprocessing
      │
      ▼
Sentiment Classification
      │
      ├───────────────┬────────────────┐
      │               │                │
      ▼               ▼                ▼
   Positive        Neutral          Negative
                                       │
                                       ▼
                              Complaint Classification
                                       │
                                       ▼
                              Confidence Evaluation
                                       │
                              ┌────────┴────────┐
                              │                 │
                              ▼                 ▼
                       No Human Review    Human Review
```

---

# 🎯 Project Objectives

The main objectives of this project are:

- Automatically analyze airline customer tweets.
- Classify each tweet as positive, negative, or neutral.
- Identify the complaint category for negative tweets.
- Calculate prediction confidence.
- Automatically identify uncertain predictions.
- Flag low-confidence predictions for human review.
- Generate structured JSON output.
- Build a lightweight and explainable NLP pipeline.

---

# ✨ Features

## 1. Sentiment Classification

The system classifies tweets into three sentiment categories:

| Sentiment | Description |
|---|---|
| Positive | Customer expresses satisfaction or a positive experience |
| Negative | Customer expresses dissatisfaction or a complaint |
| Neutral | Customer provides information without strong positive or negative emotion |

---

## 2. Complaint Classification

When a tweet is classified as negative, it is passed to a second classifier.

The system identifies the complaint category.

Supported categories include:

| Category | Description |
|---|---|
| `flight_delay` | Complaints related to delayed flights |
| `flight_cancellation` | Complaints related to cancelled flights |
| `lost_baggage` | Lost or missing baggage complaints |
| `customer_service` | Complaints about customer support or service |
| `flight_experience` | Complaints about flight experience, seats, staff, aircraft, etc. |
| `booking` | Booking and reservation problems |
| `refund_compensation` | Refund or compensation-related complaints |
| `other` | Complaints that do not match the predefined categories |

---

# 🤖 Machine Learning Architecture

The project follows a **two-stage hierarchical classification approach**.

## Stage 1: Sentiment Classification

The first model predicts:

```text
Positive
Negative
Neutral
```

The model uses:

- TF-IDF Vectorization
- Logistic Regression

---

## Stage 2: Complaint Classification

Only tweets predicted as negative are passed to the second model.

The second model predicts:

```text
flight_delay
flight_cancellation
lost_baggage
customer_service
flight_experience
booking
refund_compensation
other
```

This approach separates sentiment detection from complaint classification.

---

# 🧠 Why Two Models?

Instead of trying to classify everything with one model, this project uses two specialized classifiers.

```text
                  Tweet
                    │
                    ▼
          Sentiment Classifier
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    Positive     Neutral     Negative
                                │
                                ▼
                     Complaint Classifier
                                │
                                ▼
                       Complaint Category
```

This makes the system:

- Easier to understand
- Easier to debug
- Easier to extend
- More suitable for real-world customer-support workflows

---

# 🔤 Text Preprocessing

Before training the machine learning models, tweets are cleaned.

The preprocessing pipeline includes:

### Lowercasing

```text
"My Flight Was DELAYED"
```

becomes:

```text
"my flight was delayed"
```

### URL Removal

URLs are removed because they usually do not provide useful information for sentiment classification.

### Mention Removal

Twitter mentions such as:

```text
@airline
```

are removed.

### Hashtag Cleaning

The `#` symbol is removed while keeping the hashtag text.

Example:

```text
#badservice
```

becomes:

```text
badservice
```

### Whitespace Normalization

Extra spaces are removed to produce clean text.

---

# 🔢 Feature Engineering

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.

The vectorizer uses:

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
```

### Unigrams

Individual words are represented as features.

Example:

```text
flight
delayed
service
```

### Bigrams

Two-word combinations are also considered.

Example:

```text
flight delayed
bad service
customer service
```

Using both unigrams and bigrams helps the model capture useful phrases.

---

# 📈 Machine Learning Algorithm

## Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

Logistic Regression works well with sparse TF-IDF text representations and is:

- Fast
- Lightweight
- Easy to train
- Easy to interpret
- Suitable for multi-class classification

The model produces class probabilities, which are also used to calculate prediction confidence.

---

# 🎯 Confidence Scoring

The system calculates a confidence score for its predictions.

For example:

```json
{
    "sentiment": "negative",
    "sentiment_confidence": 0.94,
    "complaint_category": "flight_delay",
    "complaint_confidence": 0.91
}
```

A higher confidence means the model is more certain about its prediction.

However, confidence should not be interpreted as guaranteed correctness.

---

# 👨‍💼 Human-in-the-Loop System

One of the important features of this project is **human review detection**.

A machine learning model should not blindly make every decision, especially in customer-service applications.

The project uses confidence thresholds.

Default thresholds:

```text
Sentiment confidence threshold = 0.70
Complaint confidence threshold = 0.70
```

For a negative tweet:

```text
Sentiment confidence >= 0.70
AND
Complaint confidence >= 0.70
```

Then:

```text
Human Review = false
```

If either confidence score is below the threshold:

```text
Human Review = true
```

### Example: No Human Review

```text
Sentiment Confidence = 0.93
Complaint Confidence = 0.91

Human Review = false
```

### Example: Human Review Required

```text
Sentiment Confidence = 0.91
Complaint Confidence = 0.54

Human Review = true
```

This creates a **human-in-the-loop** system where humans can focus on uncertain predictions.

---

# 🔐 Why Human Review Is Important

Natural language is difficult for machine learning models to understand perfectly.

Tweets may contain:

- Sarcasm
- Ambiguous language
- Spelling mistakes
- Short messages
- Multiple complaints
- Context-dependent statements
- Mixed emotions
- Multiple possible complaint categories

For example:

```text
"Great, another 5 hour delay. Amazing service."
```

A traditional text classifier could incorrectly interpret words such as `Great` and `Amazing` without understanding the sarcastic context.

Human review provides an additional safety layer.

---

# 📊 Dataset

The project uses the:

**CrowdFlower Twitter Airline Sentiment Dataset**

Kaggle dataset identifier:

```text
crowdflower/twitter-airline-sentiment
```

The dataset contains airline-related tweets with sentiment labels and complaint information.

The relevant fields include:

```text
airline_sentiment
negativereason
text
```

The dataset is downloaded automatically using KaggleHub.

---

# 📥 Dataset Download

The project uses:

```python
import kagglehub

path = kagglehub.dataset_download(
    "crowdflower/twitter-airline-sentiment"
)
```

The dataset does not need to be manually downloaded.

When the program runs, KaggleHub downloads the latest available version of the dataset.

---

# 🔄 Complaint Category Mapping

The original dataset contains specific negative-reason labels.

These are mapped into standardized complaint categories used by the application.

Example mappings:

| Original Reason | Application Category |
|---|---|
| Flight Delayed | `flight_delay` |
| Cancelled Flight | `flight_cancellation` |
| Lost Luggage | `lost_baggage` |
| Customer Service Issue | `customer_service` |
| Flight Attendant Complaints | `flight_experience` |
| Bad Flight | `flight_experience` |
| Booking Problem | `booking` |
| Refund / Compensation | `refund_compensation` |
| Other / Unknown | `other` |

The standardized categories make the output easier to use in downstream applications.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │      Kaggle Dataset     │
                    │ Twitter Airline Dataset │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Data Processing     │
                    │                         │
                    │ • Load Dataset          │
                    │ • Clean Text            │
                    │ • Handle Labels         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      TF-IDF Vectorizer  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Sentiment Classifier    │
                    │                         │
                    │ Logistic Regression     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
                Positive      Neutral      Negative
                                              │
                                              ▼
                                  ┌─────────────────────┐
                                  │ Complaint Classifier│
                                  │                     │
                                  │ Logistic Regression │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  Complaint Category
                                             │
                                             ▼
                                  Confidence Evaluation
                                             │
                                             ▼
                                  Human Review Decision
                                             │
                                             ▼
                                       JSON Output
```

---

# 📤 JSON Output

The final predictions are generated in JSON format.

Example:

```json
{
    "text": "My flight was delayed for 6 hours and nobody helped me.",
    "sentiment": "negative",
    "sentiment_confidence": 0.9821,
    "complaint_category": "flight_delay",
    "complaint_confidence": 0.9452,
    "human_review": false
}
```

---

# 🟢 Positive Example

### Input

```text
"The flight attendant was amazing and very helpful!"
```

### Output

```json
{
    "text": "The flight attendant was amazing and very helpful!",
    "sentiment": "positive",
    "sentiment_confidence": 0.97,
    "complaint_category": null,
    "complaint_confidence": null,
    "human_review": false
}
```

Since the sentiment is positive, complaint classification is not required.

---

# 🟡 Neutral Example

### Input

```text
"What time does the flight from Delhi to Mumbai depart?"
```

### Output

```json
{
    "text": "What time does the flight from Delhi to Mumbai depart?",
    "sentiment": "neutral",
    "sentiment_confidence": 0.88,
    "complaint_category": null,
    "complaint_confidence": null,
    "human_review": false
}
```

---

# 🔴 Negative Example

### Input

```text
"My flight was delayed for 6 hours and nobody helped me."
```

### Output

```json
{
    "text": "My flight was delayed for 6 hours and nobody helped me.",
    "sentiment": "negative",
    "sentiment_confidence": 0.98,
    "complaint_category": "flight_delay",
    "complaint_confidence": 0.94,
    "human_review": false
}
```

---

# 🟡 Human Review Example

If the sentiment classifier is confident but the complaint classifier is uncertain:

```json
{
    "text": "Nobody seems to know what happened with my issue.",
    "sentiment": "negative",
    "sentiment_confidence": 0.91,
    "complaint_category": "customer_service",
    "complaint_confidence": 0.58,
    "human_review": true
}
```

The prediction is flagged because the complaint confidence is below the configured threshold.

---

# 📈 Model Evaluation

The project evaluates the performance of both machine learning models.

## Sentiment Model Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

Example:

```text
Sentiment Classification Report

              precision    recall    f1-score

negative         0.88       0.91       0.89
neutral          0.70       0.65       0.67
positive         0.86       0.82       0.84
```

The actual results depend on the train/test split and model configuration.

---

## Complaint Model Metrics

The complaint classifier is evaluated separately using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

This is important because complaint classification is a separate task from sentiment classification.

---

# 🔁 Complete Processing Pipeline

```text
                 ┌──────────────────┐
                 │  Kaggle Dataset  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Load CSV Dataset │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text Preprocessing│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ TF-IDF Features  │
                 └────────┬─────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Sentiment Classification │
              └────────────┬─────────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
         Positive       Neutral       Negative
             │             │             │
             │             │             ▼
             │             │    ┌───────────────────┐
             │             │    │Complaint Classifier│
             │             │    └─────────┬─────────┘
             │             │              │
             │             │              ▼
             │             │     Complaint Category
             │             │
             └─────────────┴──────────────┐
                                          │
                                          ▼
                                Confidence Evaluation
                                          │
                                          ▼
                                  Human Review Flag
                                          │
                                          ▼
                                    JSON Output
```

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization
- Train/Test Split
- Classification Metrics

## Data Processing

- Pandas
- Regular Expressions (`re`)

## Dataset Management

- KaggleHub
- Kaggle Twitter Airline Sentiment Dataset

## Output Format

- JSON

## Development Tools

- Git
- GitHub
- Python Virtual Environment

---

# 📁 Project Structure

```text
airline-sentiment/
│
├── main.py
├── README.md
├── .gitignore
│
├── output.json
│
└── venv/
```

### `main.py`

The main Python program contains the complete pipeline:

- Dataset download
- Dataset loading
- Text preprocessing
- Feature extraction
- Sentiment model training
- Complaint model training
- Model evaluation
- Prediction
- Confidence calculation
- Human-review decision
- JSON generation

### `README.md`

Project documentation containing:

- Project description
- Architecture
- Installation instructions
- Usage instructions
- Technology stack
- Output examples
- Future improvements

### `.gitignore`

Contains files and folders that should not be uploaded to GitHub.

Example:

```text
venv/
__pycache__/
*.pyc
.kaggle/
```

### `output.json`

Generated prediction output.

---

# ⚙️ Installation

## Prerequisites

Make sure you have installed:

- Python 3.x
- Git
- Internet connection

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/airline-sentiment-analysis.git
```

Replace `YOUR_USERNAME` with your GitHub username.

Then:

```bash
cd airline-sentiment-analysis
```

---

# 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

---

# 3. Activate the Virtual Environment

Windows CMD:

```bash
venv\Scripts\activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

After activation, you should see:

```text
(venv)
```

at the beginning of your terminal.

---

# 4. Install Dependencies

Install the required Python packages:

```bash
pip install kagglehub pandas scikit-learn
```

You can also upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# ▶️ Running the Project

After activating the virtual environment and installing the dependencies:

```bash
python main.py
```

The program will:

1. Download the Kaggle dataset.
2. Locate the dataset file.
3. Load the data.
4. Clean the tweets.
5. Train the sentiment classifier.
6. Train the complaint classifier.
7. Evaluate the models.
8. Generate predictions.
9. Calculate confidence scores.
10. Determine whether human review is required.
11. Save the predictions as JSON.

---

# 📦 Required Python Packages

The main dependencies are:

```text
kagglehub
pandas
scikit-learn
```

The project uses the following Python modules:

```python
import os
import re
import json
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
```

---

# 🧪 Example Command

```bash
python main.py
```

Expected workflow:

```text
Downloading dataset...
Loading dataset...
Preprocessing text...
Training sentiment classifier...
Evaluating sentiment model...
Training complaint classifier...
Evaluating complaint model...
Generating predictions...
Saving output.json...
Done!
```

The exact console output may vary depending on the implementation.

---

# 📊 Example Output Structure

The generated JSON follows a structure similar to:

```json
{
    "text": "My flight was delayed for 6 hours.",
    "sentiment": "negative",
    "sentiment_confidence": 0.95,
    "complaint_category": "flight_delay",
    "complaint_confidence": 0.92,
    "human_review": false
}
```

For positive and neutral tweets:

```json
{
    "text": "The flight was excellent.",
    "sentiment": "positive",
    "sentiment_confidence": 0.96,
    "complaint_category": null,
    "complaint_confidence": null,
    "human_review": false
}
```

---

# 🚀 Future Improvements

The current project uses a lightweight traditional machine learning approach. Several improvements can be made.

## 1. Transformer-Based NLP Models

The TF-IDF + Logistic Regression approach can be replaced or compared with transformer models such as:

- BERT
- RoBERTa
- DistilBERT
- BERTweet

These models can better understand context and complex language.

---

## 2. Multi-Label Complaint Classification

Currently, a negative tweet receives one complaint category.

However, a tweet may contain multiple complaints.

Example:

```text
"My flight was cancelled and I still haven't received my refund."
```

This could be classified as:

```text
flight_cancellation
refund_compensation
```

A future version could use multi-label classification.

---

## 3. REST API

The trained models can be deployed using FastAPI.

Possible architecture:

```text
Client
   │
   ▼
FastAPI
   │
   ▼
Sentiment Model
   │
   ▼
Complaint Model
   │
   ▼
Confidence Evaluation
   │
   ▼
JSON Response
```

---

## 4. Web Dashboard

A frontend dashboard could display:

- Total tweets analyzed
- Positive / negative / neutral distribution
- Complaint categories
- Confidence scores
- Human-review queue
- Airline-specific statistics
- Complaint trends

---

## 5. Model Persistence

The trained models can be saved using `joblib`.

This would prevent the application from retraining the models every time it starts.

Possible files:

```text
sentiment_model.pkl
complaint_model.pkl
tfidf_sentiment.pkl
tfidf_complaint.pkl
```

---

## 6. Human Review Dashboard

A future version could provide a dashboard where customer-service employees can review uncertain predictions.

For example:

```text
--------------------------------------------------
Tweet:
"My flight was cancelled and no one helped me."

Predicted Sentiment:
Negative

Confidence:
0.91

Complaint:
Flight Cancellation

Complaint Confidence:
0.58

Human Review:
REQUIRED
--------------------------------------------------
```

The human could then:

```text
[Accept Prediction]
[Change Category]
[Change Sentiment]
```

Corrected labels could later be used for model improvement.

---

## 7. Continuous Learning

Human corrections could be stored and periodically used to retrain the model.

```text
Prediction
    ↓
Human Review
    ↓
Correction
    ↓
Store Correct Label
    ↓
Retrain Model
    ↓
Improved Predictions
```

This would create a feedback loop for continuous improvement.

---

# ⚠️ Limitations

## Dataset Dependency

Model performance depends on the quality and distribution of the training dataset.

## Context Understanding

Traditional TF-IDF models have limited ability to understand context, sarcasm, and complex language.

## Sarcasm

Example:

```text
"Great, my flight is delayed again!"
```

The word `Great` is positive in isolation, but the overall message is negative.

## Multiple Complaints

A single tweet can contain multiple complaint types.

The current system is primarily designed for single-category complaint classification.

## Confidence Is Not Guaranteed Correctness

A confidence score such as:

```text
0.90
```

does not mean the model is guaranteed to be correct 90% of the time.

It is a model probability used to determine prediction certainty and whether human review may be appropriate.

---

# 🔒 Human Review Policy

The project intentionally avoids treating machine learning predictions as completely reliable.

The human-review mechanism is designed to identify uncertain cases.

The basic rule is:

```text
IF sentiment_confidence < threshold
    THEN human_review = true

ELSE IF sentiment == negative
    AND complaint_confidence < threshold
    THEN human_review = true

ELSE
    human_review = false
```

This makes the system more suitable for applications where incorrect automated decisions could negatively affect customers.

---

# 💼 Real-World Use Cases

This project can be adapted for:

### Airline Customer Support

Automatically identify customer complaints and route them to the appropriate department.

### Social Media Monitoring

Monitor customer sentiment toward an airline.

### Complaint Prioritization

Identify negative feedback and prioritize cases for support teams.

### Customer Experience Analytics

Analyze common sources of dissatisfaction.

### Automated Support Routing

Route complaints such as:

```text
Flight Delay
      ↓
Delay Support Team
```

or:

```text
Refund Complaint
      ↓
Finance / Refund Team
```

### Brand Monitoring

Track positive, negative, and neutral customer feedback.

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

- Natural Language Processing
- Text Classification
- Text Preprocessing
- TF-IDF
- N-grams
- Logistic Regression
- Multi-class Classification
- Hierarchical Classification
- Model Confidence
- Human-in-the-Loop AI
- Model Evaluation
- Precision
- Recall
- F1-score
- JSON Processing
- Kaggle Dataset Integration
- Python
- Git
- GitHub

---

# 🎓 Learning Outcomes

By building this project, the following concepts are demonstrated:

1. How raw text is converted into numerical machine learning features.
2. How TF-IDF represents the importance of words.
3. How Logistic Regression can be used for NLP classification.
4. How to build a multi-class sentiment classifier.
5. How to build a second-level complaint classifier.
6. How confidence scores can be used in automated systems.
7. How human review can be integrated into an AI workflow.
8. How machine learning predictions can be exported as JSON.
9. How to download and process datasets programmatically.
10. How to structure and document an NLP project.

---

# 🔍 Example End-to-End Scenario

Consider the following tweet:

```text
"My flight was cancelled and I've been waiting hours for someone to help me."
```

### Step 1: Preprocessing

```text
"my flight was cancelled and i've been waiting hours for someone to help me"
```

### Step 2: Sentiment Classification

```text
Sentiment = Negative
Confidence = 0.94
```

### Step 3: Complaint Classification

```text
Complaint = flight_cancellation
Confidence = 0.87
```

### Step 4: Human Review

Both confidence values are above `0.70`.

Therefore:

```text
Human Review = false
```

### Final JSON

```json
{
    "text": "My flight was cancelled and I've been waiting hours for someone to help me.",
    "sentiment": "negative",
    "sentiment_confidence": 0.94,
    "complaint_category": "flight_cancellation",
    "complaint_confidence": 0.87,
    "human_review": false
}
```

---

# 📌 Project Highlights

### Two-Stage NLP Classification

```text
Stage 1 → Sentiment Classification
Stage 2 → Complaint Classification
```

### Human-in-the-Loop

Low-confidence predictions are automatically flagged for manual review.

### Automated Dataset Download

The dataset is downloaded automatically using KaggleHub.

### Structured JSON Output

Predictions are generated in a machine-readable JSON format.

### Explainable Machine Learning

TF-IDF and Logistic Regression provide a lightweight and relatively interpretable baseline.

### Easy to Extend

The system can later be extended with:

- Transformers
- FastAPI
- React
- PostgreSQL
- Dashboards
- Human-review systems
- Continuous learning

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

The Twitter Airline Sentiment dataset is provided by its original dataset creator on Kaggle. Users should follow the dataset's applicable license and usage terms.

---

# 👨‍💻 Author

## Ayush Chaturvedi

**B.Tech / B.E. — Artificial Intelligence & Data Science**

Areas of interest:

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Generative AI
- Large Language Models
- Backend Development
- AI Applications

---

# ⭐ Project Summary

**Airline Sentiment & Complaint Classification** is a two-stage NLP system that converts raw airline customer tweets into structured insights.

```text
Raw Tweet
    ↓
Text Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Sentiment Classification
    ↓
┌───────────────┬───────────────┐
│               │               │
Positive      Neutral        Negative
                                │
                                ▼
                     Complaint Classification
                                │
                                ▼
                       Confidence Scoring
                                │
                                ▼
                       Human Review Check
                                │
                                ▼
                           JSON Output
```

The project demonstrates how traditional machine learning techniques such as **TF-IDF and Logistic Regression** can be combined with a **human-in-the-loop approach** to build a practical, explainable, and scalable customer feedback analysis system.

---

## ⭐ If you find this project useful, consider giving the repository a star!
