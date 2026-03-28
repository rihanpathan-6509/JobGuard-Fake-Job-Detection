# JobGuard — Fake Job Posting Detector

A machine learning web app that detects fraudulent job postings using NLP and a rule-based scoring system.

Built as part of a 4-week AIML project challenge.

---

## Problem Statement

Fake job postings are a growing threat — they waste applicants' time, steal personal data, and in some cases demand upfront payments. This project aims to automatically classify job postings as **Real**, **Suspicious**, or **Fraudulent** using a combination of ML and heuristic rules.

---

## Approach

### 1. Text Preprocessing
- Job title + description combined
- Lowercased, special characters removed, HTML entities cleaned

### 2. TF-IDF Vectorization
- Converts cleaned text into 5,000 numerical features
- Captures word importance across all postings

### 3. Engineered Features (5 extra)
| Feature | Why it matters |
|---|---|
| Word count | Fake jobs tend to have shorter descriptions |
| Char count | Correlates with posting quality |
| Employment type mentioned | Legit jobs usually specify this |
| Industry mentioned | Scam posts often skip industry |
| Job function mentioned | Another legitimacy signal |

### 4. Random Forest Classifier
- 100 decision trees, all voting together
- Trained on TF-IDF + engineered features combined
- Balanced class weights to handle dataset imbalance

### 5. Rule-Based Scoring Layer
A weighted system that flags scam-specific patterns the ML model may miss:

| Weight | Example Patterns |
|---|---|
| 3 (Hard fraud) | "registration fee", "Aadhar card details", "bank account number" |
| 2 (Strong signal) | "WhatsApp only", "urgent hiring", "earn X per week" |
| 1 (Soft warning) | "refundable deposit", "no experience required" |

**Final verdict thresholds:**
- Score 0-1 + ML says Real → Real
- Score 2-3 or mild ML doubt → Suspicious
- Score 4+ or hard flag hit → Fraudulent

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 93% |
| Fake Recall | 83% |
| Training Samples | 17,880 |
| Features | TF-IDF (5k) + 5 engineered |

---

## Project Structure

```
├── app.py                                  # Streamlit web app
├── model.pkl                               # Trained Random Forest model
├── tfidf.pkl                               # Fitted TF-IDF vectorizer
├── scaler.pkl                              # Fitted StandardScaler
├── Fake_Job_Detection_Random_Forest.ipynb  # Training notebook
├── Fake_Real_Job_Posting.csv               # Dataset
└── requirements.txt                        # Dependencies
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/rihanpathan-6509/JobGuard-Fake-Job-Detection.git
cd jobguard-fake-job-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Dependencies

```
streamlit
scikit-learn
pandas
numpy
scipy
```

---

## Dataset

- **Source:** [Kaggle — Fraudulent Job Posting](https://www.kaggle.com/datasets/subhajournal/fraudulent-job-posting))
- **Size:** 17,878 job postings
- **Label:** `fraudulent` (1 = fake, 0 = real)
- **Class imbalance:** ~95% real, ~5% fake — handled via balanced class weights

---

## Team

Built by Team 5 — First Year BTech, AIML
Walchand College of Engineering, Sangli

Team Members
1) Akash Loni
2) Rihan Pathan
3) Sneha Ingle
4) Swanand Jaju
5) Vedant Aherkar

---

## Note

This model has 93% accuracy on test data. Always do your own research before applying to any job posting.
