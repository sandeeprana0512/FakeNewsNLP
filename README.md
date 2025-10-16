# 📰 Fake News Detection Using NLP and Machine Learning

This project performs an in-depth analysis of a small dataset of news articles classified as either Fake News or Factual News, using Natural Language Processing (NLP), sentiment analysis, and machine learning.
---
## 📁 Dataset Overview

File: fake_news_data.csv

Rows: 198 articles

Columns:

title: Headline of the article

text: Full text content

date: Date of publication

fake_or_factual: Label indicating whether the news is fake or factual

## 📊 Initial Data Exploration

Displayed article distribution using bar plots.

Checked data types and ensured no missing values.

Used Seaborn and Matplotlib for visualization.

## 🧠 NLP Preprocessing (spaCy, NLTK)

Tokenization, POS tagging, and Named Entity Recognition (NER) using spaCy.

Created separate DataFrames for tokens and tags from both fake and factual articles.

Identified and visualized the most frequent:

Parts of Speech (POS)

Named Entities (ORG, GPE, PERSON, etc.)

## 🧹 Text Cleaning

Removed news agency headers (e.g., "Reuters -").

Lowercased, removed punctuation, and stopwords.

Tokenized and lemmatized text using WordNetLemmatizer.

Stored the cleaned tokens in a new column text_clean.

## 📈 N-gram Analysis

Extracted most common unigrams and bigrams from cleaned text.

Visualized top tokens (e.g., (donald, trump), (white, house)).

## 😃 Sentiment Analysis (VADER)

Used VADER to assign sentiment scores and labels (positive, neutral, negative) to each article.

Plotted sentiment distribution overall and by news type.

## 📚 Topic Modeling
### LDA (Latent Dirichlet Allocation)

Applied LDA using Gensim on fake news articles.

Evaluated coherence scores for different topic counts.

Visualized optimal number of topics.

### LSA (Latent Semantic Analysis)

Applied LSA with TfidfModel + LsiModel.

Extracted topics and analyzed word importance.

## 🤖 Classification Models
### Logistic Regression

Vectorized text using CountVectorizer.

Achieved:

Accuracy: 85%

Balanced precision/recall between fake and factual news.

### Support Vector Machine (SGDClassifier)

Slightly lower accuracy (83%) but stronger recall on fake news.

Demonstrated a good balance in identifying both classes.

## 🔧 Tech Stack

Python 3

pandas, matplotlib, seaborn

spaCy, nltk, vaderSentiment

Gensim (LDA, LSA, Coherence)

Scikit-learn (Logistic Regression, SGD Classifier)

## 📌 Key Results

Fake news showed strong NER patterns (e.g., emphasis on political figures).

Sentiment distribution was more negative in fake news.

Classification models achieved ~85% accuracy with basic Bag-of-Words features.

## 📂 File Structure
```
📁 your-repo-name/
├── fake_news_data.csv
├── fake_news_analysis.ipynb
└── README.md
```
## ✅ Future Improvements

Use TF-IDF or word embeddings (e.g., Word2Vec/BERT) for better classification.

Add explainable AI (e.g., SHAP) to interpret model decisions.

Expand dataset size for improved generalizability.

## 📜 License

This project is for educational and research purposes. Check LICENSE file for more information (if provided).
