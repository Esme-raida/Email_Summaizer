import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline

tokenizer = RegexpTokenizer(r"\w+")

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())

def fit_vectorizer(cleaned_text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(cleaned_text)
    return vectorizer

def is_valid_sentence(sent):
    return (
        4 < len(sent.split()) < 40 and
        not re.search(r"@|/Enron|Corp/Enron|Enron@|[A-Za-z]+/[A-Za-z]+", sent)
    )

def summarize_email(text, vectorizer, top_n=3):
    if not text or not isinstance(text, str):
        return ""

    vocab = vectorizer.vocabulary_
    sentences = split_sentences(text)
    sentences = [s for s in sentences if is_valid_sentence(s)]

    if len(sentences) == 0:
        return ""

    sentence_scores = []
    for sentence in sentences:
        words = tokenizer.tokenize(sentence.lower())
        words = [w for w in words if w in vocab]
        score = np.mean([vectorizer.idf_[vocab[w]] for w in words]) if words else 0
        sentence_scores.append(score)

    ranked_sentences = [s for _, s in sorted(zip(sentence_scores, sentences), reverse=True)]
    return " ".join(ranked_sentences[:top_n])

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def summarize_abstractive(text):
    if not isinstance(text, str) or len(text.strip()) < 30:
        return ""
    
    input_len = len(text.strip().split())
    adjusted_max = min(50, int(input_len * 0.6))  # e.g. 60% of input length
    adjusted_max = max(adjusted_max, 15)  # ensure it's not too short

    try:
        return summarizer(text, max_length=adjusted_max, min_length=15, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return f"[ERROR: {e}]"