from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(texts):
    
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names


def get_top_tfidf_terms(tfidf_vector, feature_names, top_n=10):
    """Get top N TF-IDF terms from a single tf-idf row."""
    row = tfidf_vector.toarray().flatten()
    top_indices = row.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices]