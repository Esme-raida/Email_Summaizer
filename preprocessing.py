import pandas as pd #used for handling tabular data
import re #regex library used for text pattern matching
import html #used for processing unescapes html entities
import nltk #this is the natural language toolkit used for tokenization, stemming etc.
import numpy as np #used for numerical operations
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


###Extraction of structured email fields
def extract_email_fields(message):
    #creating a dictionary to store the extracted fields
    fields = {
        "message_id": None,
        "date": None,
        "from": None,
        "to": None,
        "subject": None,
        "body": None
    }
### using the regex library to extract the contents in our message
    try:
        fields["message_id"] = re.search(r"Message-ID:\s*<(.*?)>", message, re.IGNORECASE).group(1)
    except:
        print("Failed to extract 'message_id'")
    try:
        fields["date"] = re.search(r"Date:\s*(.*)", message, re.IGNORECASE).group(1).split("\n")[0]
    except:
        print("Failed to extract 'date'")
    try:
        fields["from"] = re.search(r"From:\s*(.*)", message, re.IGNORECASE).group(1).split("\n")[0]
    except:
        print("Failed to extract 'from'")
    try:
        fields["to"] = re.search(r"To:\s*(.*)", message, re.IGNORECASE).group(1).split("\n")[0]
    except:
        print("Failed to extract 'to'")
    try:
        fields["subject"] = re.search(r"^Subject:\s*(.*)", message, re.IGNORECASE | re.MULTILINE)
    except:
        print("Failed to extract subject")
    try:
        ### body is everything after the last header (so we assume it's after the last "\n\n")
        fields["body"] = message.split("\n\n", 1)[1].strip()
    except:
        print("Failed to extract body")
        #converts the dictionary into a pandas series like a row
    return pd.Series(fields)


###Cleaning function
def clean_email_text(text):
    if not isinstance(text, str):
        return ""
    import re, html
    text = html.unescape(text)
    # Remove HTML tags and <<filename>> tags
    text = re.sub(r"<<.*?>>", "", text)
    text = re.sub(r"<.*?>", "", text)
    # Replace links, emails, phones
    text = re.sub(r"https?://\S+", "[LINK]", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", text)
    # Remove reply markers and headers
    text = re.sub(r"(?i)^.*Forwarded by.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*----+Original Message----+.*$", "", text, flags=re.MULTILINE)
    re.sub(r"(?im)^\s*(From|To|Subject|Sent):\s+.+$", "", text)
    # Remove metadata-style header lines
    header_keywords = [
        "Message-ID:", "Date:", "From:", "To:", "Subject:", "Mime-Version:",
        "Content-Type:", "Content-Transfer-Encoding:", "X-", "Bcc:", "Cc:",
        "Folder:", "FileName:", "Origin:", "---- Forwarded"
    ]
    pattern = re.compile(r"(?i)^(" + "|".join(map(re.escape, header_keywords)) + r").*$", re.MULTILINE)
    text = pattern.sub("", text)
    # Remove reply prefix markers
    text = re.sub(r"^\s*>.*$", "", text, flags=re.MULTILINE)
    # Remove long contacts lists (5+ emails separated by commas/semicolons/spaces)
    text = re.sub(r"(([\w\.-]+/[A-Za-z]+/[A-Za-z]+@[\w\.-]+)[,;\s]*){5,}", "[CONTACT_LIST]", text)
    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Remove common sign-off lines
    text = re.sub(r"(?i)(\n|\s)(thanks|regards|cheers|sincerely)[\s,]*[\w\s]*$", "", text)
    # Collapse multiple newlines and spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def final_email_filter(df, 
                       text_col="cleaned_text", 
                       parts_col="email_parts",
                       min_length=100,
                       min_word_count=30,
                       min_alpha_ratio=0.5,
                       max_placeholder_ratio=0.2,
                       max_symbol_ratio=0.1,
                       min_parts=1,
                       max_parts=10):
    
    df = df.copy()
    
    df["length"] = df[text_col].str.len()
    df = df[df["length"] >= min_length]
    print("After length filter:", len(df))
    
    df["word_count"] = df[text_col].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= min_word_count]
    print("After word count filter:", len(df))
    
    df["alpha_ratio"] = df[text_col].apply(
        lambda x: sum(c.isalpha() for c in x) / len(x) if len(x) > 0 else 0
    )
    df = df[df["alpha_ratio"] >= min_alpha_ratio]
    print("After alpha ratio filter:", len(df))
    
    placeholders = ["[EMAIL]", "[LINK]", "[PHONE]", "[CONTACT_LIST]"]
    df["placeholder_count"] = df[text_col].apply(
        lambda x: sum(x.count(p) for p in placeholders)
    )
    df["placeholder_ratio"] = df["placeholder_count"] / df["length"]
    df = df[~((df["length"] < 150) & (df["placeholder_ratio"] > 0.3))]
    print("After placeholder ratio filter:", len(df))
    
    symbol_chars = set("@#$%^&*+=~|<>/")
    df["symbol_count"] = df[text_col].apply(
        lambda x: sum(c in symbol_chars for c in x)
    )
    df["symbol_ratio"] = df["symbol_count"] / df["length"]
    df = df[df["symbol_ratio"] <= max_symbol_ratio]
    print("After symbol ratio filter:", len(df))

    df["unique_word_ratio"] = df[text_col].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
    df = df[df["unique_word_ratio"] > 0.5]
    print("After unique word ratio filter:", len(df))
 
    df["has_sentence"] = df[text_col].str.contains(r"[.!?]")
    df = df[df["has_sentence"]]
    print("After sentence punctuation filter:", len(df))
    
    if parts_col in df.columns:
        df = df[df[parts_col].apply(lambda x: isinstance(x, list) and min_parts <= len(x) <= max_parts)]
        print("After email parts count filter:", len(df))
    
    drop_cols = ["length", "word_count", "alpha_ratio", 
                 "placeholder_count", "placeholder_ratio", 
                 "symbol_count", "symbol_ratio", "has_sentence"]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    return df.reset_index(drop=True)



###This helps us to seperate many messages that are in a thread and isolate replies
###like, the sender sends and receives reply all in a message/thread 
###find marker markers(e.g. From) to seperate different emails
def split_email_chain(cleaned_text):
    split_patterns = r"(?=\nFrom: .+\nSent: .+\nTo: .+\nSubject:)"
    parts = re.split(split_patterns, cleaned_text)
    return [part.strip() for part in parts if part.strip()]


stop_words = set(stopwords.words("english"))###makes a set of english stopwords
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

# preprocessing Function
def nltk_preprocess(text):
    if not isinstance(text, str):
        return ""
    
    tokens = tokenizer.tokenize(text.lower())###Tokenize and coverts it to lowercase
    tokens = [t for t in tokens if t not in stop_words]###removes stopwords
    tokens = [t for t in tokens if not t.isnumeric()] ###removes pure numerical tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]###lemmatize

    return " ".join(tokens)