import tkinter as tk
from tkinter import scrolledtext, messagebox # A multi-line text input that supports scrolling.
#messagebox: For showing popup warnings or errors.
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from summarization import (
    split_sentences,
    fit_vectorizer,
    is_valid_sentence,
    summarize_email,
    summarize_abstractive
)

# Ensure NLTK tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
    #checks if sentence tokenizer is downloaded
except LookupError:
    nltk.download('punkt')

#----------------------- Hybrid summarization function ---------------------------#
def summarize_email(email_text, num_sentences=3):  # email summarizer function
    # Step 1: Extractive summarization
    sentences = sent_tokenize(email_text)  # split text into sentences
    sentences = [s.strip() for s in sentences if s.strip()]  # clean empty/whitespace sentences
    if len(sentences) == 0:
        return "No valid content found."
    
    # Dynamically choose how many summary sentences to return (between 1 and 5)
    num_sentences = min(5, max(1, len(sentences) // 2))

    # Compute TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Sum TF-IDF weights of each sentence
    sentence_scores = X.sum(axis=1).A1 #flattten the result to a 1D array

    # Rank sentences by their scores (highest first)
    ranked_sentences = sorted(
        zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True
    )#zip pairs sentences to their scores

    # Select top `num_sentences` sentences
    extractive_summary = " ".join(sent for sent, _ in ranked_sentences[:num_sentences])

    # Step 2: Abstractive summarization
    abstractive_summary = summarize_abstractive(extractive_summary)

    # Return the abstractive summary as the final output
    return abstractive_summary


#--------------------------------------------------------------------GUI Window setup-----------------------------------------------------------------#

# Create the main application window
root = tk.Tk()
root.title("Email Summarizer")  # window title
root.geometry("800x720")   # window size
root.configure(bg="#f5f5f5")

# Title
title_label = tk.Label(
    root,
    text="Email Summarizer",
    font=("Helvetica", 20, "bold"),
    bg="#f5f5f5",
    fg="#333"
)
title_label.pack(pady=(20, 10))

# Input
input_label = tk.Label(root, text="Paste the Email Below:", font=("Helvetica", 13), bg="#f5f5f5")
input_label.pack()
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=15, font=("Helvetica", 11))
#tk.WORD = wrap text at word boundaries, so words don’t get split mid-way.
input_text.pack(pady=(5, 10))

# Word count label
count_label = tk.Label(root, text="Original Email Words: 0 | Summary Words: 0", font=("Helvetica", 10), bg="#f5f5f5", fg="gray")
count_label.pack()


# Function to update the word/character count when user types
def update_count(event=None):
    #event none means the function can be triggered both automatically --> key presses or manually --> calling
    email_text = input_text.get("1.0", tk.END) #start from line 1 to the end
    summary_text = output_text.get("1.0", tk.END).strip()

    original_words = len(email_text.split())
    summary_words = len(summary_text.split())
    count_label.config(text=f"Original Email Words: {original_words} | Summary Words:{summary_words}")

input_text.bind("<KeyRelease>", update_count) # update counts on every key press

# Output
output_label = tk.Label(root, text="Summarized Output:", font=("Helvetica", 13), bg="#f5f5f5")
output_label.pack()
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=10, font=("Helvetica", 11), fg="darkblue")
output_text.pack(pady=(5, 10))

# Summarize function
def on_summarize():
    email_body = input_text.get("1.0", tk.END).strip()  # get email text
    if not email_body:
        messagebox.showwarning("Input Error", "Please enter an email to summarize.")  # if empty, warn
        return
    summary = summarize_email(email_body)  # run summarization
    output_text.delete("1.0", tk.END)  # clear previous output
    output_text.insert(tk.END, summary)  # show new summary
    update_count()
# Clear function
def on_clear():
    input_text.delete("1.0", tk.END)  # clear input text area
    output_text.delete("1.0", tk.END)  # clear output text area
    count_label.config(text="Original Email Words: 0 | Summary Words: 0") # reset counts
    update_count()
# Buttons
summarize_button = tk.Button(
    root,
    text="Summarize Email",
    command=on_summarize,
    bg="#4CAF50",
    fg="white",
    font=("Helvetica", 16, "bold"),
    padx=30,
    pady=15
)
summarize_button.pack(pady=(20, 10))

clear_button = tk.Button(
    root,
    text="Clear",
    command=on_clear,
    bg="#f44336",
    fg="white",
    font=("Helvetica", 12, "bold"),
    padx=20,
    pady=8
)
clear_button.pack(pady=(0, 20))

# Run the GUI
root.mainloop()
