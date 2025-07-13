import tkinter as tk
from tkinter import scrolledtext, messagebox

from summarization import (
    split_sentences,
    fit_vectorizer,
    is_valid_sentence,
    summarize_email,
    summarize_abstractive
)

def run_email_summarizer_gui():
    root = tk.Tk()
    root.title("Email Summarizer")
    root.geometry("800x720")
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
    input_text.pack(pady=(5, 10))

    # Word count label
    count_label = tk.Label(root, text="Words: 0 | Characters: 0", font=("Helvetica", 10), bg="#f5f5f5", fg="gray")
    count_label.pack()

    def update_count(event=None):
        content = input_text.get("1.0", tk.END)
        count_label.config(text=f"Words: {len(content.split())} | Characters: {len(content.strip())}")

    input_text.bind("<KeyRelease>", update_count)

    # Output
    output_label = tk.Label(root, text="Summarized Output:", font=("Helvetica", 13), bg="#f5f5f5")
    output_label.pack()
    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=10, font=("Helvetica", 11), fg="darkblue")
    output_text.pack(pady=(5, 10))

    def on_summarize():
        email_body = input_text.get("1.0", tk.END).strip()
        if not email_body:
            messagebox.showwarning("Input Error", "Please enter an email to summarize.")
            return
        summary = summarize_email(email_body)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, summary)

    def on_clear():
        input_text.delete("1.0", tk.END)
        output_text.delete("1.0", tk.END)
        count_label.config(text="Words: 0 | Characters: 0")

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


# # Optional: allow standalone script usage
# if __name__ == "__main__":
#     run_email_summarizer_gui()
