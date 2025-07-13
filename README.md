# Project Overview
This project focuses on processing and analyzing email data, specifically from the Enron email dataset. The main objectives are to clean the data, extract relevant features, and summarize the content of the emails for further analysis.

## Project Structure
- **main1.ipynb**: Contains the main code for processing and analyzing email data. It includes functionalities for data loading, cleaning, and summarization.
- **feature.py**: Implements feature extraction functionalities, including functions for extracting relevant features such as TF-IDF scores and keywords.
- **preprocessing.py**: Contains functions for preprocessing the email data, including text cleaning and extraction of email fields.
- **main_enron_emails.csv**: The dataset containing the Enron email data used for analysis and feature extraction.

## Setup Instructions
1. Ensure you have Python installed on your machine.
2. Install the required libraries:
   - pandas
   - numpy
   - nltk
   - transformers
   - scikit-learn
   - rouge-score
3. Download the necessary NLTK resources by running the following commands in your Python environment:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage Guidelines
- Open `main1.ipynb` to start processing the email data.
- Use `feature.py` to implement and test feature extraction functions.
- Modify `preprocessing.py` as needed to enhance data cleaning and preparation steps.
- Refer to the dataset `main_enron_emails.csv` for the email data used in this project.

## Contributing
Contributions to improve the functionality and performance of this project are welcome. Please submit a pull request or open an issue for discussion.