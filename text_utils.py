import re

def clean_text(text: str) -> str: # i/p: str, o/p: str
    """
    Cleans and normalizes input text for BERT model.

    Args:
        text (str): Input string (e.g., candidate or job skills)

    Returns:
        str: Cleaned text.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation/symbols
    text = re.sub(r'\s+', ' ', text)     # normalize multiple spaces
    return text
