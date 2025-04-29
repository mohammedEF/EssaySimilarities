import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from nltk.corpus import stopwords
import spacy
import pandas as pd
# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def preprocess_text(text: str):
    """
    Preprocess a single essay text into various useful formats.
    
    Args:
        text: Raw essay text
        
    Returns:
        Dictionary with various preprocessed versions of the text
    """
    # Basic tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words_no_stop = [w for w in words if w.isalnum() and w not in stopwords.words('english')]
    
    # N-grams
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # spaCy processing
    doc = nlp(text)

    essay_analysis = {
        'raw': text,
        'sentences': sentences,
        'words': words,
        'words_no_stop': words_no_stop,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'pos_tags': pos_tags,
        'spacy_doc': doc
    }

    return essay_analysis

def preprocess_dataset(df):
        """
        Preprocess all essays in the dataset and split by time period.
        
        Returns:
            Dictionary with preprocessed essays split by time period
        """
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        preprocessed = {
            'pre_gpt': [],
            'post_gpt': []
        }
        
        for _, row in df.iterrows():
            processed = preprocess_text(row['essay_text'])
            preprocessed[row['time_period']].append(processed)
            
        print(f"Preprocessed {len(preprocessed['pre_gpt'])} pre-GPT essays and {len(preprocessed['post_gpt'])} post-GPT essays")
        
        return preprocessed



df = pd.read_csv('C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/combined_processed.csv')
preprocess_dataset(df)