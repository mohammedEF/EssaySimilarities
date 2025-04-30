
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_essays(input_file, pre_or_post):
    """Process raw essays from Excel file and return cleaned DataFrame"""
    # Load the Excel file
    df = pd.read_excel(input_file)
    
    # Filter out essays with less than 15 words
    df = df[df['essay_text'].apply(lambda x: len(str(x).split()) >= 15)]
    
    # Initialize preprocessing tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Apply basic cleaning
    df['essay_text'] = df['essay_text'].str.strip()
    df = df.dropna(subset=['essay_text'])
    
    # Create time_period column
    df['time_period'] = pre_or_post
    
    return df[['essay_text', 'time_period']]

def analyze_data(df):
    """Analyze processed DataFrame and add linguistic features"""
    # Validate required columns
    required_cols = ['essay_text', 'time_period']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Add word count feature
    df['word_count'] = df['essay_text'].apply(lambda x: len(word_tokenize(x)))
    
    # Print dataset statistics
    print(f"Analyzed dataset with {len(df)} essays")
    print(f"Pre-GPT essays: {len(df[df['time_period'] == 'pre_gpt'])}")
    print(f"Post-GPT essays: {len(df[df['time_period'] == 'post_gpt'])}")
    print("\nDataset statistics:")
    print(df.describe())
    '''
    # Plot word count distribution
    sns.histplot(df['word_count'])
    plt.title('Word Count Distribution of Essays')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()
    '''
    return df

def process_essays(input_excel_path):
    """Main function to process and analyze essays"""
    # Preprocess the data
    processed_df = preprocess_essays(input_excel_path, pre_or_post='post_gpt')
    
    # Analyze the data
    analyzed_df = analyze_data(processed_df)
    return analyzed_df

# Example usage
if __name__ == "__main__":
    input_path = "C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/ESP 2024 - CLEAN.xlsx"
    ESP_2024_df = process_essays(input_path)

ESP_2024_df.to_csv('C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/post_gpt_processed.csv')
