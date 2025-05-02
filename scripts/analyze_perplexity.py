import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import json

def analyze_perplexity(df):
    """
    Analyze GPT-2 perplexity scores for different time periods.

    Args:
        df: DataFrame with 'time_period' and 'essay_text'

    Returns:
        Dictionary with perplexity statistics
    """
    print("Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    results = {}
    for period in df['time_period'].unique():
        texts = df[df['time_period'] == period]['essay_text'].tolist()
        perplexities = []

        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encodings.input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
                perplexities.append(torch.exp(neg_log_likelihood).item())

        results[period] = {
            'mean_perplexity': np.mean(perplexities) if perplexities else 0,
            'std_perplexity': np.std(perplexities) if perplexities else 0,
            'perplexity_values': perplexities
        }

    return results



df = pd.read_csv("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/combined_processed.csv")
perplexity_analysis = analyze_perplexity(df)


pre_perplexity_values = [round(x, 3) for x in perplexity_analysis['pre_gpt']['perplexity_values']]
post_perplexity_values = [round(x, 3) for x in perplexity_analysis['post_gpt']['perplexity_values']]

perplexity_values = {'pre_perplexity_values': pre_perplexity_values, 'post_perplexity_values': post_perplexity_values}

with open("C:/Users/PRECISION 5550/Desktop/Essays Project/new/Perplexity Analysis/perplexity_values.json", "w") as f:
    json.dump(perplexity_values, f)