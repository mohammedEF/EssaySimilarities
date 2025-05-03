import lexical_diversity as ld
import numpy as np
from nltk.probability import FreqDist
import pickle

def analyze_lexical_features(preprocessed):
    """
    Analyze lexical features for each time period.
    
    Args:
        preprocessed: Dictionary of preprocessed essays by time period
        
    Returns:
        Dictionary of lexical metrics by time period
    """
    results = {period: {} for period in preprocessed.keys()}
    
    # Track metrics by length buckets to control for length differences
    length_buckets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, float('inf'))]
    buckets_metrics = {period: {str(bucket): {"ttr": [], "mtld": [], "richness": []} 
                                for bucket in length_buckets} 
                        for period in preprocessed.keys()}
    
    for period, essays in preprocessed.items():
        # Combine all essays for corpus-level analysis
        all_words = []
        all_words_no_stop = []
        
        # Essay-level metrics
        ttr_scores = []
        mtld_scores = []
        vocab_richness = []
        essay_lengths = []
        
        for essay in essays:
            words = essay['words']
            words_no_stop = essay['words_no_stop']
            word_count = len(words)
            essay_lengths.append(word_count)
            
            all_words.extend(words)
            all_words_no_stop.extend(words_no_stop)
            
            # Skip very short essays for certain metrics
            if word_count >= 20:  # Lowered threshold to include more essays
                # Type-Token Ratio
                ttr = len(set(words)) / word_count
                ttr_scores.append(ttr)
                
                # Add to appropriate length bucket
                for low, high in length_buckets:
                    if low <= word_count < high:
                        bucket_key = str((low, high))
                        buckets_metrics[period][bucket_key]["ttr"].append(ttr)
                        break
                
                # MTLD (Measure of Textual Lexical Diversity)
                try:
                    if word_count >= 50:  # MTLD needs more tokens for stability
                        mtld = ld.mtld(words)
                        mtld_scores.append(mtld)
                        
                        for low, high in length_buckets:
                            if low <= word_count < high:
                                bucket_key = str((low, high))
                                buckets_metrics[period][bucket_key]["mtld"].append(mtld)
                                break
                except:
                    pass
                
                # Vocabulary richness (unique words / total words)
                if words_no_stop:
                    richness = len(set(words_no_stop)) / len(words_no_stop)
                    vocab_richness.append(richness)
                    
                    for low, high in length_buckets:
                        if low <= word_count < high:
                            bucket_key = str((low, high))
                            buckets_metrics[period][bucket_key]["richness"].append(richness)
                            break
        
        # Save overall results
        results[period]['ttr_mean'] = np.mean(ttr_scores) if ttr_scores else 0
        results[period]['ttr_std'] = np.std(ttr_scores) if ttr_scores else 0
        results[period]['ttr_scores'] = ttr_scores if ttr_scores else 0
        results[period]['mtld_mean'] = np.mean(mtld_scores) if mtld_scores else 0
        results[period]['mtld_std'] = np.std(mtld_scores) if mtld_scores else 0
        results[period]['mtld_scores'] = mtld_scores if mtld_scores else 0
        results[period]['vocab_richness_mean'] = np.mean(vocab_richness) if vocab_richness else 0
        results[period]['vocab_richness_std'] = np.std(vocab_richness) if vocab_richness else 0
        results[period]['vocab_richness'] = vocab_richness if vocab_richness else 0
        results[period]['mean_length'] = np.mean(essay_lengths) if essay_lengths else 0
        results[period]['length_std'] = np.std(essay_lengths) if essay_lengths else 0
        results[period]['essay_lengths'] = essay_lengths if essay_lengths else 0
        
        # Save bucketed results (controlled for length)
        results[period]['length_controlled'] = {}
        for bucket, metrics in buckets_metrics[period].items():
            bucket_results = {}
            for metric_name, values in metrics.items():
                if values:
                    bucket_results[f"{metric_name}_mean"] = np.mean(values)
                    bucket_results[f"{metric_name}_std"] = np.std(values)
                    bucket_results[f"{metric_name}_count"] = len(values)
            results[period]['length_controlled'][bucket] = bucket_results
        
        # Corpus-level frequency analysis
        word_freq = FreqDist(all_words_no_stop)
        results[period]['top_words'] = word_freq.most_common(20)
        
        # Hapax legomena (words that occur only once)
        hapax = [word for word, freq in word_freq.items() if freq == 1]
        results[period]['hapax_percentage'] = len(hapax) / len(word_freq) if word_freq else 0
        
        # Academic/formal word ratio using a simple approach
        academic_words = set([
            'therefore', 'however', 'thus', 'furthermore', 'consequently', 
            'analyze', 'approach', 'concept', 'context', 'data', 'evidence',
            'factor', 'impact', 'method', 'perspective', 'process', 'research',
            'significant', 'specific', 'structure', 'theory', 'variable'
        ])
        academic_word_count = sum(1 for word in all_words_no_stop if word.lower() in academic_words)
        results[period]['academic_word_ratio'] = academic_word_count / len(all_words_no_stop) if all_words_no_stop else 0
        
    return results


with open("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/preprocessed.pkl", "rb") as f:
    preprocessed = pickle.load(f)

results = analyze_lexical_features(preprocessed)
#print(results)

# visualizing results _______________________________________________________________________________________________________________________________________________________
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


sns.set(style="whitegrid")

# 1. TTR and Vocabulary Richness Comparison
plt.figure(figsize=(10, 6))
metrics = ['ttr_mean', 'vocab_richness_mean']
labels = ['TTR', 'Vocabulary Richness']
pre_values = [results['pre_gpt'][m] for m in metrics]
post_values = [results['post_gpt'][m] for m in metrics]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, pre_values, width, label='Pre-GPT')
plt.bar(x + width/2, post_values, width, label='Post-GPT')
plt.xticks(x, labels)
plt.ylabel('Score')
plt.title('Lexical Diversity Comparison')
plt.legend()
plt.show()

# 2. Length-Controlled TTR Visualization
plt.figure(figsize=(10, 6))
length_bins = ['(0,50)', '(50,100)', '(100,150)', '(150,200)', '(200, inf)']
pre_ttr = [results['pre_gpt']['length_controlled'][b]['ttr_mean'] for b in results['pre_gpt']['length_controlled']]
post_ttr = [results['post_gpt']['length_controlled'][b]['ttr_mean'] for b in results['post_gpt']['length_controlled']]

df_ttr = pd.DataFrame({
    'Length Bin': length_bins * 2,
    'TTR': pre_ttr + post_ttr,
    'Group': ['Pre-GPT']*5 + ['Post-GPT']*5
})

sns.barplot(x='Length Bin', y='TTR', hue='Group', data=df_ttr)
plt.title('Length-Controlled TTR Comparison')
plt.xticks(rotation=45)
plt.show()

# 3. Pre-GPT Top Words
plt.figure(figsize=(12, 8))
words, counts = zip(*results['pre_gpt']['top_words'])
sns.barplot(x=counts, y=words, palette='viridis')
plt.title('Pre-GPT Top Words')
plt.xlabel('Frequency')
plt.show()

# 4. Post-GPT Top Words
plt.figure(figsize=(12, 8))
words, counts = zip(*results['post_gpt']['top_words'])
sns.barplot(x=counts, y=words, palette='viridis')
plt.title('Post-GPT Top Words')
plt.xlabel('Frequency')
plt.show()

# 5. Hapax Legomena and Academic Word Ratio
plt.figure(figsize=(10, 6))
hapax = [results['pre_gpt']['hapax_percentage'], results['post_gpt']['hapax_percentage']]
academic = [results['pre_gpt']['academic_word_ratio'], results['post_gpt']['academic_word_ratio']]

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, hapax, width, label='Hapax %')
plt.bar(x + width/2, academic, width, label='Academic Words %')
plt.xticks(x, ['Pre-GPT', 'Post-GPT'])
plt.ylabel('Percentage')
plt.title('Unique Words vs Academic Vocabulary')
plt.legend()
plt.show()

# 6. Text Length Distribution
plt.figure(figsize=(10, 6))
length_data = [
    results['pre_gpt']['mean_length'],
    results['post_gpt']['mean_length']
]
std_data = [
    results['pre_gpt']['length_std'],
    results['post_gpt']['length_std']
]

plt.bar(['Pre-GPT', 'Post-GPT'], length_data, yerr=std_data, 
        capsize=10, alpha=0.7, color=['blue', 'orange'])
plt.ylabel('Mean Length (chars)')
plt.title('Text Length Comparison with Std Dev')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats

# Extract data from results
pre_gpt = results['pre_gpt']
post_gpt = results['post_gpt']

# 1. TTR Test (Welch's t-test for unequal variances)
ttr_pre = pre_gpt['ttr_scores']
ttr_post = post_gpt['ttr_scores']


# normality check before deciding the statistical test ___________________________________________________________________________________________________

from statsmodels.graphics.gofplots import qqplot
def check_normality(data, name):
    """Check normality with statistical tests and QQ-plot"""
    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ks_stat, ks_p = stats.kstest(data, 'norm')
    
    # QQ-plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    qqplot(data, line='s', ax=plt.gca())
    plt.title(f'QQ-plot for {name}')
    
    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=30, density=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Distribution of {name}')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'shapiro': (shapiro_stat, shapiro_p),
        'ks': (ks_stat, ks_p)
    }

# Check normality for pre-GPT
print("=== Pre-GPT Normality Check ===")
pre_norm = check_normality(np.array(ttr_pre), "Pre-GPT TTR")

# Check normality for post-GPT
print("\n=== Post-GPT Normality Check ===")
post_norm = check_normality(np.array(ttr_post), "Post-GPT TTR")

# Interpret results
alpha = 0.05
print(f"\nNormality Test Results (α = {alpha}):")
print("Pre-GPT:")
print(f"  Shapiro-Wilk: p = {pre_norm['shapiro'][1]:.3e} {'(Normal)' if pre_norm['shapiro'][1] > alpha else '(Non-normal)'}")
print(f"  Kolmogorov-Smirnov: p = {pre_norm['ks'][1]:.3e} {'(Normal)' if pre_norm['ks'][1] > alpha else '(Non-normal)'}")

print("\nPost-GPT:")
print(f"  Shapiro-Wilk: p = {post_norm['shapiro'][1]:.3e} {'(Normal)' if post_norm['shapiro'][1] > alpha else '(Non-normal)'}")
print(f"  Kolmogorov-Smirnov: p = {post_norm['ks'][1]:.3e} {'(Normal)' if post_norm['ks'][1] > alpha else '(Non-normal)'}")

''' we will go with non-parametric Mann-Whitney U test as the distribution shows non-normality '''

u_stat, p_mannwhitney = stats.mannwhitneyu(ttr_pre, ttr_post)
print(f"Mann-Whitney U: U = {u_stat:.0f}, p = {p_mannwhitney}")
#______________________________________________________________________________________________________________________________________________________


# 2. Academic Word Usage (Two-proportion z-test)
total_words_pre = sum(pre_gpt['essay_lengths'])
total_words_post = sum(post_gpt['essay_lengths'])

academic_words_pre = pre_gpt['academic_word_ratio'] * total_words_pre
academic_words_post = post_gpt['academic_word_ratio'] * total_words_post

z_academic, p_academic = proportions_ztest(
    [academic_words_pre, academic_words_post],
    [total_words_pre, total_words_post],
    alternative='two-sided'
)

# 3. Hapax Legomena (Two-proportion z-test)
hapax_pre = pre_gpt['hapax_percentage'] * total_words_pre
hapax_post = post_gpt['hapax_percentage'] * total_words_post

z_hapax, p_hapax = proportions_ztest(
    [hapax_pre, hapax_post],
    [total_words_pre, total_words_post],
    alternative='two-sided'
)


# Print results
print(f"""
Statistical Significance Results:
---------------------------------

2. Academic Word Usage:
   z = {z_academic:.2f}, p = {p_academic} {'***' if p_academic < 0.001 else ''}

3. Hapax Legomena:
   z = {z_hapax:.2f}, p = {p_hapax} {'***' if p_hapax < 0.001 else ''}

""")

