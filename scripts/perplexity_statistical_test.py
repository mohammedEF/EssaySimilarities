import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats

with open('C:/Users/PRECISION 5550/Desktop/Essays Project/new/Perplexity Analysis/perplexity_values.json', 'r') as f:
    perplexity_values = json.load(f)

pre_perplexity_vlaues = perplexity_values['pre_perplexity_values']
post_perplexity_values = perplexity_values['post_perplexity_values']



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
pre_norm = check_normality(np.array(pre_perplexity_vlaues), "pre_perplexity_values")

# Check normality for post-GPT
print("\n=== Post-GPT Normality Check ===")
post_norm = check_normality(np.array(post_perplexity_values), "post_perplexity_values")

# Interpret results
alpha = 0.05
print(f"\nNormality Test Results (α = {alpha}):")
print("Pre-GPT:")
print(f"  Shapiro-Wilk: p = {pre_norm['shapiro'][1]:.3e} {'(Normal)' if pre_norm['shapiro'][1] > alpha else '(Non-normal)'}")
print(f"  Kolmogorov-Smirnov: p = {pre_norm['ks'][1]:.3e} {'(Normal)' if pre_norm['ks'][1] > alpha else '(Non-normal)'}")

print("\nPost-GPT:")
print(f"  Shapiro-Wilk: p = {post_norm['shapiro'][1]:.3e} {'(Normal)' if post_norm['shapiro'][1] > alpha else '(Non-normal)'}")
print(f"  Kolmogorov-Smirnov: p = {post_norm['ks'][1]:.3e} {'(Normal)' if post_norm['ks'][1] > alpha else '(Non-normal)'}")


sns.histplot(pre_perplexity_vlaues)
plt.show()

sns.histplot(post_perplexity_values)
plt.show()


import numpy as np
from scipy import stats

# Perform Welch’s t-test (does not assume equal variances)
t_stat, p_value_t = stats.ttest_ind(pre_perplexity_vlaues, post_perplexity_values, equal_var=False)

# Perform Mann-Whitney U test (non-parametric)
u_stat, p_value_u = stats.mannwhitneyu(pre_perplexity_vlaues, post_perplexity_values, alternative='two-sided')

# Interpret results
alpha = 0.05
print(f"Welch’s t-test: p-value = {p_value_t}")
print(f"Mann-Whitney U test: p-value = {p_value_u}\n")
