import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

#Download necessary NLTK data files
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



"""****************************************************************     PREPROCESSING    *****************************************************"""




def preprocess_essay(essay, max_length=200):

    # 1. Lowercase the text
    essay = essay.lower()

    # 2. Remove punctuation
    essay = re.sub(f"[{re.escape(string.punctuation)}]", "", essay)

    # 3. Tokenize into words
    tokens = word_tokenize(essay)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Length normalization (truncate or pad with empty strings)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens.extend(["" for _ in range(max_length - len(tokens))])

    # Return the preprocessed essay as a single string
    return " ".join(tokens)

def preprocess_essays(pre_essays, max_length=200):

    return [preprocess_essay(essay, max_length=max_length) for essay in pre_essays]

# processing pre_GPT essays
pre_essays = ['I wanted to join the Machine learning club of my newly joined college, but lock down prevented that, still I got in touch with the club head to introduce myself and the future possibility to join the club. We got to know each other and he referred me to "Code in Place" a global teaching initiative by Stanford University to teach people Python programming. Though I already knew how to code in python( I\'m self-taught using multiple online resources), I thought this was a fabulous opportunity to make some connections. Code in Place is where I found the link to Global Summer Institute, and since Code in Place ends this week, I thought this might be another opportunity I can pursue. I am a very friendly person and like making connections, both professional and private. I am interested in the stock market and do some forex trading every now and then to have some extra money at the end of the month. Reading books is my preferred way of spending leisure time. Some of my favourite authors are Agatha Christie, Dan Brown, Gillian Flynn, and R.L Stine. But nowadays I\'m trying to read more financial books like Rich dad Poor dad, Think and grow rich, etc. Sometimes they get a little dull since the reason I liked reading was to have an escape from reality, but I figured these things are important too for my adult life. The current book I\'m reading is Elon Musk by Ashlee Vance. He\'s my favourite billionaire and reading about his life inspires me.',
 "Like I said previously, I love making connections. I plan to do some really daring stuff in the future and one thing that all of my idols have in common is that they were able to do what they did, by having a team of like-minded and highly persistent people. After reading the brochure of the program, there's no doubt that I'll be learning advanced stuff if I get accepted, but what caught my eye is the diverse faculty, having classmates from all around the world, and especially the career guidance system. Data science is a relatively new field so there's a lack of good mentorship, at least from where I come. Having a chance to socialize and study under such accomplished faculty from prestigious universities, can make all the difference in where I end up.\nI am from India and currently pursuing an undergraduate degree in Data Science from a community college. Unfortunately, money was a factor for me, so I couldn't afford a fancier college, and I hate to admit it, but I was not driven, as much as I am now, back when I was in high school. There are just 4 students in my course, and we don't even have a professor who has some experience as a Data Scientist or something related to that. So networking is my main focus for now.\n",
 "Though it might sound far-fetched, my lifelong dream is to somehow contribute to the field of physics and computing. I'm learning data science to apply it in fields like particle physics and astronomy. I know this might not sound much feasible but I'm confident that if I make the right decisions and work hard, I can achieve it. The problem that I think is with society nowadays is that people are getting into the sciences for all the wrong reasons. Most of the talented people who can do so much for society end up making apps and websites to make money. Like I said I am a fan of Elon Musk, not because he is a multi-billionaire, but because of his ideals. All of his companies are aimed at the continuation of the human species. The two problems that I think should be addressed immediately are finding more ways of efficiently using renewable energy and the increasing antibiotic resistance of bacteria. Both of these problems are a matter of ‘when’ not ‘If’. The negative effects of non-renewable energy sources are taught in the 6th grade. Every kid knows that burning coal, and driving petrol cars is taxing the environment, but still, there is not much active research in solving this problem or finding some sustainable solutions to it. About the increasing resistance in bacteria problem, I didn’t have the slightest idea about this, till I heard Bill Gates talk about it in one of his Youtube videos. I looked into it and found countless articles and videos online, and was shocked to learn the truth. The increasing use of antibiotics in treating human and livestock diseases is slowly but surely making bacteria immune to it. That’s why newer and more powerful drugs have to be developed every year to battle this crisis, but there is a limit to what humans can create. The time might come soon when bacteria are resistant to all human drugs. This genuinely scares me.",
 'One might think that I am more interested in Physics than in Data Science. Well, it’s because yes I am. I had this enthusiasm back when I completed high school in 2018. So there I was an 18-year-old who just finished high school with good grades, having a dream, and all the confidence in the world. I was so determined and naive back then. I took admission in a nearby public college which had a good Physics department. I still remember I was so excited on my first day, a world full of possibilities, the beginning of my dream career. I’d say reality started to set in on Day 2. For the first time in my life, I was started to get bored of Physics, the lectures were all full of mathematical formulas, no more interesting facts about stars but it was all about forming the equation of motion of a ball rolling downhill. I dropped out of college in 3 months. I think I lacked persistence. I was afraid to do the hard work and spent most of my days playing games. After dropping out, I planned to join the navy, since my father was in that trade and it paid handsomely. I couldn’t crack the exam the first 2 times and it wasted another year of my life. The 3rd time I was so well prepared, I was confident that no one could stop me, but I was wrong. Apparently, a global pandemic can. The entrance of 2020 was cancelled. This time I made sure that I won’t be a coward and face my problems head-on. I scoured the internet for possible career options at that time. It was almost the end of the year and most of the other entrances had already happened. It was during this time that I had rekindled my passion for astronomy and found that Machine Learning is an up-and-coming tool in that domain. And I was lucky enough that a college where my friend studied had newly opened a Data Science department. Fast-forwarding to now, I am currently my department’s topper and learning more and more about Machine Learning every day and I’m more motivated than ever.',
 'Before India experienced the second wave of the pandemic, I got myself enrolled with a not-for-profit organization called the Environmentalist Foundation of India in Chennai. It conducted weekly rallies for cleaning up and beautifying water bodies. I had gone to a few of them and helped rid the ponds and beaches of their plastic waste, painted murals on the walls, planted new saplings and installed a watering system. This was important to me as one of my fundamental beliefs is to save the environment from the brutal force that we exert on it. I always wanted to give back something from which we take a lot from and these rallies expanded my view on global crisis in general. It showed me that environmental degradation is much more serious than is known. I understood that the smallest of acts create a huge impact in the long run and that every step counts.\n\nIn addition to environmental sustenance, a personal passion of mine is dancing. During the lockdown, I created my public dance page on Instagram and worked on releasing my choreographies. This meant a lot to me as dance is my form of expression and emotional release of everything. It was one of the things I was scared to do and this leap of faith helped me build not only self-confidence but happiness through a physical movement that tends to get restricted during a lockdown. I plan to start dance and fitness classes soon and share happiness while donating 75 percent of the proceedings to an orphanage.\n\nI have always wanted to be financially independent and one step that I took towards this goal is by getting to know more about stock markets and making a few investments. Not only did this help me make my own decisions, but also taught me about stability, patience, and educational guesses.\xa0\n']

post_essays = [
    "During my final year of university, I was tasked with leading a group project for a business analytics course. Despite not having previous leadership experience, I volunteered to lead the group as others were hesitant. The task was to analyze a large dataset, and my group members were overwhelmed by the complexity. I spent additional hours learning advanced analytical techniques and programming languages to guide the team effectively. I held extra meetings, communicated regularly, and divided the workload fairly. This experience pushed me beyond my comfort zone but resulted in a successful project and a deeper understanding of leadership.",
    "I had always been comfortable with my routine tasks at work, but when our department faced an unexpected challenge to meet a critical deadline, I decided to step up. I took on extra responsibilities, stayed late, and collaborated with other teams to troubleshoot technical issues. It was the first time I managed a multi-team effort, which required learning new software tools and facilitating communication between departments. Though it was stressful and unfamiliar, I pushed through, and we completed the project two days early. This experience taught me the importance of adaptability and teamwork when facing tough deadlines.",
    "When I was working on a client’s project that involved integrating multiple databases, the scope of the task expanded unexpectedly, requiring me to learn a new software system within a week. This was far beyond my usual skill set, but I committed to the challenge. I spent long hours after work studying, attending webinars, and experimenting with the new system. By the end of the week, not only had I successfully integrated the databases, but I also streamlined the process for future use. It was a taxing experience but one that immensely developed my problem-solving skills.",
    "Last year, I decided to participate in a charity marathon even though I had never run more than 5 kilometers before. It was a daunting 21-kilometer race, and I had just six weeks to train. Pushing myself well beyond my physical comfort zone, I dedicated myself to a strict training schedule. I woke up early every morning, adjusted my diet, and ran progressively longer distances. By race day, I felt prepared and successfully completed the marathon, raising a significant amount for charity. This experience taught me the power of discipline and determination in achieving goals that initially seem out of reach.",
    "At my previous job, I was part of a marketing team tasked with increasing online engagement for a new product launch. One of our strategies wasn’t yielding the desired results, so I proposed a social media campaign that targeted a different audience segment. Although I had limited experience in this area, I spent late nights researching best practices and learning new digital marketing tools. My initiative resulted in a 25% increase in engagement, exceeding our target by 10%. This experience taught me the value of stepping out of my comfort zone and trusting my instincts when tackling new challenges."
]

preprocessed_preGPT_essays = preprocess_essays(pre_essays, max_length=100)
for i, essay in enumerate(preprocessed_preGPT_essays):
    print(f"Preprocessed Essay {i+1}:\n{essay}\n")



preprocessed_postGPT_essays = preprocess_essays(post_essays, max_length=100)
for i, essay in enumerate(preprocessed_postGPT_essays):
    print(f"Preprocessed Essay {i+1}:\n{essay}\n")


"""****************************************************************     Quantifying Changes in Writing (TF-IDF + Transformers)    *****************************************************"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def compute_similarity_matrix(essays, method='tfidf'):

    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(essays)
        similarity_matrix = cosine_similarity(tfidf_matrix)
    elif method == 'transformer':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(essays)
        similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def quantify_style_changes(preprocessed_preGPT_essays, preprocessed_postGPT_essays):

    # Compute intra-group similarity for pre-GPT essays
    intra_pre_tfidf = compute_similarity_matrix(preprocessed_preGPT_essays, method='tfidf')
    intra_pre_transformer = compute_similarity_matrix(preprocessed_preGPT_essays, method='transformer')

    # Compute intra-group similarity for post-GPT essays
    intra_post_tfidf = compute_similarity_matrix(preprocessed_postGPT_essays, method='tfidf')
    intra_post_transformer = compute_similarity_matrix(preprocessed_postGPT_essays, method='transformer')

    # Combine essays for inter-group similarity
    combined_essays = preprocessed_preGPT_essays + preprocessed_postGPT_essays
    inter_tfidf = compute_similarity_matrix(combined_essays, method='tfidf')
    inter_transformer = compute_similarity_matrix(combined_essays, method='transformer')

    # Extract inter-group similarities
    n_pre = len(preprocessed_preGPT_essays)
    n_post = len(preprocessed_postGPT_essays)
    inter_tfidf_vals = inter_tfidf[:n_pre, n_pre:].flatten()
    inter_transformer_vals = inter_transformer[:n_pre, n_pre:].flatten()

    results = {
        'intra_pre': {
            'tfidf': np.mean(intra_pre_tfidf[np.triu_indices_from(intra_pre_tfidf, k=1)]),
            'transformer': np.mean(intra_pre_transformer[np.triu_indices_from(intra_pre_transformer, k=1)])
        },
        'intra_post': {
            'tfidf': np.mean(intra_post_tfidf[np.triu_indices_from(intra_post_tfidf, k=1)]),
            'transformer': np.mean(intra_post_transformer[np.triu_indices_from(intra_post_transformer, k=1)])
        },
        'inter_group': {
            'tfidf': np.mean(inter_tfidf_vals),
            'transformer': np.mean(inter_transformer_vals)
        }
    }

    return results, intra_pre_tfidf, intra_post_tfidf, inter_tfidf_vals, intra_pre_transformer, intra_post_transformer, inter_transformer_vals


results = quantify_style_changes(preprocessed_preGPT_essays, preprocessed_postGPT_essays)
print("Quantified Style Changes:", results)

"""*************************************************************   VISUALIZATION   **************************************************************"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot heatmaps
def plot_heatmap(matrix, title, ax):
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Essay Index")
    ax.set_ylabel("Essay Index")

# Function to plot bar plots for average similarities
def plot_bar(results, ax):
    labels = ['Pre-GPT (TF-IDF)', 'Post-GPT (TF-IDF)', 'Inter-Group (TF-IDF)',
              'Pre-GPT (Transformer)', 'Post-GPT (Transformer)', 'Inter-Group (Transformer)']
    values = [results['intra_pre']['tfidf'], results['intra_post']['tfidf'], results['inter_group']['tfidf'],
              results['intra_pre']['transformer'], results['intra_post']['transformer'], results['inter_group']['transformer']]
    
    ax.bar(labels, values, color=['blue', 'orange', 'green', 'blue', 'orange', 'green'])
    ax.set_ylabel("Average Similarity")
    ax.set_title("Average Similarity Scores")
    ax.tick_params(axis='x', rotation=45)

# Extract the results and similarity matrices from the results variable
results_dict, intra_pre_tfidf, intra_post_tfidf, inter_tfidf_vals, intra_pre_transformer, intra_post_transformer, inter_transformer_vals = results

# Create subplots for heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot heatmaps
plot_heatmap(intra_pre_tfidf, "Pre-GPT Essays (TF-IDF Similarity)", axes[0, 0])
plot_heatmap(intra_post_tfidf, "Post-GPT Essays (TF-IDF Similarity)", axes[0, 1])
plot_heatmap(intra_pre_transformer, "Pre-GPT Essays (Transformer Similarity)", axes[1, 0])
plot_heatmap(intra_post_transformer, "Post-GPT Essays (Transformer Similarity)", axes[1, 1])

plt.tight_layout()
plt.show()

# Create bar plot for average similarities
fig, ax = plt.subplots(figsize=(10, 6))
plot_bar(results_dict, ax)
plt.tight_layout()
plt.show()


"""****************************************************************     Statistical Validation    *****************************************************"""

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Extract similarity matrices and values from the results variable
results_dict, intra_pre_tfidf, intra_post_tfidf, inter_tfidf_vals, intra_pre_transformer, intra_post_transformer, inter_transformer_vals = results

# Pre-GPT TF-IDF similarities (flattened upper triangle of the matrix, excluding diagonal)
pre_tfidf_similarities = intra_pre_tfidf[np.triu_indices_from(intra_pre_tfidf, k=1)]

# Post-GPT TF-IDF similarities (flattened upper triangle of the matrix, excluding diagonal)
post_tfidf_similarities = intra_post_tfidf[np.triu_indices_from(intra_post_tfidf, k=1)]

# Pre-GPT Transformer similarities (flattened upper triangle of the matrix, excluding diagonal)
pre_transformer_similarities = intra_pre_transformer[np.triu_indices_from(intra_pre_transformer, k=1)]

# Post-GPT Transformer similarities (flattened upper triangle of the matrix, excluding diagonal)
post_transformer_similarities = intra_post_transformer[np.triu_indices_from(intra_post_transformer, k=1)]

# Inter-group similarities (TF-IDF and Transformer)
inter_tfidf_similarities = inter_tfidf_vals
inter_transformer_similarities = inter_transformer_vals

# Function to perform statistical tests
def perform_statistical_tests(pre_vals, post_vals):
    # Independent t-test
    t_stat, t_pvalue = ttest_ind(pre_vals, post_vals, equal_var=False)
    
    # Mann-Whitney U test
    u_stat, u_pvalue = mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
    
    return {
        't_test': {'statistic': t_stat, 'p_value': t_pvalue},
        'mann_whitney_u': {'statistic': u_stat, 'p_value': u_pvalue}
    }

# Perform tests for TF-IDF similarities (Pre-GPT vs Post-GPT)
tfidf_results = perform_statistical_tests(pre_tfidf_similarities, post_tfidf_similarities)

# Perform tests for Transformer similarities (Pre-GPT vs Post-GPT)
transformer_results = perform_statistical_tests(pre_transformer_similarities, post_transformer_similarities)

# Perform tests for inter-group similarities (Pre-GPT vs Inter-Group)
inter_results_tfidf = perform_statistical_tests(pre_tfidf_similarities, inter_tfidf_similarities)
inter_results_transformer = perform_statistical_tests(pre_transformer_similarities, inter_transformer_similarities)

# Print results
print("TF-IDF Similarities Statistical Tests (Pre-GPT vs Post-GPT):")
print(f"T-test: Statistic = {tfidf_results['t_test']['statistic']}, p-value = {tfidf_results['t_test']['p_value']}")
print(f"Mann-Whitney U: Statistic = {tfidf_results['mann_whitney_u']['statistic']}, p-value = {tfidf_results['mann_whitney_u']['p_value']}")

print("\nTransformer Similarities Statistical Tests (Pre-GPT vs Post-GPT):")
print(f"T-test: Statistic = {transformer_results['t_test']['statistic']}, p-value = {transformer_results['t_test']['p_value']}")
print(f"Mann-Whitney U: Statistic = {transformer_results['mann_whitney_u']['statistic']}, p-value = {transformer_results['mann_whitney_u']['p_value']}")

print("\nInter-Group TF-IDF Similarities Statistical Tests (Pre-GPT vs Inter-Group):")
print(f"T-test: Statistic = {inter_results_tfidf['t_test']['statistic']}, p-value = {inter_results_tfidf['t_test']['p_value']}")
print(f"Mann-Whitney U: Statistic = {inter_results_tfidf['mann_whitney_u']['statistic']}, p-value = {inter_results_tfidf['mann_whitney_u']['p_value']}")

print("\nInter-Group Transformer Similarities Statistical Tests (Pre-GPT vs Inter-Group):")
print(f"T-test: Statistic = {inter_results_transformer['t_test']['statistic']}, p-value = {inter_results_transformer['t_test']['p_value']}")
print(f"Mann-Whitney U: Statistic = {inter_results_transformer['mann_whitney_u']['statistic']}, p-value = {inter_results_transformer['mann_whitney_u']['p_value']}")