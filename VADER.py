import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from business_related_words import business_related_words
# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load NLTK stop words
stop_words = set(stopwords.words('english'))

# Load reviews from CSV
data = pd.read_csv('reviewsnew.csv')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to extract sentiment-bearing adjectives
def extract_sentiment_adjectives(review):
    words = word_tokenize(review.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    
    
    # Perform part-of-speech tagging
    pos_tags = pos_tag(filtered_words)

    
    sentiment_adjectives = []
    for word, pos in pos_tags:
        if pos == 'JJ' and abs(sia.polarity_scores(word)['compound']) > 0.2:
            sentiment_adjectives.append(word)
            
              
    return sentiment_adjectives



# Extract sentiment-bearing adjectives from all reviews
all_sentiment_adjectives = [adjective for review in data['Review'] for adjective in extract_sentiment_adjectives(review)]

# Count the occurrences of each sentiment adjective
sentiment_adjective_counter = Counter(all_sentiment_adjectives)

# Plot the most common sentiment adjectives
num_most_common_adjectives = 10
most_common_sentiment_adjectives = sentiment_adjective_counter.most_common(num_most_common_adjectives)

adjective_labels, adjective_counts = zip(*most_common_sentiment_adjectives)

plt.bar(adjective_labels, adjective_counts)
plt.title(f"Top {num_most_common_adjectives} Most Common Sentiments ")
plt.xlabel("Sentiments")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('most_common_adjectives.png')
plt.show()




# Get the top negative adjectives with compound score < 0.2
top_negative_adjectives = [(adjective, score) for adjective, score in sentiment_adjective_counter.items() if score >= 10 and sia.polarity_scores(adjective)['compound'] < -0.2]

# Sort the negative adjectives based on their count
top_negative_adjectives.sort(key=lambda x: x[1], reverse=True)

# Select the top 10 negative adjectives
top_negative_adjectives = top_negative_adjectives[:10]


# Filter adjectives with negative polarity
negative_adjectives = [adjective for adjective, _ in top_negative_adjectives]

# Get the top 10 most common positive adjectives
# Get the top positive adjectives with compound score > 0.2
top_positive_adjectives = [(adjective, score) for adjective, score in sentiment_adjective_counter.items() if score >= 10 and sia.polarity_scores(adjective)['compound'] > 0.2]

# Sort the positive adjectives based on their count
top_positive_adjectives.sort(key=lambda x: x[1], reverse=True)

# Select the top 10 positive adjectives
top_positive_adjectives = top_positive_adjectives[:10]

# Extract the adjective words from the sorted list
positive_adjectives = [adjective for adjective, _ in top_positive_adjectives]

# Filter adjectives with positive polarity
positive_adjectives = [adjective for adjective, _ in top_positive_adjectives]

# Plot the most common negative and positive adjectives
plt.figure(figsize=(12, 6))
plt.bar(negative_adjectives, [sentiment_adjective_counter[adj] for adj in negative_adjectives], color='red', alpha=0.6, label='Negative')
plt.bar(positive_adjectives, [sentiment_adjective_counter[adj] for adj in positive_adjectives], color='green', alpha=0.6, label='Positive')
plt.title("Top 10 Most Common Negative and Positive Sentiments")
plt.xlabel("Sentiments")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('negative_positive_adjectives.png')
plt.show()




#This code is for The top 15 Occurence of Business-Related Word in Positive and Negative Reviews 


# Separate positive and negative reviews
positive_reviews = []
negative_reviews = []

for review in data['Review']:
    sentiment_score = sia.polarity_scores(review)['compound']
    if sentiment_score > 0.2:
        positive_reviews.append(review)
        
    elif sentiment_score < -0.2:
        negative_reviews.append(review)
       


def count_business_related_words(review, business_words):
    words = word_tokenize(review.lower())
    filtered_words = [word for word in words if word in business_words]
    
    return Counter(filtered_words)

# Define business-related words


# Count occurrences of business-related words in positive and negative reviews
positive_business_word_counts = Counter()
negative_business_word_counts = Counter()



for review in positive_reviews:
    positive_business_word_counts += count_business_related_words(review, business_related_words)

for review in negative_reviews:
    negative_business_word_counts += count_business_related_words(review, business_related_words)



# Get the top 15 most common business-related words in positive and negative reviews
num_top_words = 15
top_positive_business_words = positive_business_word_counts.most_common(num_top_words)
top_negative_business_words = negative_business_word_counts.most_common(num_top_words)

# Extract the words and their counts from the tuples
top_positive_words, positive_counts = zip(*top_positive_business_words)
top_negative_words, negative_counts = zip(*top_negative_business_words)

# Create a grouped bar plot for top business-related words in positive and negative reviews
width = 0.4  # Width of the bars
ind = np.arange(num_top_words)  # X locations for the groups

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, positive_counts, width, label='Positive', color='green')
rects2 = ax.bar(ind + width/2, negative_counts, width, label='Negative', color='red')

ax.set_title(f'Top {num_top_words} Occurrences of Business-Related Words in Positive and Negative Reviews')
ax.set_xlabel('Business-Related Words')
ax.set_ylabel('Occurrences')
ax.set_xticks(ind)
ax.set_xticklabels(top_positive_words, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig('Business_words.png')
plt.show()





