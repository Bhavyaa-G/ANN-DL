from collections import Counter, defaultdict

# Example text
text = "I like to play football and I like to watch football."

# Tokenize the text
tokens = nltk.word_tokenize(text.lower())

# Function to generate N-grams
def generate_ngrams(tokens, n):
    return list(nltk.ngrams(tokens, n))

# Function to calculate N-gram probabilities
def ngram_probabilities(ngrams):
    freq = Counter(ngrams)
    total = sum(freq.values())
    probabilities = {ngram: count / total for ngram, count in freq.items()}
    return freq, probabilities
print(f"FREQUENCIES: {freq}")
print(f"PROBABILITIES: {probabilities}")