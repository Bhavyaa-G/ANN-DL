
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# Sample text
text = "Natural Language Processing is a fascinating field of AI."

# Tokenize the text
tokens = word_tokenize(text)

# Get stop words
stop_words = set(stopwords.words('english'))

# Filter tokens to include only stop words
stop_word_tokens = [word for word in tokens if word.lower() in stop_words]

# Perform POS tagging
pos_tags = nltk.pos_tag(stop_word_tokens)

print("POS Tags of Stop Words:", pos_tags)