import nltk
import gensim
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt')
sentences = ["hello im bhavya", "sun rises in the east", "The sun is shining brightly", "We enjoyed a peaceful walk in the park."]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
model = Word2Vec(tokenized_sentences, vector_size = 100, window = 5, min_count = 1, sg = 1)
vector = model.wv["east"]
print(vector)
