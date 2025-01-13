import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documents=["Word embeddings are a type of word representation.",
    "They are used to map words to vectors of real numbers.",
    "Word2Vec is a popular algorithm for generating word embeddings.",
    "It is often used in natural language processing tasks."]
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(documents)
df=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
print(df)