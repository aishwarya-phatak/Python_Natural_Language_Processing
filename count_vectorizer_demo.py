#count vectorizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love machine learning",
          "I understand AI",
          "I am learning Natural Language Processing",
          "Machine Learning, Deep Learning, Natural Language Processing",
          "It is difficult to learn statistics and mathematics",
          "I am excited to learn NLP",
          "NLP needs machine learning and deep learning knowledge"]

#count vectorizer
vectorizer = CountVectorizer(stop_words='english')

#vocabulary
print(vectorizer.vocabulary)

#fit transform
count_matrix = vectorizer.fit_transform(corpus)
print(count_matrix)
print(type(count_matrix))
print(type(count_matrix.toarray()))
#feature names
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.DataFrame(count_matrix.toarray(),
             columns=feature_names)

#head
print(df.head())

#info
print(df.info())

#describe
print(df.describe())