#tf-idf

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["The cat sat on the mat",
          "The mat is red",
          "The cat is black"]

vectorizer = TfidfVectorizer()
print("-------------------")
print(vectorizer.vocabulary)

tf_idf_matrix = vectorizer.fit_transform(corpus)
print("-------------------")
print(tf_idf_matrix.shape)
print(tf_idf_matrix)

print(vectorizer.get_feature_names_out())

print("-----------idf calculation through coding---imp ------------")
for e1,e2 in zip(vectorizer.get_feature_names_out(),vectorizer.idf_):
    print(e1,e2)

print("----------------------------------")
df = pd.DataFrame(tf_idf_matrix.toarray(),
                  columns = vectorizer.get_feature_names_out())

print(df.info())
print(df.head())
print(df.describe())