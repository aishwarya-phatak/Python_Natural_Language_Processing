#tf-idf

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

corpus = ["The cat sat on the mat",
          "The mat is red",
          "The cat is black"]

# ['cat','sat','mat','red','black']


vectorizer = TfidfVectorizer(stop_words='english')
print("-------------------")
print(vectorizer.vocabulary)

tf_idf_matrix = vectorizer.fit_transform(corpus)
print("-------------------")
print(tf_idf_matrix.shape)
print(tf_idf_matrix)

print(vectorizer.get_feature_names_out())

df = pd.DataFrame(tf_idf_matrix.toarray(),
                  columns = vectorizer.get_feature_names_out())
print(df.info())
print(df.head())
print(df.describe())