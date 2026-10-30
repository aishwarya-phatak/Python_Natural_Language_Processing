import re
import ssl
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# this function is only for ssl certificate verification process bypass as it raises an error for ssl connection on macOS systems
def setup_nltk_data():
    """Download all required NLTK data"""
    try:
        # Disable SSL verification for corporate networks (optional)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:  # ← Fixed: aligned with the inner try
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"⚠️  Could not download NLTK data: {e}")
        print("Please run: python -m nltk.downloader punkt_tab")


# Run once
setup_nltk_data()

df = pd.read_csv("dataset/IMDB Dataset.csv")

print("------------------")
print(df.head())
print("------------------")
print(df.info())
print("------------------")
print(df.describe())

stop_words = set(stopwords.words('english'))
print(stop_words)

def clean_text(text):
    text = text.lower()
    text = re.sub("r<.*?>"," ",text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [each_word for each_word in words if each_word not in stop_words]
    return ' '.join(words)

df['review_new'] = df['review'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review_new']).toarray()    #imp - convert csr matrix to array as we need to
print("------------------")
print(type(X))

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy * 100,"%")