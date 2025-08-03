
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/johndpope/fake-news-detection/master/fake_or_real_news.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))

# Predict
text = input("Enter news text:\n")
prediction = model.predict([text])[0]
print(f"\nPrediction: {prediction}")
