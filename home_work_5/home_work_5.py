import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

data = pd.read_csv("data.csv")
print(data)

reviews = data["review"]
sentiments = data["sentiment"]
sentiments_encoded = label_encoder.fit_transform(sentiments)


label_encoder = LabelEncoder()
sentiments_encoded = label_encoder.fit_transform(sentiments)



def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))  # conversion into set for fast searching
    review = [w for w in review if w not in swords]
    # splitted paragraph'ları space ile birleştiriyoruz return
    return(" ".join(review))



vectorizer = TfidfVectorizer(preprocessor=process)
X = vectorizer.fit_transform(reviews)


X_train, X_test, y_train, y_test = train_test_split(X, sentiments_encoded, test_size=0.2, random_state=42)

#Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)

#SGDClassifier
sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
sgd_pred = sgd_model.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_pred)
print("SGDClassifier Accuracy:", sgd_accuracy)

#RidgeClassifier
ridge_model = RidgeClassifier()
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_accuracy = accuracy_score(y_test, ridge_pred)
print("RidgeClassifier Accuracy:", ridge_accuracy)


