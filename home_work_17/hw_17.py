import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import MiniBatchKMeans


url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
data = pd.read_csv(url, header=None, names=['Category', 'Title', 'Description'])
data['Text'] = data['Title'] + " " + data['Description']
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(data['Text'])
n_components = 100
rp = SparseRandomProjection(n_components=n_components, random_state=42)
X_projected = rp.fit_transform(X)
n_clusters = 4
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=100)
kmeans.fit(X_projected)
data['Cluster'] = kmeans.labels_
print(data['Cluster'].value_counts())
print(pd.crosstab(data['Category'], data['Cluster']))
incorrect_classifications = data[data['Category'] != data['Cluster']]
print(incorrect_classifications.head())
for i, row in incorrect_classifications.iterrows():
    print(f"Original Category: {row['Category']}, Cluster: {row['Cluster']}")
    print(f"Text: {row['Text']}\n")
