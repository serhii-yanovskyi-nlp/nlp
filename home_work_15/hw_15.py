import nltk
from sklearn.datasets import fetch_20newsgroups
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


newsgroups_data = fetch_20newsgroups(subset='all')
sentences = nltk.sent_tokenize(" ".join(newsgroups_data.data))
tagged_data = [TaggedDocument(words=nltk.word_tokenize(sent), tags=[str(i)]) for i, sent in enumerate(sentences)]
model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=2, workers=4, epochs=40)
model.save("doc2vec")