import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from scipy.stats import spearmanr

corpus = api.load("wiki-english-20171001")


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


processed_corpus = [preprocess_text(text) for text in corpus]
model = Word2Vec(processed_corpus)
model.save("word2vec_wiki_preprocessed.model")
wordsim_file = api.load("wordsim353.tsv")
human_similarities = []
model_similarities = []
for pair in wordsim_file:
    word1, word2, human_similarity = pair
    if word1 in model.wv and word2 in model.wv:
        model_similarity = model.wv.similarity(word1, word2)
        human_similarities.append(float(human_similarity))
        model_similarities.append(model_similarity)

correlation, _ = spearmanr(human_similarities, model_similarities)
print("Spearman:", correlation)
