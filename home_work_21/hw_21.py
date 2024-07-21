import numpy as np
from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cosine
import gensim.downloader as api


phrases = [
    "To be, or not to be, that is the question.",
    "All the world’s a stage, and all the men and women merely players.",
    "A horse! a horse! my kingdom for a horse!"
]


def preprocess_text(text):
    return text.lower().split()

processed_phrases = [preprocess_text(phrase) for phrase in phrases]
w2v_model = Word2Vec(sentences=processed_phrases, vector_size=100, window=5, min_count=1, workers=4)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_phrases)]
d2v_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
ft_model = FastText(sentences=processed_phrases, vector_size=100, window=5, min_count=1, workers=4)
glove_vectors = api.load("glove-wiki-gigaword-100")

def sentence_vector_w2v(sentence, model):
    words = preprocess_text(sentence)
    return np.mean([model.wv[word] for word in words if word in model.wv], axis=0)

def sentence_vector_d2v(sentence, model):
    return model.infer_vector(preprocess_text(sentence))

def sentence_vector_ft(sentence, model):
    words = preprocess_text(sentence)
    return np.mean([model.wv[word] for word in words if word in model.wv], axis=0)

def sentence_vector_glove(sentence, model):
    words = preprocess_text(sentence)
    return np.mean([model[word] for word in words if word in model], axis=0)
vec_w2v = [sentence_vector_w2v(phrase, w2v_model) for phrase in phrases]
vec_d2v = [sentence_vector_d2v(phrase, d2v_model) for phrase in phrases]
vec_ft = [sentence_vector_ft(phrase, ft_model) for phrase in phrases]
vec_glove = [sentence_vector_glove(phrase, glove_vectors) for phrase in phrases]
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)
print("Cosine similarities (Word2Vec):")
for i in range(len(vec_w2v)):
    for j in range(i+1, len(vec_w2v)):
        print(f"Phrase {i+1} vs Phrase {j+1}: {cosine_similarity(vec_w2v[i], vec_w2v[j])}")

print("Cosine similarities (Doc2Vec):")
for i in range(len(vec_d2v)):
    for j in range(i+1, len(vec_d2v)):
        print(f"Phrase {i+1} vs Phrase {j+1}: {cosine_similarity(vec_d2v[i], vec_d2v[j])}")

print("Cosine similarities (FastText):")
for i in range(len(vec_ft)):
    for j in range(i+1, len(vec_ft)):
        print(f"Phrase {i+1} vs Phrase {j+1}: {cosine_similarity(vec_ft[i], vec_ft[j])}")

print("Cosine similarities (GloVe):")
for i in range(len(vec_glove)):
    for j in range(i+1, len(vec_glove)):
        print(f"Phrase {i+1} vs Phrase {j+1}: {cosine_similarity(vec_glove[i], vec_glove[j])}")