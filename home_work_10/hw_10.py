import pandas as pd
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import random


data = pd.read_csv('Donald-Tweets!.csv')


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text



data['cleaned_text'] = data['Tweet_Text'].apply(preprocess_text)

all_text = ' '.join(data['cleaned_text'])

def generate_ngrams(text, n):
    words = word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return n_grams


n_grams = generate_ngrams(all_text, 3)


freq_dist = FreqDist(n_grams)


def generate_next_word(prev_word):
    if (prev_word,) in freq_dist:
        next_candidates = freq_dist[(prev_word,)]
        return random.choice(next_candidates)[1]
    else:
        return None


def generate_text(seed_word, num_words=20):
    current_word = seed_word
    generated_text = [current_word[0]]

    for _ in range(num_words - 1):
        next_word = generate_next_word(current_word)
        if next_word:
            generated_text.append(next_word)
            current_word = (current_word[1], next_word)
        else:
            break

    return ' '.join(generated_text)


seed_bigram = random.choice(list(freq_dist.keys()))
generated_text = generate_text(seed_bigram, num_words=20)
print(seed_bigram, generated_text)
