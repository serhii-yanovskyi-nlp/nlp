import gensim.downloader as api
import pandas as pd


model = api.load("word2vec-google-news-300")
words = ["king", "queen", "apple", "banana"]
similar_words = {word: model.most_similar(word, topn=5) for word in words}

similar_words_dict = {word: [similar_word[0] for similar_word in similar_words[word]] for word in words}
similar_words_df = pd.DataFrame(similar_words_dict)
print(similar_words_df)
word_pairs = [("king", "queen"), ("apple", "banana"), ("man", "woman")]
similarities = {f"{pair[0]}-{pair[1]}": model.similarity(pair[0], pair[1]) for pair in word_pairs}
similarities_df = pd.DataFrame(list(similarities.items()), columns=["word_pair", "similarity"]).set_index("word_pair")
print(similarities_df)

transformation_examples = [
    ("king", "man", "queen"),
    ("apple", "fruit", "banana"),
    ("Paris", "France", "Berlin"),
]

transformation_results = {
    f"{x}-{y}+{z}": model.most_similar(positive=[z, y], negative=[x], topn=1)[0][0] for x, y, z in transformation_examples
}

transformation_results_df = pd.DataFrame(list(transformation_results.items()), columns=["transformation", "result"]).set_index("transformation")

print(transformation_results_df)
