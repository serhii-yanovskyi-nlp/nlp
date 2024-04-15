import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

data = pd.read_csv('articles.csv')
data_list = data['text'].tolist()

tokenizer = RegexpTokenizer(r'\w+')

tfi_df = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range=(1, 1),
                        tokenizer=tokenizer.tokenize)

train_data = tfi_df.fit_transform(data_list)

num_topics = 10

model = LatentDirichletAllocation(n_components=num_topics)
lda_matrix = model.fit_transform(train_data)
lda_components = model.components_
terms = tfi_df.get_feature_names_out()

for index, component in enumerate(lda_components):
    zipped = zip(terms, component)
    top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:10]
    top_terms_list = list(dict(top_terms_key).keys())
    print("Topic " + str(index) + ": ", top_terms_list)
