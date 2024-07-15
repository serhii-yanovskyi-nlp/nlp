from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
sentences = [
    ("He sat on the bank of the river.", "He deposited money in the bank."),
    ("She went to the bank to deposit her paycheck.", "The bank of the river was overflowing with water.")
]


def get_word_vector(sentence, word):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.squeeze()
    tokenized_text = tokenizer.tokenize(sentence)
    word_index = tokenized_text.index(word)
    word_vector = last_hidden_states[word_index]
    return word_vector.detach().numpy()
for sent1, sent2 in sentences:
    word1 = "bank" if "bank" in sent1 else sent1.split()[-2]
    word2 = "bank" if "bank" in sent2 else sent2.split()[-1]

    vec1 = get_word_vector(sent1, word1)
    vec2 = get_word_vector(sent2, word2)

    similarity = cosine_similarity([vec1], [vec2])[0][0]
    print(f"Схожість між '{word1}' в реченні '{sent1}' та '{word2}' в реченні '{sent2}': {similarity:.4f}")
