#stanfordnlp.download('en')
#nltk.download('treebank')
#nltk.download('universal_tagset')
import nltk
import spacy
import stanfordnlp
import time
from nltk.corpus import treebank


dataset = treebank.tagged_sents(tagset='universal')
train_data = dataset[:3000]
test_data = dataset[3000:]
nltk_pos_tagger = nltk.DefaultTagger('NN')
start_time = time.time()
nltk_pos_tagger.tag_sents(train_data)
nltk_training_time = time.time() - start_time
spacy_nlp = spacy.load("en_core_web_sm")


def evaluate_spacy_pos_tagger(test_data):
    correct = 0
    total = 0
    for sentence in test_data:
        tokens, gold_tags = zip(*sentence)
        doc = spacy_nlp(" ".join(tokens))
        predicted_tags = [token.pos_ for token in doc]
        correct += sum(1 for predicted, gold in zip(predicted_tags, gold_tags) if predicted == gold)
        total += len(gold_tags)
    accuracy = correct / total
    return accuracy

stanford_nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize,pos')

def evaluate_stanford_pos_tagger(test_data):
    correct = 0
    total = 0
    for sentence in test_data:
        tokens, gold_tags = zip(*sentence)
        doc = stanford_nlp(" ".join(tokens))
        predicted_tags = [word.upos for sent in doc.sentences for word in sent.words]
        correct += sum(1 for predicted, gold in zip(predicted_tags, gold_tags) if predicted == gold)
        total += len(gold_tags)
    accuracy = correct / total
    return accuracy

nltk_accuracy = nltk_pos_tagger.evaluate(test_data)
spacy_accuracy = evaluate_spacy_pos_tagger(test_data)
stanford_accuracy = evaluate_stanford_pos_tagger(test_data)


print(f"NLTK POS Tagger Accuracy: {nltk_accuracy:.4f}")
print(f"spaCy POS Tagger Accuracy: {spacy_accuracy:.4f}")
print(f"StanfordNLP POS Tagger Accuracy: {stanford_accuracy:.4f}")

