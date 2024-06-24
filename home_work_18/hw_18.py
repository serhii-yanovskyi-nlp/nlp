import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.metrics import classification_report
import random

# Функция для автоматической разметки текстов
def auto_annotate(texts):
    nlp = spacy.load("en_core_web_sm")
    annotated_data = []
    for text in texts:
        doc = nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        annotated_data.append((text, {"entities": entities}))
    return annotated_data

# Функция для обучения модели NER
def train_ner_model(train_data, n_iter=20):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {itn}, Loss: {losses}")

    return nlp

# Оценка модели на тестовых данных
def evaluate_model(model, test_data):
    true_entities = []
    predicted_entities = []
    for text, annot in test_data:
        doc = model(text)
        true_entities.append([ent[2] for ent in annot["entities"]])
        predicted_entities.append([ent.label_ for ent in doc.ents])
    flatten = lambda l: [item for sublist in l for item in sublist]
    true_entities = flatten(true_entities)
    predicted_entities = flatten(predicted_entities)
    if len(true_entities) != len(predicted_entities):
        print("Warning: Number of true entities and predicted entities do not match.")
    if predicted_entities:
        print(classification_report(true_entities, predicted_entities))
    else:
        print("No entities found in test data for evaluation.")

file_path = "training_data.txt"
train_texts = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            train_texts.append(line)
test_texts = [
    "Apple's headquarters are situated in California.",
    "Jeff Bezos founded Amazon in Seattle, Washington.",
]
train_data = auto_annotate(train_texts)
test_data = auto_annotate(test_texts)
ner_model = train_ner_model(train_data)
evaluate_model(ner_model, test_data)
