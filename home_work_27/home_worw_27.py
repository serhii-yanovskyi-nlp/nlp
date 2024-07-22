from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, DefaultDataCollator
import tensorflow as tf
import pandas as pd
from datasets import Dataset
from transformers import pipeline

df = pd.read_csv('qa_dataset.csv')

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = [eval(answer) for answer in examples["answers"]]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]["answer_start"][0]
        start_char = answer
        end_char = start_char + len(answers[i]["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_idx = context_start
            while start_idx < len(offset) and offset[start_idx][0] <= start_char:
                start_idx += 1
            start_positions.append(start_idx - 1)

            end_idx = context_end
            while end_idx >= 0 and offset[end_idx][1] >= end_char:
                end_idx -= 1
            end_positions.append(end_idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = tokenized_datasets.to_tf_dataset(
    columns=["attention_mask", "input_ids", "start_positions", "end_positions"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator
)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer)

model.fit(tf_train_dataset, epochs=3)

question_answering_model = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "Who designed the Eiffel Tower?"
context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."

result = question_answering_model(question=question, context=context)
print(f"Answer: {result['answer']}")
