"""this is the last assignment of Natural language proccessing
   this was made by group 11 comprised of:
   Miruna Lungu (S5882206)
   Andrejs Tupikins (S5607442)
   Prayer Aguebor (S5901782)
"""

import re
import os
import html
import evaluate
import torch as pt
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

target_names = ["World", "Sports", "Business", "Sci/Tech"]

ABBREVIATIONS = {
    r"\b(?:US|U\.S\.?)\b": "America",
    r"\b(?:EU|E\.U\.?)\b": "European Union",
    r"\b(?:U\.K\.?|UK)\b": "United Kingdom",
    r"\b(?:U\.S\.A\.?|USA)\b": "United States of America",
    r"\b(?:UN|U\.N\.?)\b": "United Nations",
    r"\b(?:NATO)\b": "North Atlantic Treaty Organization",
    r"\b(?:IMF)\b": "International Monetary Fund",
}

CONTRACTIONS = {
    r"won\s*['']t": "will not",
    r"can\s*['']t": "can not",
    r"shan\s*['']t": "shall not",
    r"should\s*['']t": "should not",
    r"was\s*['']t": "was not",
    r"is\s*['']t": "is not",
    r"what\s*['']s": "what is",
    r"that\s*['']s": "that is",
    r"he\s*['']s": "he is",
    r"she\s*['']s": "she is",
    r"it\s*['']s": "it is",
    r"i\s*['']m": "i am",
    r"(\w+)\s*['']ve": r"\1 have",
    r"(\w+)\s*['']re": r"\1 are",
    r"(\w+)\s*['']ll": r"\1 will",
    r"(\w+)\s*['']d": r"\1 would",
    r"(\w+)\s*n't": r"\1 not",
}

def text_preprocessing(text):
    text = re.sub(r"(?<!&)#(\d+);", r"&#\1;", text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()

    for pattern, full_name in ABBREVIATIONS.items():
        text = re.sub(pattern, full_name, text, flags=re.IGNORECASE)
    for pattern, repl in CONTRACTIONS.items():
        text = re.sub(pattern, repl, text, flags=re.I)

    text = text.lower()
    text = re.sub(r"(\d+(?:\.\d+)?)\s*million", r"\1m", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*billion", r"\1b", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*thousand", r"\1k", text, flags=re.I)
    text = re.sub(r"\d+(?:\.\d+)?", "<num>", text)

    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    text = url_pattern.sub("", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\b(\w+)\s+'s\b", r"\1's", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess(example):
    example["text"] = text_preprocessing(example["text"])
    return example

news_dataset = load_dataset("ag_news")

train_dataset = news_dataset["train"].map(preprocess, num_proc=4)
test_dataset = news_dataset["test"].map(preprocess, num_proc=4)

checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=4)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

test_valid = test_dataset.train_test_split(test_size=0.5, seed=42)
val_data = test_valid["train"]
test_data = test_valid["test"]
train_data = train_dataset

training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=2000,
    save_steps=2000,
    eval_strategy='steps',
    save_strategy='steps',
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=2000,
    logging_strategy='steps',
    seed=42
)

metric_accuracy = evaluate.load("accuracy")
metric_f1_score = evaluate.load("f1")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1_score.compute(predictions=predictions, references=labels, average="weighted")
    precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

save_dir_bert = "saved_models"
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
early_stop = EarlyStoppingCallback(2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if os.path.isdir(save_dir_bert):
    tokenizer_FT = DistilBertTokenizer.from_pretrained(save_dir_bert)
    model_FT = DistilBertForSequenceClassification.from_pretrained(save_dir_bert).to(device)
    trainer = Trainer(
        model=model_FT,
        args=training_args,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
else:
    print("Training model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[early_stop],
        data_collator=data_collator,
    )

    trainer.train()
    model.eval()
    model.save_pretrained(save_dir_bert)
    tokenizer.save_pretrained(save_dir_bert)

predictions = trainer.predict(test_data)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_data["labels"].numpy()

print(classification_report(true_labels, predicted_labels, target_names=target_names))
ConfusionMatrixDisplay.from_predictions(
    true_labels, predicted_labels, display_labels=target_names, cmap="Blues"
)
plt.show()
plt.savefig("confusion_matrix.png", bbox_inches="tight")