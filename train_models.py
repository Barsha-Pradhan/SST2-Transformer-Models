print("Script started")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

# Load SST2 dataset
dataset = load_dataset("glue", "sst2")

# Smaller dataset for faster training
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))

# Accuracy metric
metric = evaluate.load("accuracy")

# Smaller models first for faster downloads
models = {
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "albert": "albert/albert-base-v2",
    "mobilebert": "google/mobilebert-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "bert": "google-bert/bert-base-uncased"
}

# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Train all models
for name, checkpoint in models.items():

    print("\n==============================")
    print(f"Training {name}")
    print("==============================\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2
    )

    # Training settings
    training_args = TrainingArguments(
        output_dir=f"./results/{name}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    print(f"\n{name} Results:")
    print(results)

    # Save model
    model.save_pretrained(f"./models/{name}")
    tokenizer.save_pretrained(f"./models/{name}")

print("\nAll models trained successfully.")