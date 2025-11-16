# src/models_transformer.py
"""
Robust transformer training script that supports older/newer transformers versions.
- If TrainingArguments accepts evaluation_strategy, it uses the modern Trainer flow.
- Otherwise it falls back to a simpler TrainingArguments and runs eval manually.
"""
import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def prepare_datasets(train_csv, val_csv, tokenizer, max_length=256, debug=False):
    data_files = {}
    if train_csv is not None:
        data_files['train'] = train_csv
    if val_csv is not None:
        data_files['validation'] = val_csv

    ds = load_dataset('csv', data_files=data_files)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=max_length)

    if debug:
        if 'train' in ds:
            ds['train'] = ds['train'].select(range(min(200, len(ds['train']))))
        if 'validation' in ds:
            ds['validation'] = ds['validation'].select(range(min(200, len(ds['validation']))))

    tokenized = ds.map(tokenize, batched=True)
    return tokenized

def train_transformer(train_csv, val_csv, model_name, out_dir, epochs, batch_size, lr, max_len, debug, seed):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = prepare_datasets(train_csv, val_csv, tokenizer, max_length=max_len, debug=debug)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Try to create TrainingArguments with modern kwargs; fallback if not supported
    use_modern_args = True
    try:
        training_args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=min(64, batch_size*2),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=seed,
            fp16=False,
            save_total_limit=2
        )
    except TypeError:
        # Older transformers - fallback (no evaluation_strategy/load_best_model_at_end)
        use_modern_args = False
        training_args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=min(64, batch_size*2),
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=200,
            seed=seed,
            fp16=False,
            save_total_limit=2
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get('train', None),
        eval_dataset=tokenized.get('validation', None) if use_modern_args else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if use_modern_args else None,
    )

    # If modern args are available, Trainer will handle evaluation and best-model saving.
    if use_modern_args:
        trainer.train()
        trainer.save_model(out_dir)
        print(f"Saved transformer model to {out_dir}")
        return

    # Otherwise manually train for `epochs` epochs and evaluate after each epoch
    print("Using fallback training loop (no built-in evaluation). Training epoch-by-epoch and evaluating manually.")
    for epoch in range(int(epochs)):
        print(f"Starting epoch {epoch+1}/{int(epochs)}")
        trainer.train(resume_from_checkpoint=None)
        # manual evaluation
        if 'validation' in tokenized:
            print("Running manual evaluation on validation set...")
            # run predictions
            eval_preds = trainer.predict(tokenized['validation'])
            # compute metrics using compute_metrics
            metrics = compute_metrics(eval_preds)
            print(f"Validation metrics (epoch {epoch+1}): {metrics}")
        # save after each epoch
        ckpt_dir = os.path.join(out_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.save_model(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")

    print(f"Training finished. Final model saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=False, help="train csv path")
    parser.add_argument("--val", required=False, help="val csv path")
    parser.add_argument("--model", default="distilbert-base-uncased", help="transformer model name")
    parser.add_argument("--output", default="models/transformer", help="output dir")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--debug", action="store_true", help="run quick debug (small subset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train_transformer(args.train, args.val, args.model, args.output, args.epochs, args.batch_size, args.lr, args.max_len, args.debug, args.seed)
