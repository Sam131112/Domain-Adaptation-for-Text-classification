import transformers
import os
import math
print(transformers.__version__)
import torch
model_checkpoint = "bert-base-cased"
#model_checkpoint = "allenai/scibert_scivocab_cased"
#model_checkpoint = "roberta-large"
batch_size = 16
from transformers import AutoConfig
import transformers
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,\
                                        TrainingArguments, Trainer,default_data_collator
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
from collections import Counter
from datasets import load_dataset,load_metric
import datasets
import numpy as np
import random

# Baseline Bert-SO from the paper 

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_function(examples):
    result = tokenizer(examples['text'], padding="max_length", max_length=512, truncation=True)
    return result

def compute_metrics(p):
    f1_metric = load_metric("f1")
    pr_metric = load_metric('precision')
    re_metric = load_metric('recall')
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    results1 = f1_metric.compute(predictions=predictions, references=labels,average="weighted")
    results2 = pr_metric.compute(predictions=predictions, references=labels,average="weighted")
    results3 = re_metric.compute(predictions=predictions, references=labels,average="weighted")
    return {
        "precision": results2["precision"],
        "recall": results3["recall"],
        "f1": results1["f1"],
    }


def main():   
            
            seed = np.random.randint(0,1000000)
            fix_all_seeds(seed)
            dataset = load_dataset('csv',delimiter="\t",data_files='books/review_labels.csv')
            dataset = datasets.concatenate_datasets([dataset['train']])
            dataset_src = dataset.train_test_split(0.2,shuffle=False)
            
            dataset = load_dataset('csv',delimiter="\t",data_files='electronics/review_labels.csv')
            dataset = datasets.concatenate_datasets([dataset['train']])
            dataset_trg = dataset.train_test_split(0.2,shuffle=False)
            
            processed_datasets_src = dataset_src.map(preprocess_function,batched=True,\
                                      desc="Running tokenizer on dataset",)

            processed_datasets_trg = dataset_trg.map(preprocess_function,batched=True,\
                                      desc="Running tokenizer on dataset",)
            
            processed_datasets_src.remove_columns_(["text"])
            processed_datasets_trg.remove_columns_(["text"])
            
            config = AutoConfig.from_pretrained(model_checkpoint,num_labels=2,)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,config=config,)
            
            args = TrainingArguments(
                "sanity-chunk",
                evaluation_strategy = "epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=5,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_steps=100,
                overwrite_output_dir=True,
                load_best_model_at_end=True,
                metric_for_best_model = "eval_f1",
                seed = seed,
            )
            
            trainer = Trainer(
                    model,
                    args,
                    train_dataset=processed_datasets_trg['train'],
                    eval_dataset= processed_datasets_trg['test'],
                    data_collator=default_data_collator,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                )
            
            trainer.train(resume_from_checkpoint=None)
            #p = trainer.predict(processed_datasets_src['test'])
            p = trainer.predict(processed_datasets_trg['test'])
            
            y_hat = np.argmax(p.predictions,1)
            y = p.label_ids
            f1_metric = load_metric("f1")
            out = f1_metric.compute(predictions=y_hat,references=y)
            return out['f1']



output = []
for i in range(5):
            print("-"*100)
            output.append(main())
            print("-"*100)
            

print(np.mean(output),np.std(output))
