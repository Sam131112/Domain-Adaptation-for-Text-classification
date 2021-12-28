from datasets import load_dataset,load_metric
import datasets
import pickle
import os
from itertools import permutations
import time
import numpy as np
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import copy
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    SchedulerType,
    get_scheduler,
    set_seed,
    AutoModel,
    default_data_collator,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
model_checkpoint = "bert-base-cased"
import random
from torch.utils.data import DataLoader,RandomSampler
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
data_collator_mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.25)

f1_metric = load_metric("f1")
pr_metric = load_metric('precision')
re_metric = load_metric('recall')


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,comb_data):
        self.data = comb_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_data,mlm_data = self.data[idx][0],self.data[idx][1]
        return token_data, mlm_data



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

    
def tokenize_function(examples):
            examples["text"] = [
                line for line in examples["text"] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=False,
                return_special_tokens_mask=True,
            )

def load_data_tokenize(path):
        raw_datasets = load_dataset("text", data_files=path)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                "text",
                data_files=path,
                split=f"train[:{20}%]",
            )
            raw_datasets["train"] = load_dataset(
                 "text",
                data_files=path,
                split=f"train[{20}%:]",
            )
        tokenized_datasets = raw_datasets.map(tokenize_function,batched=True,remove_columns=["text"])
        tokenized_datasets = tokenized_datasets.map(group_texts,batched=True)
        return tokenized_datasets

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

def create_data_list(data1,data2):
    data1 = [batch for step,batch in enumerate(data1)]
    data2 = [batch for step,batch in enumerate(data2)]
    store = []
    if min(len(data1),len(data2))==len(data1):
        small = data1
        big = data2
    else:
        small = data2
        big = data1
    for i in range(len(small)):
        store.append((big[i],small[i]))
    for j in range(i,len(big),1):
        sample = int(np.random.randint(0,len(small),1)[0])
        store.append((big[j],small[sample]))
    return store

def group_texts(examples):
            max_seq_length = 512
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        
def get_data_slice(i):
                    inputs = {
                   'input_ids':i['input_ids'].cuda(),\
                   'attention_mask':i['attention_mask'].cuda(),\
                   'labels':i['labels'].cuda(),
                       }
                    return inputs 


class Udalm(torch.nn.Module):
    def __init__(self,labels,alpha):
        super().__init__()
        self.num_labels = labels
        self.alpha = alpha
        self.base_model = AutoModel.from_pretrained(model_checkpoint,output_hidden_states=False)
        self.dropout = torch.nn.Dropout(self.base_model.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size,self.num_labels)
        self._init_weights(self.classifier)
        self.mlm = BertOnlyMLMHead(self.base_model.config)
        self._init_weights(self.mlm)
        
    def _init_weights(self, modules):
        """Initialize the weights"""
        for module in modules.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        

    def forward(self,token_data,mlm_data):
        out1 = self.base_model(input_ids=token_data['input_ids'].squeeze(1), \
                               attention_mask=token_data['attention_mask'].squeeze(1))
        out2 = self.base_model(input_ids=mlm_data['input_ids'].squeeze(1), \
                               attention_mask=mlm_data['attention_mask'].squeeze(1))
        token_clf_out = self.classifier(self.dropout(out1[1]))
        mlm_out = self.mlm(out2.last_hidden_state)
        return (token_clf_out,mlm_out)
    
    
def prepare_data(in_domain,out_domain):
            dataset = load_dataset('csv',delimiter="\t",data_files=f'{in_domain}/review_labels.csv')
            dataset = datasets.concatenate_datasets([dataset['train']])
            dataset_src = dataset.train_test_split(0.2,shuffle=False)
            
            dataset_trg = load_dataset('csv',delimiter="\t",data_files=f'{out_domain}/review_labels.csv')
            #dataset = datasets.concatenate_datasets([dataset['train']])
            #dataset_trg = dataset.train_test_split(0.2,shuffle=False)
            
            processed_datasets_src = dataset_src.map(preprocess_function,batched=True,\
                                      desc="Running tokenizer on dataset",)

            processed_datasets_trg = dataset_trg.map(preprocess_function,batched=True,\
                                      desc="Running tokenizer on dataset",)
            
            processed_datasets_src.remove_columns_(["text"])
            processed_datasets_trg.remove_columns_(["text"])
            tokenized_data_mlm = load_data_tokenize(f"{out_domain}/reduced_rev.txt")
            train_dataloader_mlm = DataLoader(tokenized_data_mlm['train'],collate_fn=data_collator_mlm,\
                                     batch_size = 1,drop_last=True)
            eval_dataloader_mlm = DataLoader(tokenized_data_mlm['validation'],\
                                             collate_fn=data_collator_mlm,\
                                             batch_size = 1,drop_last=True)
            
            train_dataloader_src =DataLoader(processed_datasets_src['train'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 1,drop_last=True)
            eval_dataloader_src = DataLoader(processed_datasets_src['test'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 1,drop_last=True)
            test_dataloader_tgt = DataLoader(processed_datasets_trg['train'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 1,drop_last=True)
            
            train_data = create_data_list(train_dataloader_src,train_dataloader_mlm)
            eval_data = create_data_list(eval_dataloader_src,eval_dataloader_mlm)
            final_test_data = create_data_list(test_dataloader_tgt,train_dataloader_mlm)
            
            train_final = CombinedDataset(train_data)
            eval_final = CombinedDataset(eval_data)
            test_final = CombinedDataset(final_test_data)
            
            final_train_loader = DataLoader(train_final, batch_size=16,\
                                            shuffle=True,drop_last=True)
            final_eval_loader = DataLoader(eval_final, batch_size=16,\
                                           shuffle=True,drop_last=True)
            
            final_test_loader = DataLoader(test_final,batch_size=16,\
                                           shuffle=True,drop_last=True)
            
            return final_train_loader,final_eval_loader,final_test_loader
        
    
def run_train(final_train_loader,final_eval_loader):
            model = Udalm(2,0.35)
            model.cuda()
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                    {
                            "params": [p for n, p in model.named_parameters() \
                                       if not any(nd in n for nd in no_decay)],
                            "weight_decay": 1e-2,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() \
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                        ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
            fct_loss = CrossEntropyLoss()
            scheduler = ExponentialLR(optimizer=optimizer,gamma=0.95,last_epoch=-1,verbose=True)
            
            best_f1 = -1
            best_loss = 1e5
            for epoch in range(6):
                print(f'EPOCH NO: {epoch}')
                model.eval()
                val_loss = 0.0
                val_loss_p1 = 0.0
                val_loss_p2 = 0.0
                token_predictions_store = []
                token_gold_store = []
                for step, batch in enumerate(final_eval_loader):
                    with torch.no_grad():
                        input1 = get_data_slice(batch[0])
                        input2 = get_data_slice(batch[1])
                        out1,out2 = model(input2,input1)
                        token_predictions_store.append(out1)
                        token_gold_store.append(input2['labels'])
                        prediction_loss = fct_loss(out1.view(-1,model.num_labels),\
                                                   input2['labels'].view(-1))
                        masked_lm_loss = fct_loss(out2.view(-1,\
                                                  model.base_model.config.vocab_size),\
                                                  input1['labels'].view(-1))
                        loss = prediction_loss*model.alpha + masked_lm_loss*(1-model.alpha)
                        val_loss_p1 = prediction_loss.item()+ val_loss_p1
                        val_loss_p2 = masked_lm_loss.item()+ val_loss_p2
                        val_loss = val_loss + loss.item()

                predictions = torch.vstack(token_predictions_store)
                references = torch.vstack(token_gold_store)
                predictions = torch.argmax(predictions,dim=-1)
                print(predictions.shape,references.shape)
                y_pred = predictions.detach().cpu().clone().numpy()
                y_true = references.squeeze(1).detach().cpu().clone().numpy()
                print(y_pred.shape,y_true.shape)
                eval_f1 = f1_metric.compute(predictions=y_pred, references=y_true)
                print('-'*100)
                print(eval_f1)
                print(f'Epoch {epoch} validation loss {val_loss/len(final_eval_loader)}')
                print(f'Epoch {epoch} validation loss part1 {val_loss_p1/len(final_eval_loader)}')
                print(f'Epoch {epoch} validation loss part2 {val_loss_p2/len(final_eval_loader)}')
                if eval_f1['f1'] > best_f1:
                    best_f1 = eval_f1['f1']
                    best_loss = val_loss/len(final_eval_loader)
                    torch.save(model.state_dict(),"saved_model/udalm_amazon.bin")
                print('-'*100)
        
                model.train()
                epoch_loss = 0.0
                epoch_loss_p1 = 0.0
                epoch_loss_p2 = 0.0
                for step, batch in enumerate(final_train_loader):
                    input1 = get_data_slice(batch[0])
                    input2 = get_data_slice(batch[1])
                    optimizer.zero_grad()
                    out1,out2 = model(input2,input1)
                    prediction_loss = fct_loss(out1.view(-1,model.num_labels),\
                                               input2['labels'].view(-1))
                    masked_lm_loss = fct_loss(out2.view(-1,model.base_model.config.vocab_size),\
                                              input1['labels'].view(-1))
                    loss = prediction_loss*model.alpha + masked_lm_loss*(1-model.alpha)
                    epoch_loss_p1 = prediction_loss.item()+ epoch_loss_p1
                    epoch_loss_p2 = masked_lm_loss.item()+ epoch_loss_p2
                    epoch_loss = epoch_loss + loss.item()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print(f'Epoch {epoch} training loss {epoch_loss/len(final_train_loader)}')
                print(f'Epoch {epoch} training loss part1 {epoch_loss_p1/len(final_train_loader)}')
                print(f'Epoch {epoch} training loss part2 {epoch_loss_p2/len(final_train_loader)}')
                print('**************************************************************************')
            print(f'Best eval F1, loss {best_f1},{best_loss}')
    


def run_test(data):
    
        model = Udalm(2,0.35)
        model.cuda()
        model.load_state_dict(torch.load("saved_model/udalm_amazon.bin"))
        model.eval()
        token_predictions_store = []
        token_gold_store = []
        for step, batch in enumerate(data):
            with torch.no_grad():
                input1 = get_data_slice(batch[0])
                input2 = get_data_slice(batch[1])
                out1,out2 = model(input2,input1)
                token_predictions_store.append(out1)
                token_gold_store.append(input2['labels'])

        predictions = torch.vstack(token_predictions_store)
        references = torch.vstack(token_gold_store)
        predictions = torch.argmax(predictions,dim=-1)
        print(predictions.shape,references.shape)
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.squeeze(1).detach().cpu().clone().numpy()
        test_f1 = f1_metric.compute(predictions=y_pred, references=y_true)
        print(f'Test F1 score {test_f1}')
        return test_f1['f1']

        
def main(in_domain,out_domain):
        
            seed = np.random.randint(0,10000)
            fix_all_seeds(seed)
            final_train_loader,final_eval_loader,final_test_loader = prepare_data(in_domain,out_domain)       
            run_train(final_train_loader,final_eval_loader)                        
            return (seed,run_test(final_test_loader))   


        
start_time = time.perf_counter()
f_name = [z for z in permutations(['dvd','music','books','kitchen_housewares','electronics'],2)]
for in_domain,out_domains in f_name:
            test_best = {}
            for i in range(3):
                    out = main(in_domain,out_domains)
                    test_best[out[0]] = out[1]
            print(test_best.values(),np.mean(list(test_best.values())),np.std(list(test_best.values())))
            fname = out_domains+"_"+in_domain+".p"
            pickle.dump(test_best,open(f'baseline_amazon/udalm/{fname}',"wb"))
end_time = time.perf_counter()
print(f'Time Elapsed {(end_time-start_time)/60.0}')
