# Domain Adaptation for Token Classification task (NER)
# Application of BertTweet for NER 

import transformers
import os
import pickle
from itertools import permutations
import math
import time
print(transformers.__version__)
import torch
#model_checkpoint = "roberta-base"
#model_checkpoint = "allenai/scibert_scivocab_cased"
#model_checkpoint = "roberta-large"
#model_checkpoint = "bert-base-cased"
model_checkpoint = "vinai/bertweet-base"
batch_size = 16
from transformers import AutoConfig,AutoModel
import transformers
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,\
                                        TrainingArguments, Trainer,default_data_collator,AdamW, \
                                        DataCollatorForTokenClassification
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, normalization=True,)
from torch.utils.data import DataLoader,RandomSampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from transformers.models.bert.modeling_bert import BertPooler
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from collections import Counter
fct = CrossEntropyLoss()
from datasets import load_dataset,load_metric
import datasets
import numpy as np
import random
from seqeval.metrics import accuracy_score as seq_accuracy
from seqeval.metrics import f1_score as seq_f1

data_collator = DataCollatorForTokenClassification(tokenizer)



def compute_metrics(p,label_list):
    metric = load_metric("seqeval")
    predictions, labels = p

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    #results = seq_accuracy(true_labels,true_predictions)
    return (true_predictions,true_labels,{
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        #"accuracy": results,
    })



def my_collator(features):
        print('Feature here')   # For testing collator function
        print(features)
        print("Inside",len(features))
        batch = tokenizer.pad(
            features,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        print("After",batch)
        return batch


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def not_fast(data,length_batch):
    # If Fast tokenizer not available use this function else dont !
    word_id = [None]
    #print("Here",data,len(data))
    for j,lab in enumerate(data):
            temp = tokenizer(lab,is_split_into_words=True,truncation=True,max_length=64,padding=True)['input_ids'][1:-1]
            word_id.extend([j]*len(temp))
    
    if len(word_id)<length_batch-1:
        word_id.extend([None]*(length_batch-len(word_id)-1))
    word_id.append(None)
    #print(word_id,len(word_id),label,len(label))
    #print(word_id,len(word_id))
    return word_id
    
    
def tokenize_and_align_labels(examples):
    task = "chunk"
    label_all_tokens = True
    tokenized_inputs = tokenizer.batch_encode_plus(examples["tokens"],is_split_into_words=True,truncation=True,max_length=64,padding="max_length")
    length_batch = len(tokenized_inputs['input_ids'][0])
    labels = []
    #print(word_ids_all)
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def tokenize_and_align_labels_new(examples):
    task = "chunk"
    label_all_tokens = True
    tokenized_inputs = tokenizer.encode_plus(examples["tokens"],is_split_into_words=True,truncation=True,max_length=64,padding="max_length")
    #print(tokenized_inputs,len(tokenized_inputs['input_ids']))
    length_batch = len(tokenized_inputs['input_ids'])
    labels = []
    word_ids = not_fast(examples["tokens"],length_batch)
    #print(word_ids_all)
    label = examples[f"{task}_tags"]
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label[word_idx])
        else:
            label_ids.append(label[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx


    tokenized_inputs["labels"] = label_ids[:64]  # Eliminate minor corner cases
    #print(tokenized_inputs['input_ids'])
    #print(tokenized_inputs['labels'])
    #print(len(tokenized_inputs['input_ids']),len(tokenized_inputs['labels']))
    return tokenized_inputs


def create_split(dataset,in_domain=True):
    if in_domain:
        data = datasets.concatenate_datasets([dataset['BC'],dataset['BN']])   #NW,MZ,BN,BC
        data_final = data.train_test_split(0.25,shuffle=False)
    else:
        data = datasets.concatenate_datasets([dataset['NW'],dataset['MZ']])
        #data = data.shuffle(seed=42)
        #spliter  = round(len(data)*0.2)
        #train_indices = np.arange(len(data)-2*spliter)
        #train_indices = np.arange(len(data)-spliter)
        #valid_indices = np.arange(len(data)-2*spliter, len(data)-spliter)
        #test_indices = np.arange(len(data)-spliter,len(data))
        #data_train = data.select(train_indices)
        #data_validation = data.select(valid_indices)
        #data_test = data.select(test_indices)
        #data_final = datasets.DatasetDict({"train":data_train,"test":data_test,"validation":data_validation})
        data_final = datasets.DatasetDict({"train":data,})
    return data_final





class PretrainedSequenceModel(torch.nn.Module):
    def __init__(self,labels):
        super().__init__()
        self.num_labels = labels
        self.base_model = AutoModel.from_pretrained(model_checkpoint,output_hidden_states=False,add_pooling_layer=False)
        #Pretrained Using MLM and saved 
        #self.base_model.load_state_dict(torch.load("ontonotes_processed/news_.bin"))
        self.dropout = torch.nn.Dropout(self.base_model.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size,self.num_labels)
        self._init_weights(self.classifier)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
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
        

    def forward(self,data):
        output = self.base_model(input_ids=data['input_ids'], \
                               attention_mask=data['attention_mask'])
        
        sequence_out = self.dropout(output[0])
        logits = self.classifier(sequence_out)
        active_loss = data['attention_mask'].view(-1) == 1
        active_logits = logits.view(-1,self.num_labels)
        active_labels = torch.where(active_loss, data['labels'].view(-1),\
                                    torch.tensor(self.loss_fct.ignore_index).type_as(data['labels']))
        loss = self.loss_fct(active_logits, active_labels)
        return (loss,logits)
    
    
    
def prepare_data():
        task = "chunk"
        #dataset = load_dataset('ontonotes_processed/ontoloader.py')
        dataset = load_dataset("/ukp-storage-1/sarkar/Domain_Adaptation/TwitterNer/data_loader.py")
        #dataset_src = create_split(dataset,in_domain=True)
        #dataset_trg = create_split(dataset,in_domain=False)
        label_list = dataset['train'].features[f"{task}_tags"].feature.names
        processed_datasets = dataset.map(tokenize_and_align_labels_new,batched=False,\
                                  desc="Running tokenizer on dataset",)
        
        #processed_datasets_src = dataset_src.map(tokenize_and_align_labels,batched=True,\
        #                          batch_size =batch_size ,desc="Running tokenizer on dataset",)

        #processed_datasets_trg = dataset_trg.map(tokenize_and_align_labels,batched=True,\
        #                          batch_size =batch_size , desc="Running tokenizer on dataset",)

        processed_datasets.remove_columns_(['chunk_tags', 'id','tokens'])
        #processed_datasets_trg.remove_columns_(['chunk_tags', 'id','tokens'])


        train_dataloader_src =DataLoader(processed_datasets['train'],\
                                                     collate_fn=data_collator,\
                                                     batch_size =batch_size ,drop_last=True)
        eval_dataloader_src = DataLoader(processed_datasets['validation'],\
                                         collate_fn=data_collator,\
                                         batch_size = batch_size ,drop_last=True)
        test_dataloader_tgt = DataLoader(processed_datasets['test'],\
                                         collate_fn=data_collator,\
                                         batch_size =batch_size ,drop_last=True)
        
        return train_dataloader_src,eval_dataloader_src,test_dataloader_tgt,label_list
    

def run_train(model,final_train_loader,final_eval_loader,label_list):
    
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
            #fct_loss = CrossEntropyLoss()
            scheduler = ExponentialLR(optimizer=optimizer,gamma=0.95,last_epoch=-1,verbose=True)
            
            best_loss = 1e5
            best_f1 = -1
            for epoch in range(10):
                print(f'EPOCH NO: {epoch}')
                model.eval()
                val_loss = 0.0
                token_predictions_store = []
                token_gold_store = []
                for step, batch in enumerate(final_eval_loader):
                    with torch.no_grad():
                        data = {'input_ids':batch['input_ids'].cuda(),\
                               'attention_mask':batch['attention_mask'].cuda(),\
                               'labels':batch['labels'].cuda()}
                        loss,logits = model(data)
                        preds = logits.detach().cpu().clone().numpy()
                        preds = np.argmax(preds,axis=2)
                        gold = batch['labels'].cpu().clone().numpy()
                        x_axis , y_axis = preds.shape
                        to_fill = 64-y_axis
                        if y_axis < 64:
                            fill1 = np.empty((x_axis,to_fill),dtype=np.int8)
                            fill2 = np.empty((x_axis,to_fill),dtype=np.int8)
                            fill1.fill(0)
                            fill2.fill(-100)
                            preds = np.concatenate([preds,fill1],axis=1)
                            gold = np.concatenate([gold,fill2],axis=1)
                            
                        #print(preds.shape,gold.shape)
                        token_predictions_store.append(preds)
                        token_gold_store.append(gold)
                        val_loss = val_loss + loss.item()

                
                print(len(token_predictions_store),len(token_gold_store))
                y_pred = np.concatenate(token_predictions_store,axis=0)
                y_true = np.concatenate(token_gold_store,axis=0)
                print(y_pred.shape,y_true.shape)
                _ , _ , eval_ = compute_metrics((y_pred,y_true),label_list=label_list)
                print('-'*100)
                print(eval_)
                print(f'Epoch {epoch} val loss {val_loss/len(final_eval_loader)}')
                if eval_['f1'] > best_f1:
                    best_loss = val_loss/len(final_eval_loader)
                    best_f1 = eval_['f1']
                    torch.save(model.state_dict(),"saved_model/pretrained_onto.bin")
                print('-'*100)
        
                model.train()
                epoch_loss = 0.0
                for step, batch in enumerate(final_train_loader):
                    data = {'input_ids':batch['input_ids'].cuda(),\
                               'attention_mask':batch['attention_mask'].cuda(),\
                               'labels':batch['labels'].cuda()}
                    optimizer.zero_grad()
                    loss,_ = model(data)
                    epoch_loss = epoch_loss + loss.item()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print(f'Epoch {epoch} training loss {epoch_loss/len(final_train_loader)}')
                print('**************************************************************************')
            print(f'Best F1 score {best_f1},{best_loss}')

def run_test(model,data,label_list):
              
        #model = PretrainedSequenceModel(6)
        model.cuda()
        model.load_state_dict(torch.load("saved_model/pretrained_onto.bin"))
        model.eval()
        token_predictions_store = []
        token_gold_store = []
        for step, batch in enumerate(data):
                    with torch.no_grad():
                        data = {'input_ids':batch['input_ids'].cuda(),\
                               'attention_mask':batch['attention_mask'].cuda(),\
                               'labels':batch['labels'].cuda()}
                        loss,logits = model(data)
                        preds = logits.detach().cpu().clone().numpy()
                        preds = np.argmax(preds,axis=2)
                        gold = batch['labels'].cpu().clone().numpy()
                        x_axis , y_axis = preds.shape
                        to_fill = 64-y_axis
                        if y_axis < 64:
                            fill1 = np.empty((x_axis,to_fill),dtype=np.int8)
                            fill2 = np.empty((x_axis,to_fill),dtype=np.int8)
                            fill1.fill(0)
                            fill2.fill(-100)
                            preds = np.concatenate([preds,fill1],axis=1)
                            gold = np.concatenate([gold,fill2],axis=1)
                            
                        #print(preds.shape,gold.shape)
                        token_predictions_store.append(preds)
                        token_gold_store.append(gold)
                        loss = loss + loss.item()

        y_pred = np.concatenate(token_predictions_store,axis=0)
        y_true = np.concatenate(token_gold_store,axis=0)
        print(y_pred.shape,y_true.shape)
        _ , _ , test_ = compute_metrics((y_pred,y_true),label_list=label_list)
        print(f'Test F1 score {test_}')
        return test_['f1']
    

def baseline2():
        train_dataloader_src , eval_dataloader_src,test_dataloader_tgt,label_list = prepare_data()
        model = PretrainedSequenceModel(len(label_list))
        run_train(model,train_dataloader_src,eval_dataloader_src,label_list)
        model = PretrainedSequenceModel(len(label_list))
        return run_test(model,test_dataloader_tgt,label_list)
    
    
    
start_time = time.perf_counter()

for _ in range(1):
            output = []
            for i in range(5):
                    output.append(baseline2())
            print(output,np.mean(output),np.std(output),np.max(output))
            fname = "tweetbert.p"
            pickle.dump(output,open(f'TweetNer/{fname}',"wb"))
end_time = time.perf_counter()
print(f'Time Elapsed {(end_time-start_time)/60.0}')
