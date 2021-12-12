import torch
import transformers
import os
import time
import pickle
import math
from collections import Counter
print(transformers.__version__)
import torch
model_checkpoint = "bert-base-uncased"
batch_size = 16
from transformers import AutoConfig,AutoModel
import transformers
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,\
                                        TrainingArguments, Trainer,default_data_collator,AdamW
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
from torch.utils.data import DataLoader,RandomSampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from transformers.models.bert.modeling_bert import BertPooler
from datasets import load_dataset,load_metric
import numpy as np
import random
import datasets
from torch.autograd import Function

f1_metric = load_metric("f1")
pr_metric = load_metric('precision')
re_metric = load_metric('recall')
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import ConcatDataset
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,comb_data):
        self.data = comb_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_data,mlm_data = self.data[idx][0],self.data[idx][1]
        return token_data, mlm_data
def get_data_slice(i):
                    inputs = {
                   'input_ids':i['input_ids'].squeeze(1).cuda(),\
                   'attention_mask':i['attention_mask'].squeeze(1).cuda(),\
                   'labels':i['labels'].squeeze(1).cuda(),
                       }
                    return inputs 

def preprocess_function(examples):
        result = tokenizer(examples['text'], padding="max_length", max_length=512, truncation=True)
        return result

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_data_equal(data1,data2):
    # Downsample MLM data
    data1 = [batch for step,batch in enumerate(data1)]
    data2 = [batch for step,batch in enumerate(data2)]
    data2_indices = np.random.choice(len(data2),len(data1))
    data2_final = [data2[i] for i in data2_indices]
    store = []
    for i,j in zip(data1,data2_final):
        store.append((i,j))
    return store
    
    

def create_data_list(data1,data2):
    # Upsample Classification Data 
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


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class SentiMentClf(torch.nn.Module):
    def __init__(self,base_config,num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.base_config = base_config
        self.dropout = torch.nn.Dropout(self.base_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.base_config.hidden_size,self.num_labels)
        self._init_weights(self.classifier)
    
    
    def _init_weights(self, modules):
        """Initialize the weights"""
        for module in modules.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.base_config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
    
    
    def forward(self,data):
        clf_out = self.classifier(self.dropout(data))
        return clf_out




class DANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_checkpoint,output_hidden_states=False,\
                                                    add_pooling_layer=False)
        self.pooler = BertPooler(self.base_model.config)
        self._init_weights(self.pooler)
        self.task_classifier = SentiMentClf(self.base_model.config,2)
        self.domain_classifier = SentiMentClf(self.base_model.config,2)
        
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
        

    def forward(self,task_data,domain_data=False,train_mode=True):
        if train_mode:
            out = self.base_model(input_ids=task_data['input_ids'], \
                               attention_mask=task_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            task_out = self.task_classifier(out)
            
            out = self.base_model(input_ids=domain_data['input_ids'], \
                               attention_mask=domain_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            domain_out = self.domain_classifier(out)
            return task_out,domain_out
        else:
            out = self.base_model(input_ids=task_data['input_ids'], \
                               attention_mask=task_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            task_out = self.task_classifier(out)
            return task_out
        
        


class DANNauto(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_checkpoint,output_hidden_states=False,\
                                                    add_pooling_layer=False)
        self.pooler = BertPooler(self.base_model.config)
        self._init_weights(self.pooler)
        self.task_classifier = SentiMentClf(self.base_model.config,2)
        self.domain_classifier = SentiMentClf(self.base_model.config,2)
        self.my_grad = ReverseLayerF.apply
        
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
        

    def forward(self,task_data,domain_data=False,train_mode=True):
        if train_mode:
            out = self.base_model(input_ids=task_data['input_ids'], \
                               attention_mask=task_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            task_out = self.task_classifier(out)
            
            out = self.base_model(input_ids=domain_data['input_ids'], \
                               attention_mask=domain_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            domain_out = self.domain_classifier(self.my_grad(out,1))
            return task_out,domain_out
        else:
            out = self.base_model(input_ids=task_data['input_ids'], \
                               attention_mask=task_data['attention_mask'])           
            out = self.pooler(out.last_hidden_state)
            task_out = self.task_classifier(out)
            return task_out

def prepare_data(in_domain,out_domain):
        seed = np.random.randint(0,10000)
        fix_all_seeds(seed)

        dataset = load_dataset('csv',delimiter="\t",data_files=f'{in_domain}/review_labels.csv')
        dataset = datasets.concatenate_datasets([dataset['train']])
        dataset_src = dataset.train_test_split(0.2,shuffle=True)

        dataset = load_dataset('csv',delimiter="\t",data_files=f'{out_domain}/review_labels.csv')
        dataset = datasets.concatenate_datasets([dataset['train']])
        dataset_trg = dataset.train_test_split(0.2,shuffle=True)

        if os.path.exists(f'Multitask_data/multitask_{in_domain}_n_{out_domain}_data.csv'):
            dataset_domain = load_dataset('csv',delimiter="\t",\
                                          data_files=f'Multitask_data/multitask_{in_domain}_n_{out_domain}_data.csv')
        elif os.path.exists(f'Multitask_data/multitask_{out_domain}_n_{in_domain}_data.csv'):
            dataset_domain = load_dataset('csv',delimiter="\t",\
                                          data_files=f'Multitask_data/multitask_{out_domain}_n_{in_domain}_data.csv')
        dataset = datasets.concatenate_datasets([dataset_domain['train']])
        dataset_domain = dataset.train_test_split(0.2,shuffle=True)
        
        processed_datasets_src = dataset_src.map(preprocess_function,batched=True,\
                                      desc="Running tokenizer on dataset",)

        processed_datasets_trg = dataset_trg.map(preprocess_function,batched=True,\
                                  desc="Running tokenizer on dataset",)
        
        processed_domain = dataset_domain.map(preprocess_function,batched=True,\
                                  desc="Running tokenizer on dataset",)
        
        

        processed_datasets_src.remove_columns_(["text"])
        processed_datasets_trg.remove_columns_(["text"])
        processed_domain.remove_columns_(["text"])
        
        
        train_dataloader_src =DataLoader(processed_datasets_src['train'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 1,drop_last=True)
        eval_dataloader_src =DataLoader(processed_datasets_src['test'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 16,drop_last=True)
        
        test_dataloader_trg =DataLoader(processed_datasets_trg['test'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 16,drop_last=True)
        
        train_dataloader_domain = DataLoader(processed_domain['train'],\
                                             collate_fn=default_data_collator,\
                                             batch_size = 1,drop_last=True)
        
        
        train_data = create_data_equal(train_dataloader_src,train_dataloader_domain)
        print("Check Data Sizes",len(train_dataloader_src),len(train_dataloader_domain),len(train_data))
        train_final = CombinedDataset(train_data)
        final_train_loader = DataLoader(train_final, batch_size=16,\
                                            shuffle=True,drop_last=True)       
        
        return final_train_loader,eval_dataloader_src,test_dataloader_trg

def run_train_a(train_loader,eval_loader):
            seed = np.random.randint(0,100)
            fix_all_seeds(seed)
            model = DANN()
            model.cuda()
            no_decay = ["bias", "LayerNorm.weight"]
            param_all1 = {}
            param_all2 = {}
            param_all3 = {}
            for n,p in model.named_parameters():
                if 'domain_classifier' not in n and 'task_classifier' not in n:
                    param_all1[n]=p
            
            for n,p in model.named_parameters():
                if 'task_classifier' in n:
                    param_all2[n]=p
            
            for n,p in model.named_parameters():
                if 'domain_classifier' in n:
                    param_all3[n]=p
                    
                    
            optimizer_grouped_transformer = [
                    {
                            "params": [p for n, p in param_all1.items() \
                                       if not any(nd in n for nd in no_decay)],
                            "weight_decay": 1e-2,
                    },
                    {
                        "params": [p for n, p in param_all1.items() \
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                        ]
                    
            optimizer_grouped_task = [
                    {
                            "params": [p for n, p in param_all2.items() \
                                       if not any(nd in n for nd in no_decay)],
                            "weight_decay": 1e-2,
                    },
                    {
                        "params": [p for n, p in param_all2.items() \
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                        ]
            
            
            optimizer_grouped_domain = [
                    {
                            "params": [p for n, p in param_all3.items() \
                                       if not any(nd in n for nd in no_decay)],
                            "weight_decay": 1e-2,
                    },
                    {
                        "params": [p for n, p in param_all3.items() \
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                        ]
            optimizer_encoder = AdamW(optimizer_grouped_transformer, lr=5e-5)
            optimizer_task = AdamW(optimizer_grouped_task, lr=5e-5)
            optimizer_domain = AdamW(optimizer_grouped_domain, lr=5e-5)
            fct_loss = CrossEntropyLoss()
            scheduler1 = ExponentialLR(optimizer=optimizer_encoder,gamma=0.85,last_epoch=-1,verbose=True)
            scheduler2 = ExponentialLR(optimizer=optimizer_task,gamma=0.85,last_epoch=-1,verbose=True)
            scheduler3 = ExponentialLR(optimizer=optimizer_domain,gamma=0.85,last_epoch=-1,verbose=True)
            
            best_f1 = -1
            best_loss = 1e5
            for epoch in range(5):
                print(f'EPOCH NO: {epoch}')
                model.eval()
                val_loss = 0.0
                token_predictions_store = []
                token_gold_store = []
                for step, batch in enumerate(eval_loader):
                    with torch.no_grad():
                        input1 = {'input_ids':batch['input_ids'].cuda(),\
                                  'attention_mask':batch['attention_mask'].cuda(),\
                                     'labels':batch['labels'].cuda()}
                        out  = model(input1,False,False)
                        token_predictions_store.append(out)
                        token_gold_store.append(input1['labels'])
                        loss = fct_loss(out,input1['labels'])
                        val_loss = val_loss + loss.item()

                predictions = torch.vstack(token_predictions_store)
                references = torch.hstack(token_gold_store)
                predictions = torch.argmax(predictions,axis=1)
                print(predictions.shape,references.shape)
                y_pred = predictions.detach().cpu().clone().numpy()
                y_true = references.detach().cpu().clone().numpy()
                eval_f1 = f1_metric.compute(predictions=y_pred, references=y_true)
                print('-'*100)
                print(eval_f1)
                print(f'Epoch {epoch} validation loss {val_loss/len(eval_loader)}')
                if eval_f1['f1'] > best_f1:
                    best_f1 = eval_f1['f1']
                    best_loss = val_loss/len(eval_loader)
                    torch.save(model.state_dict(),"saved_model/dann_amazon.bin")
                print('-'*100)
        
                model.train()
                epoch_loss = 0.0
                domain_epoch_loss = 0.0
                for step, batch in enumerate(train_loader):
                    input1 = get_data_slice(batch[1])
                    input2 = get_data_slice(batch[0])
                    out1,out2 = model(input1,input2,True)
                    task_loss = fct_loss(out1,input1['labels'])
                    domain_loss = fct_loss(out2,input2['labels'])
                    epoch_loss = epoch_loss + task_loss.item()
                    domain_epoch_loss = domain_epoch_loss + domain_loss.item()
                    task_loss.backward()
                    optimizer_encoder.step()
                    optimizer_task.step()
                    optimizer_encoder.zero_grad()
                    optimizer_task.zero_grad()
                    domain_loss.backward()
                    for param in optimizer_encoder.param_groups[0]['params']:
                        if param.grad is not None:
                            param.grad.data.mul_(-1)
                    for param in optimizer_encoder.param_groups[1]['params']:
                        if param.grad is not None:
                            param.grad.data.mul_(-1)
                    optimizer_encoder.step()
                    optimizer_domain.step()
                    optimizer_encoder.zero_grad()
                    optimizer_domain.zero_grad()
                    
                scheduler1.step()
                scheduler2.step()
                scheduler3.step()
                print(f'Epoch {epoch} training task loss {epoch_loss/len(train_loader)}')
                print(f'Epoch {epoch} training domain loss {domain_epoch_loss/len(train_loader)}')
                print('**************************************************************************')
            print(f'Best eval F1, loss {best_f1},{best_loss}')
            return seed

def run_train_b(train_loader,eval_loader):
            seed = np.random.randint(0,100)
            fix_all_seeds(seed)
            model = DANNauto()
            model.cuda()
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped = [
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
            optimizer = AdamW(optimizer_grouped, lr=5e-5)
            fct_loss = CrossEntropyLoss()
            scheduler = ExponentialLR(optimizer=optimizer,gamma=0.85,last_epoch=-1,verbose=True)
            
            best_f1 = -1
            best_loss = 1e5
            for epoch in range(5):
                print(f'EPOCH NO: {epoch}')
                model.eval()
                val_loss = 0.0
                token_predictions_store = []
                token_gold_store = []
                for step, batch in enumerate(eval_loader):
                    with torch.no_grad():
                        input1 = {'input_ids':batch['input_ids'].cuda(),\
                                  'attention_mask':batch['attention_mask'].cuda(),\
                                     'labels':batch['labels'].cuda()}
                        out  = model(input1,False,False)
                        token_predictions_store.append(out)
                        token_gold_store.append(input1['labels'])
                        loss = fct_loss(out,input1['labels'])
                        val_loss = val_loss + loss.item()

                predictions = torch.vstack(token_predictions_store)
                references = torch.hstack(token_gold_store)
                predictions = torch.argmax(predictions,axis=1)
                print(predictions.shape,references.shape)
                y_pred = predictions.detach().cpu().clone().numpy()
                y_true = references.detach().cpu().clone().numpy()
                eval_f1 = f1_metric.compute(predictions=y_pred, references=y_true)
                print('-'*100)
                print(eval_f1)
                print(f'Epoch {epoch} validation loss {val_loss/len(eval_loader)}')
                if eval_f1['f1'] > best_f1:
                    best_f1 = eval_f1['f1']
                    best_loss = val_loss/len(eval_loader)
                    torch.save(model.state_dict(),"saved_model/dann_amazon.bin")
                print('-'*100)
        
                model.train()
                epoch_loss = 0.0
                domain_epoch_loss = 0.0
                for step, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    input1 = get_data_slice(batch[1])
                    input2 = get_data_slice(batch[0])
                    out1,out2 = model(input1,input2,True)
                    task_loss = fct_loss(out1,input1['labels'])
                    domain_loss = fct_loss(out2,input2['labels'])
                    epoch_loss = epoch_loss + task_loss.item()
                    domain_epoch_loss = domain_epoch_loss + domain_loss.item()
                    loss = task_loss + domain_loss
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print(f'Epoch {epoch} training task loss {epoch_loss/len(train_loader)}')
                print(f'Epoch {epoch} training domain loss {domain_epoch_loss/len(train_loader)}')
                print('**************************************************************************')
            print(f'Best eval F1, loss {best_f1},{best_loss}')
            return seed
        
def run_test(data):
        model = DANNauto()
        model.cuda()
        model.load_state_dict(torch.load("saved_model/dann_amazon.bin"))
        model.eval()
        token_predictions_store = []
        token_gold_store = []
        for step, batch in enumerate(data):
                    with torch.no_grad():
                        input1 = {'input_ids':batch['input_ids'].cuda(),\
                                  'attention_mask':batch['attention_mask'].cuda(),\
                                     'labels':batch['labels'].cuda()}
                        out  = model(input1,False,False)
                        token_predictions_store.append(out)
                        token_gold_store.append(input1['labels'])

        predictions = torch.vstack(token_predictions_store)
        references = torch.hstack(token_gold_store)
        predictions = torch.argmax(predictions,axis=1)
        print(predictions.shape,references.shape)
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
        test_f1 = f1_metric.compute(predictions=y_pred, references=y_true)
        print(f'Test F1 score {test_f1}')
        return test_f1['f1']
    

def main(in_domain,out_domains):
    train_loader, eval_loader, test_loader = prepare_data(in_domain,out_domains)
    seed = run_train_b(train_loader,eval_loader)
    return (seed,run_test(test_loader))

start_time = time.perf_counter()
out_domains = "dvd"
in_domains = ['music','books','kitchen_housewares','electronics']
for in_domain in in_domains:
    test_best = {}
    for i in range(5):
            out = main(in_domain,out_domains)
            test_best[out[0]] = out[1]
    print(test_best.values(),np.mean(list(test_best.values())),np.std(list(test_best.values())))
    fname = out_domains+"_"+in_domain+".p"
    pickle.dump(test_best,open(f'baseline_out/Dann/{fname}',"wb"))
end_time = time.perf_counter()
print(f'Time Elapsed {(end_time-start_time)/60.0}')