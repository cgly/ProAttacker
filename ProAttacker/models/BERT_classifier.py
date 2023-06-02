import os

import torch
import torch.nn as nn

from dataloaders.BERT_cls_loader import BERTClsDataloader
from transformers import BertForSequenceClassification,AlbertForSequenceClassification,RobertaForSequenceClassification,DistilBertModel,DistilBertForSequenceClassification,AutoModelForSequenceClassification
class BERTinfer(nn.Module):
    def __init__(self,
                 attack_model,
                 pretrained_dir,
                 nclasses,
                 case,
                 batch_size=64,
                 attack_second=False,
                 model=None):
        super(BERTinfer, self).__init__()
        # construct dataset loader
        self.case = case
        self.dataset = BERTClsDataloader(case,pretrained_dir)
        if ('distil' in str(attack_model)) or ('roberta' in str(attack_model)):
            self.remove_token_typy_flag=True
        else:
            self.remove_token_typy_flag = False
        # construct model
        if model == None:
            if 'distil' in str(attack_model):
                self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_dir,num_labels=nclasses).cuda()
            elif 'roberta' in str(attack_model):
                self.model = RobertaForSequenceClassification.from_pretrained(pretrained_dir,num_labels=nclasses).cuda()
            elif 'albert' in str(attack_model):
                self.model = AlbertForSequenceClassification.from_pretrained(pretrained_dir,num_labels=nclasses).cuda()
            else:  # bert
                self.model = BertForSequenceClassification.from_pretrained(pretrained_dir,num_labels=nclasses).cuda()
        else:
            self.model = model.cuda()
        # Switch the model to eval mode.
        self.model.eval()
        self.batch_size = batch_size
        self.attack_second = attack_second
        
    def convert_to_cap(self, texts, marks):
        assert len(texts[0]) == len(marks)
        for i in range(len(texts)):
            for j in range(len(marks)):
                if marks[j] == 'cap':
                    texts[i][j] = texts[i][j].capitalize()
                elif marks[j] == 'upper':
                    texts[i][j] = texts[i][j].upper()
        return texts

    def text_pred(self, text1, text2=None, marks=None):
        assert text2 is None or isinstance(text2, str)
        # for textfooler and base, convert lower case to uppercase
        if marks is not None and self.case == 'cased':
            text1 = self.convert_to_cap(text1, marks)
        text1 = [" ".join(x) for x in text1]
        text2 = [text2] * len(text1)
        # for two sentences attack, if only one sentence,
        # the text2 is None
        if self.attack_second:
            texts = list(zip(text2, text1))
        else:
            texts = list(zip(text1, text2))
        
        # transform text data into indices and create batches
        if self.remove_token_typy_flag:
            dataloader = self.dataset.transform_text_notype(
                texts, max_seq_length=None, batch_size=self.batch_size)

            probs_all = []
            for input_ids, input_mask in dataloader:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                #token_ids = token_ids.cuda()

                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=input_mask)[0]
                    probs = nn.functional.softmax(logits, dim=-1)
                    probs_all.append(probs)

            return torch.cat(probs_all, dim=0)
        else:
            dataloader = self.dataset.transform_text(
                texts, max_seq_length=None, batch_size=self.batch_size)

            probs_all = []
            for input_ids, input_mask, token_ids in dataloader:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                token_ids = token_ids.cuda()

                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=token_ids)[0]
                    probs = nn.functional.softmax(logits, dim=-1)
                    probs_all.append(probs)

            return torch.cat(probs_all, dim=0)
