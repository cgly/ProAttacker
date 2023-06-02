import os
import random
import time
from collections import Counter
import pandas as pd
import nltk
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, AutoTokenizer, AlbertForMaskedLM,AlbertTokenizer,BertTokenizer,BertForMaskedLM

from config import load_arguments
from utils.hyper_parameters import class_names, nclasses, thres
from dataloaders.dataloader import read_corpus
from models.similarity_model import USE
#from models.similarity_model_zhuhe import USE_torch
from models.BERT_classifier import BERTinfer
from models.attack_location_search import get_attack_sequences,get_target_attack_sequences,get_bae_attack_sequences
from models.attack_operations import *
from models.pipeline import FillMaskPipeline
#from models.Roberta import RobertaForMaskedLM
from transformers import RobertaForMaskedLM
from evaluate import evaluate
from transformers import BertForMaskedLM,BertTokenizer
import wandb


# for token check
import re
punct_re = re.compile(r'\W')
words_re = re.compile(r'\w')

#from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

def attack(example, predictor, stop_words_set, fill_mask,args, sim_predictor=None,
           synonym_num=50, attack_second=False, attack_loc=None,
           thres_=None):
    true_label = example[0]
    if attack_second:
        text_ls = example[2].split()
        text2 = example[1]
    else:
        text_ls = example[1].split()
        text2 = example[2]
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], text2).squeeze()
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max()

    #初始预测就出错
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, []
    num_queries = 1



    # find attack sequences according to predicted probablity change
    attack_sequences, num_query = get_attack_sequences(args,
        text_ls, fill_mask, predictor, sim_predictor,
        orig_probs, orig_label, stop_words_set, punct_re, words_re,
        text2=text2, attack_loc=attack_loc, thres=thres_)
    num_queries += num_query

    # perform attack sequences
    attack_logs = []
    text_prime = text_ls.copy()
    prev_prob = orig_prob
    insertions = []
    merges = []
    forbid_replaces = set()
    forbid_inserts = set()
    forbid_merges = set(range(5))
    num_changed = 0
    new_label = orig_label
    for attack_info in attack_sequences:
        num_queries += synonym_num
        idx = attack_info[0]
        attack_type = attack_info[1]
        orig_token = attack_info[2]
        # check forbid replace operations
        if attack_type == 'insert' and idx in forbid_inserts:
            continue
        if attack_type == 'merge' and idx in forbid_merges:
            continue
        if attack_type == 'replace' and idx in forbid_replaces:
            continue
        
        # shift the attack index by insertions history
        shift_idx = idx
        for prev_insert_idx in insertions:
            if idx >= prev_insert_idx:
                shift_idx +=1
        for prev_merge_idx in merges:
            if idx >= prev_merge_idx + 1:
                shift_idx -= 1
        
        if attack_type == 'replace':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                untarget_word_replacement(
                    args,shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, sim_predictor, text2, thres=thres_)
        elif attack_type == 'insert':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                untarget_word_insertion(
                    args,shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, punct_re, words_re, sim_predictor, text2, thres=thres_)
        elif attack_type == 'merge':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                untarget_word_merge(
                    args,shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, sim_predictor, text2, thres=thres_)
        
        if prob_diff < 0:
    #         import ipdb; ipdb.set_trace()
            if attack_type == 'replace':
                text_prime[shift_idx] = synonym
                # forbid_inserts.add(idx)
                # forbid_inserts.add(idx+1)
                forbid_merges.add(idx-1)
                forbid_merges.add(idx)
            elif attack_type == 'insert':
                text_prime.insert(shift_idx, synonym)
                # append original attack index
                insertions.append(idx)
                forbid_merges.add(idx-1)
                # forbid_merges.add(idx)
                for i in [-1, 1]:
                    forbid_inserts.add(idx + i)
            elif attack_type == 'merge':
                text_prime[shift_idx] = synonym
                del text_prime[shift_idx+1]
                merges.append(idx)
                # forbid_inserts.add(idx)
                forbid_inserts.add(idx+1)
                # forbid_inserts.add(idx+2)
                # forbid_replaces.add(idx-1)
                forbid_replaces.add(idx)
                forbid_replaces.add(idx+1)
                for i in [-1, 1]:
                    forbid_merges.add(idx + i)
            cur_prob = new_prob[orig_label].item()
            attack_logs.append([idx, attack_type, orig_token, synonym, syn_prob,
                                semantic_sim, prob_diff, cur_prob])
            prev_prob = cur_prob
            num_changed += 1
            # if attack successfully!
            #祝贺：攻击成功条件判断
            if np.argmax(new_prob) != orig_label:
                new_label = np.argmax(new_prob)
                break

    return ' '.join(text_prime), num_changed, orig_label, new_label, num_queries, attack_logs

def nornal_main(args,plm_path,read_num,dataset):
    begin_time = time.time()
    #args = load_arguments('ag')

    # get data to attack
    examples = read_corpus(args.attack_file,read_num=read_num)
    if args.data_size is None:
        args.data_size = len(examples)
    examples = examples[args.data_idx:args.data_idx + args.data_size]  # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    model = BERTinfer(args.target_model, args.target_model_path,
                      nclasses[args.dataset], args.case,
                      batch_size=args.batch_size,
                      attack_second=args.attack_second)
    predictor = model.text_pred
    print("Model built!")

    # prepare context predictor
    #for Linux envs
    # tokenizer = RobertaTokenizer.from_pretrained("/home/zhuhe/distilroberta-base")
    # model = RobertaForMaskedLM.from_pretrained("/home/zhuhe/distilroberta-base")
    #for windows envs
    if 'roberta' in str(plm_path):
        tokenizer = RobertaTokenizer.from_pretrained(plm_path)
        model = RobertaForMaskedLM.from_pretrained(plm_path)
    elif 'albert' in str(plm_path):
        tokenizer = AlbertTokenizer.from_pretrained(plm_path)
        model = AlbertForMaskedLM.from_pretrained(plm_path)
    # elif 'de' in str(plm_path): #deberta
    #     tokenizer = DebertaTokenizer.from_pretrained(plm_path)
    #     model = DebertaForMaskedLM.from_pretrained(plm_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(plm_path)
        model = BertForMaskedLM.from_pretrained(plm_path)

    # mask LM
    fill_mask = FillMaskPipeline(model, tokenizer, topk=args.synonym_num)

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    num_sample = 0
    orig_failures = 0.
    adv_failures = 0.
    skipped_idx = []
    changed_rates = []
    nums_queries = []
    attack_texts = []
    target_record=[]
    new_texts = []
    label_names = class_names[args.dataset]


    stop_words_set = set(nltk.corpus.stopwords.words('english'))
    print('Start attacking!')

    # 用于类别统计
    class_num = {}  # 类别数
    fail_class_num = {}  # 初始预测失败数
    success_class_num = {}  # 初始预测成功数
    success_class_rate = {}  # 初始预测成功比
    targetAttack_fail_num = {}  # 每一个类别 多少初始预测成功 但是攻击失败的
    targetAttack_num = {}  # 类别分流数
    targetAttack_rate = {}  # 类别分流攻击比 = 类别分流数 / 初始预测成功数 （反映在预测成功的数据中，攻击每一类的成功率）

    for idx, example in enumerate(tqdm(examples)):
        true_label = example[0]
        if example[2] is not None:
            single_sentence = False
            attack_text = example[2] if args.attack_second else example[1]
            ref_text = example[1] if args.attack_second else example[2]
        else:
            single_sentence = True
            attack_text = example[1]
        if len(tokenizer.encode(attack_text)) > args.max_seq_length:
            skipped_idx.append(idx)
            continue
        num_sample += 1

        # 统计每类数量
        class_num[true_label] = class_num.get(true_label, 0) + 1

        # 攻击的核心代码
        #  '', 0, orig_label, orig_label, 0, []

        new_text, num_changed, orig_label, new_label, num_queries, attack_logs = \
            attack(example, predictor, stop_words_set,
                   fill_mask,args, sim_predictor=use,
                   synonym_num=args.synonym_num,
                   attack_second=args.attack_second,
                   attack_loc=args.attack_loc,
                   thres_=thres[args.dataset])
        ######初始预测阶段失败
        if true_label != orig_label:
            orig_failures += 1
            # 统计每一类分类失败的数量
            fail_class_num[true_label] = fail_class_num.get(true_label, 0) + 1
        else:
            nums_queries.append(num_queries)

        changed_rate = 1.0 * num_changed / len(attack_text.split())

        #初始预测成功 但是无目标攻击失败
        if true_label == orig_label and true_label == new_label:
            targetAttack_fail_num[true_label] = targetAttack_fail_num.get(true_label, 0) + 1

        # 初始预测成功 无目标攻击成功
        if true_label == orig_label and true_label != new_label:
            sympol = str(true_label) + str(new_label)
            targetAttack_num[sympol] = targetAttack_num.get(sympol, 0) + 1


            adv_failures += 1
            attack_texts.append(attack_text)
            #
            target_record.append(sympol)
            new_texts.append(new_text)
            changed_rates.append(changed_rate)


    #测试分类数据
    # print(class_num)
    # print(fail_class_num)
    # print(target_class_num)
    success_class_num=dict(Counter(class_num) - Counter(fail_class_num))

    # 计算预测成功率 success_class_rate = {}  # 初始预测成功比 success_class_num/ class_num
    for key, val in success_class_num.items():
        success_class_rate[key] = val / class_num[key]

    #计算类别分流率 targetAttack_rate = {}  # 类别分流攻击比 = 类别分流数 / 初始预测成功数 （反映在预测成功的数据中，攻击每一类的成功率）

    target_success_class_num = dict(Counter(success_class_num) - Counter(targetAttack_fail_num))
    target_class_rate = {}
    for key, val in target_success_class_num.items():
        target_class_rate[key] = val/success_class_num[key]


    for key,val in targetAttack_num.items():
        ori_class=int(key[0])
        targetAttack_rate[key]=val/target_success_class_num[ori_class]

    #根据攻击类别 计算详细数据
    text_dic = {
        "target_record": target_record,
        "attack_texts": attack_texts,
        "new_texts": new_texts,
        "changed_rates": changed_rates
    }
    eval_list = pd.DataFrame(text_dic)
    mutil_target_res = {}
    name_list = ['orig_ppl', 'adv_ppl', 'bert_score', 'sim_score', 'gram_err','changed_rate']

    #需要按类别统计时启动
    # for val in set(target_record):
    #     current_list = eval_list[eval_list["target_record"] == val]
    #     att_text = current_list["attack_texts"].tolist()
    #     new_text = current_list["new_texts"].tolist()
    #     changed_rate = np.mean(current_list["changed_rates"].tolist())
    #     #  add
    #     with torch.no_grad():
    #         orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(att_text, new_text, use, args)
    #     mutil_target_res[val] = dict(zip(name_list, [orig_ppl, adv_ppl, bert_score, sim_score, gram_err,changed_rate]))


    orig_acc = (1 - orig_failures / num_sample) * 100
    attack_rate = 100 * adv_failures / (num_sample - orig_failures)
    message = '\nFor Generated model {} / Target model {} : original accuracy: {:.3f}%, attack success: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, num of samples: {:d}, time: {:.1f}\n'.format(
        args.sample_file, args.target_model, orig_acc, attack_rate,
        np.mean(changed_rates) * 100, np.mean(nums_queries), num_sample, time.time() - begin_time)
    print(message)
    torch.cuda.empty_cache()



    orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(attack_texts, new_texts, use, args)

    #计算每一类的测评指标

    print("Skipped indices: ", skipped_idx)
    print("Processing time: %d" % (time.time() - begin_time))
    #写入训练参数
    train_conifg='target func: {0},Use prompt: {1} ,prompt_template: {2} ,prompt_location: {3} ,prompt_target: {4} , read_num: {5} ,search_mothed: {6}\n'. \
        format('normal class',args.use_prompt,args.prompt_template,args.prompt_location,args.prompt_target,read_num ,args.attack_loc)

    ###

    #2022年3月29日20:00:49 测试wandb
    config = wandb.config  # Initialize config
    config.MaskModel =  args.mlm_model
    config.TargetModel =args.target_model
    config.sample_num=  read_num
    config.attack_loc = args.attack_loc
    config.dataset_name=thres[args.dataset].get('dataset_name')
    #
    config.replace_prob = thres[args.dataset].get('replace_prob')
    config.insert_prob = thres[args.dataset].get('insert_prob')
    config.merge_prob = thres[args.dataset].get('merge_prob')
    #
    config.replace_sim = thres[args.dataset].get('replace_sim')
    config.insert_sim = thres[args.dataset].get('insert_sim')
    config.merge_sim = thres[args.dataset].get('merge_sim')
    #
    config.prob_diff = thres[args.dataset].get('prob_diff')
    config.sim_window = thres[args.dataset].get('sim_window')
    #prompt

    config.attack_model = args.attack_model
    config.model_idx = args.model_idx

    config.use_prompt = args.use_prompt
    config.prompt_template=args.prompt_template
    config.prompt_location=args.prompt_location
    config.prompt_target=args.prompt_target
    config.label_names = label_names

    wandb.log({
        # "Generated model": args.sample_file,
        # "Target model": args.target_model,
        "original accuracy":orig_acc,
        "attack success": attack_rate,
        "avg_changed_rate": np.mean(changed_rates) * 100,
        "num of queries": np.mean(nums_queries),
        "num of samples": num_sample,
        "Original ppl": orig_ppl,
        "Adversarial ppl": adv_ppl,
        "BertScore": bert_score,
        "SimScore": sim_score,
        "gram_err": gram_err,
        "time":time.time() - begin_time,
        "classes_num":str(class_num), ## 各类类别数
        "success_class_num":str(success_class_num),# 初始预测成功数
        "success_class_rate":str(success_class_rate), # 初始预测成功比
        "target_class_rate":str(target_class_rate), #攻击成功率
        "targetAttack_num":str(targetAttack_num),# 类别分流数
        "targetAttack_rate":str(targetAttack_rate),# 类别攻击 成功率
        "mutil_target_res":str(mutil_target_res) #eval 测评信息
    })

    wandb.save(os.path.join(args.output_dir, args.sample_file))



if __name__ == "__main__":


    args = load_arguments()

    #build wandb project name
    project_name="last-ProAttacker-"+str(args.dataset)
    wandb.init(project=project_name)

    #select MASK LM

    plm_path = os.path.join('./MLM_model', args.mlm_model)

    print("start ###single### attack")
    nornal_main(args,plm_path,read_num=int(args.read_num),dataset=args.dataset)

    torch.cuda.empty_cache()