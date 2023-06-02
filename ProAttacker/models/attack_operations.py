import numpy as np
import nltk
from utils.hyper_parameters import class_names
from utils.utils import build_prompt_input

# #祝贺添加 一个用于构建模板的方法 现在只适用于ag_news
# def build_propmt_template(targetLabel):
#     labelWord=class_names.get('ag').get(targetLabel)
#     prompt_template="It is a "+labelWord+" news, "
#     return prompt_template

def similairty_calculation(indices, orig_texts, new_texts,
                           sim_predictor, attack_types=None, thres=None):
    # compute semantic similarity
    half_sim_window = (thres['sim_window'] - 1) // 2
    orig_locals = []
    new_locals = []
    for i in range(len(indices)):
        idx = indices[i]
        len_text = len(orig_texts[i])
        if idx >= half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = idx - half_sim_window
            text_range_max = idx + half_sim_window + 1
        elif idx < half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = 0
            text_range_max = thres['sim_window']
        elif idx >= half_sim_window and len_text - idx - 1 < half_sim_window:
            text_range_min = len_text - thres['sim_window']
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text
        orig_locals.append(" ".join(orig_texts[i][text_range_min:text_range_max]))
        if attack_types[i] == 'merge':
            text_range_max -= 1
        if attack_types[i] == 'insert':
            text_range_min -= 1
        new_locals.append(" ".join(new_texts[i][text_range_min:text_range_max]))

    return sim_predictor.semantic_sim(orig_locals, new_locals)[0]

def word_replacement(args,targetLabel,target_prob,replace_idx, text_prime, generator, target_model,
                     orig_prob, orig_label, sim_predictor, text2=None, thres=None):
    len_text = len(text_prime)
    orig_token = text_prime[replace_idx]
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[replace_idx] = '<mask>'
    mask_token = [orig_token]
    #祝贺 在此处添加 prompt信息 用作第二次生成
    #build_propmt_template(targetLabel)

    tar_labelWord = class_names.get(thres.get('dataset_name')).get(targetLabel)
    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input=build_prompt_input(" ".join(mask_input), args, tar_labelWord, ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input, mask_token, mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:replace_idx] + [synonym] + text_prime[min(replace_idx + 1, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()

    tar_prob_diffs = (-(new_probs[:, targetLabel] - target_prob)).cpu().numpy()

    # compute semantic similarity
    semantic_sims = similairty_calculation(
        [replace_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['replace'] * len(new_texts), thres=thres)
    
    # create filter mask
    #按约束条件过滤的核心代码
    # 祝贺修改
    # attack_mask = prob_diffs < thres['prob_diff']
    #attack_mask = tar_prob_diffs <0 #5。16之前运行都得判断条件
    attack_mask = tar_prob_diffs <thres['prob_diff']

    prob_mask = syn_probs > thres['replace_prob']
    semantic_mask = semantic_sims > thres['replace_sim']

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0

    tar_prob_diffs *= (attack_mask * semantic_mask * prob_mask*maskSymbol_mask)
    best_idx = np.argmin(tar_prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            if (synonyms[i] == '[MASK]' or synonyms[i] == '<mask>'): continue
            collections.append([tar_prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], tar_prob_diffs[best_idx], semantic_sims[best_idx], \
            new_probs[best_idx].cpu().numpy(), collections


def word_insertion(args,targetLabel,target_prob,insert_idx, text_prime, generator, target_model,
                   orig_prob, orig_label, punct_re, words_re, sim_predictor,
                   text2=None, thres=None):
    len_text = len(text_prime)
    mask_input = text_prime.copy()
    mask_input.insert(insert_idx, '<mask>')

    # 祝贺 在此处添加 prompt信息 用作第二次生成
    tar_labelWord = class_names.get(thres.get('dataset_name')).get(targetLabel)
    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input = build_prompt_input(" ".join(mask_input), args, tar_labelWord, ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input)
    #synonyms, syn_probs = generator([build_propmt_template(targetLabel) + " ".join(mask_input)])
    synonyms, syn_probs = synonyms[0], syn_probs[0]


    new_texts = [text_prime[:insert_idx] + [synonym] + text_prime[min(insert_idx, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()

    tar_prob_diffs = (-(new_probs[:, targetLabel] - target_prob)).cpu().numpy()

    semantic_sims = similairty_calculation(
        [insert_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['insert'] * len(new_texts), thres=thres)
    
    # create filter mask
    # 祝贺修改
    # attack_mask = prob_diffs < thres['prob_diff']

    # attack_mask = tar_prob_diffs <0 #5。16之前运行都得判断条件
    attack_mask = tar_prob_diffs < thres['prob_diff']

    prob_mask = syn_probs > thres['insert_prob']
    semantic_mask = semantic_sims > thres['insert_sim']
    punc_mask = np.ones(attack_mask.shape)
    for i in range(len(punc_mask)):
        # don't insert punctuation
        if punct_re.search(synonyms[i]) is not None and words_re.search(synonyms[i]) is None:
            punc_mask[i] = 0

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0


    tar_prob_diffs *= (attack_mask * punc_mask * prob_mask * semantic_mask*maskSymbol_mask)
    best_idx = np.argmin(tar_prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and punc_mask[i] and prob_mask[i]:
            if (synonyms[i] == '[MASK]' or synonyms[i] == '<mask>'): continue
            collections.append([tar_prob_diffs[i], syn_probs[i], semantic_sims[i], text_prime[insert_idx-1], synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], tar_prob_diffs[best_idx], \
            semantic_sims[best_idx], new_probs[best_idx].cpu().numpy(), collections
            

def word_merge(args,targetLabel,target_prob,merge_idx, text_prime, generator, target_model,
               orig_prob, orig_label, sim_predictor, text2=None, thres=None):



    len_text = len(text_prime)
    orig_token = " ".join([text_prime[merge_idx], text_prime[merge_idx+1]])
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[merge_idx] = '<mask>'
    del mask_input[merge_idx+1]
    mask_token = [orig_token]

    # 祝贺 在此处添加 prompt信息 用作第二次生成
    tar_labelWord = class_names.get(thres.get('dataset_name')).get(targetLabel)
    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input = build_prompt_input(" ".join(mask_input), args, tar_labelWord, ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input, mask_token,mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:merge_idx] + [synonym] + text_prime[min(merge_idx + 2, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()
    #祝贺添加
    tar_prob_diffs = (-(new_probs[:, targetLabel] - target_prob)).cpu().numpy()

    semantic_sims = similairty_calculation(
        [merge_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['merge'] * len(new_texts), thres=thres)

    # create filter mask
    #祝贺修改
    #attack_mask = prob_diffs < thres['prob_diff']

    # attack_mask = tar_prob_diffs <0 #5。16之前运行都得判断条件
    attack_mask = tar_prob_diffs < thres['prob_diff']

    prob_mask = syn_probs > thres['merge_prob']
    semantic_mask = semantic_sims > thres['merge_sim']

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0

    tar_prob_diffs *= (attack_mask * semantic_mask * prob_mask * maskSymbol_mask)
    best_idx = np.argmin(tar_prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            if (synonyms[i] == '[MASK]' or synonyms[i] == '<mask>'): continue
            collections.append([tar_prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], tar_prob_diffs[best_idx], semantic_sims[best_idx], \
            new_probs[best_idx].cpu().numpy(), collections


def untarget_word_replacement(args,replace_idx, text_prime, generator, target_model,
                     orig_prob, orig_label, sim_predictor, text2=None, thres=None):
    len_text = len(text_prime)
    orig_token = text_prime[replace_idx]
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[replace_idx] = '<mask>'
    mask_token = [orig_token]

    #在此处添加 prompt信息 用作第二次生成


    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input = build_prompt_input(" ".join(mask_input), args, "no target", ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input, mask_token, mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:replace_idx] + [synonym] + text_prime[min(replace_idx + 1, len_text):] for synonym in
                 synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()

    # compute semantic similarity
    semantic_sims = similairty_calculation(
        [replace_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['replace'] * len(new_texts), thres=thres)

    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['replace_prob']
    semantic_mask = semantic_sims > thres['replace_sim']

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0

    prob_diffs *= (attack_mask * semantic_mask * prob_mask * maskSymbol_mask)
    best_idx = np.argmin(prob_diffs)

    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            if (synonyms[i] == '[MASK]' or synonyms[i] == '<mask>'):
                continue
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x: x[0])

    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], semantic_sims[best_idx], \
           new_probs[best_idx].cpu().numpy(), collections


def untarget_word_insertion(args,insert_idx, text_prime, generator, target_model,
                   orig_prob, orig_label, punct_re, words_re, sim_predictor,
                   text2=None, thres=None):
    len_text = len(text_prime)
    mask_input = text_prime.copy()
    mask_input.insert(insert_idx, '<mask>')

    # 祝贺 在此处添加 prompt信息 用作第二次生成

    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input = build_prompt_input(" ".join(mask_input), args, "no target", ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:insert_idx] + [synonym] + text_prime[min(insert_idx, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()

    semantic_sims = similairty_calculation(
        [insert_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['insert'] * len(new_texts), thres=thres)

    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['insert_prob']
    semantic_mask = semantic_sims > thres['insert_sim']
    punc_mask = np.ones(attack_mask.shape)
    for i in range(len(punc_mask)):
        # don't insert punctuation
        if punct_re.search(synonyms[i]) is not None and words_re.search(synonyms[i]) is None:
            punc_mask[i] = 0

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0


    prob_diffs *= (attack_mask * punc_mask * prob_mask * semantic_mask*maskSymbol_mask)
    best_idx = np.argmin(prob_diffs)

    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and punc_mask[i] and prob_mask[i]:
            if (synonyms[i]=='[MASK]' or synonyms[i]=='<mask>'): continue
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], text_prime[insert_idx - 1], synonyms[i]])
    collections.sort(key=lambda x: x[0])

    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], \
           semantic_sims[best_idx], new_probs[best_idx].cpu().numpy(), collections


def untarget_word_merge(args,merge_idx, text_prime, generator, target_model,
               orig_prob, orig_label, sim_predictor, text2=None, thres=None):
    len_text = len(text_prime)
    orig_token = " ".join([text_prime[merge_idx], text_prime[merge_idx + 1]])
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[merge_idx] = '<mask>'
    del mask_input[merge_idx + 1]
    mask_token = [orig_token]

    # 祝贺 在此处添加 prompt信息 用作第二次生成
    ori_LabelWord = class_names.get(thres.get('dataset_name')).get(orig_label)
    input = build_prompt_input(" ".join(mask_input), args, "no target", ori_LabelWord,text2)

    if 'roberta' not in str(args.mlm_model):
        input=[i.replace('<mask>','[MASK]') for i in input]

    synonyms, syn_probs = generator(input, mask_token, mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:merge_idx] + [synonym] + text_prime[min(merge_idx + 2, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()

    semantic_sims = similairty_calculation(
        [merge_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['merge'] * len(new_texts), thres=thres)

    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['merge_prob']
    semantic_mask = semantic_sims > thres['merge_sim']

    maskSymbol_mask = np.ones(attack_mask.shape)
    for i in range(len(maskSymbol_mask)):
        # don't insert [MASK]
        if synonyms[i] == '[MASK]' or synonyms[i] == '<mask>':
            maskSymbol_mask[i] = 0

    prob_diffs *= (attack_mask * semantic_mask * prob_mask*maskSymbol_mask)
    best_idx = np.argmin(prob_diffs)

    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            if (synonyms[i] == '[MASK]' or synonyms[i] == '<mask>'): continue
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x: x[0])

    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], semantic_sims[best_idx], \
           new_probs[best_idx].cpu().numpy(), collections