import argparse
import math
import re

import torch
from tqdm import tqdm
import numpy as np
#import language_check

import language_tool_python as  language_check

from bert_score import score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models.similarity_model import USE
import pandas as pd

def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string

def calculate_ppl(texts, model, tokenizer):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    with torch.no_grad():
        for text in tqdm(texts):
            text = process_string(text)
            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True))
            if len(input_ids) < 2:
                continue
            input_ids = input_ids.cuda()
            outputs = model(input_ids, labels=input_ids)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity.item()

def evaluate(refs, hypos, use,args):
    # GPT2 model for perpelexity calculation

    # USE similarity score
    print('Evaluating USE similarity:')
    sim_score = []
    for i in tqdm(range(len(refs))):
        sim_score.append(use.semantic_sim([refs[i]], [hypos[i]])[0][0])
    sim_score = np.mean(sim_score)
    print('USE sim score: ', sim_score)

    torch.cuda.empty_cache()

    tokenizer = GPT2Tokenizer.from_pretrained("./MLM_model/gpt2")
    model = GPT2LMHeadModel.from_pretrained("./MLM_model/gpt2").cuda()


    print("Evaluating references ppl ...")
    ref_ppl = calculate_ppl(refs, model, tokenizer)
    print("Reference ppl: %.4f" % ref_ppl)

    print("Evaluating hypothesis ppl ...")
    hy_ppl = calculate_ppl(hypos, model, tokenizer)
    print("Hypothesis ppl: %.4f" % hy_ppl)

    torch.cuda.empty_cache()

    #BERT SCORE:
    # with torch.no_grad():
    #     P, R, F1 = score(hypos, refs,model_type='./MLM_model/roberta-large',num_layers=17, lang='en', verbose=True)
    # print('BERT score: ', F1.mean())



    # # evalute number of grammar errors
    print('Evaluating number of grammar error:')
    tool = language_check.LanguageTool('en-US')
    grammar_diffs = []
    for i in tqdm(range(len(refs))):
        grammar_ref = len(tool.check(process_string(refs[i])))
        grammar_hypo = len(tool.check(process_string(hypos[i])))
        grammar_diffs.append(grammar_hypo - grammar_ref)
    gramar_err = np.mean(grammar_diffs)
    print("number of grammar difference: ", gramar_err)
    print('\n')
    #gramar_err=0

    #return ref_ppl, hy_ppl, sim_score, gramar_err
    #return ref_ppl, hy_ppl, F1.mean(), sim_score, gramar_err
    return ref_ppl, hy_ppl, 0, sim_score, gramar_err


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eval_file",
                            type=str,
                            required=False,
                            help="adversarial output file for evaluation.")
    argparser.add_argument("--USE_cache_path",
                            type=str,
                            default='./tmp',
                            help="Path to the USE encoder cache.")
    args = argparser.parse_args()
    use = USE(args.USE_cache_path)

    logPath = r"E:\code\textcode\MyTextAttack\Textfooler-ag-log.csv"
    log = pd.read_csv(logPath, header=0, encoding="UTF-8")
    refs = [line.replace('[[', "").replace(']]', "") for line in list(log['original_text'])]
    hypos = [line.replace('[[', "").replace(']]', "") for line in list(log['perturbed_text'])]

    # print("Evaluating %s", args.eval_file)
    # with open(args.eval_file, 'r', encoding='utf8') as f:
    #     for line in f:
    #         if "orig sent" in line:
    #             refs.append(line.split('\t')[1].strip())
    #         if "adv sent" in line:
    #             hypos.append(line.split('\t')[1].strip())
    assert len(refs) == len(hypos)
    ref_ppl, hy_ppl, F1, sim_score, gramar_err=evaluate(refs, hypos, use,args)
    print(ref_ppl, hy_ppl, F1, sim_score, gramar_err)
    