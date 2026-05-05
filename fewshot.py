from base64 import encode

import tqdm
from openprompt.data_utils.text_classification_dataset import ToxiProcessor,ProsProcessor,CbProcessor
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import pandas as pd
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='../Model/bert_base_chinese')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--kptw_lr", default=0.05, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=str)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--manual_or_soft", default="manual", type=str)
args = parser.parse_args()

import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}


if args.dataset == "cyberbullying":
    dataset['train'] = CbProcessor().get_train_examples("datasets/TextClassification/cyberbullying/")
    dataset['test'] = CbProcessor().get_test_examples("datasets/TextClassification/cyberbullying/")
    class_labels =  CbProcessor().get_labels()
    scriptsbase = "TextClassification/cyberbullying"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 512
    batch_s = args.batch_size
else:
    raise NotImplementedError

if args.manual_or_soft == "manual":
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"./scripts/{scriptsbase}/manual_template.txt",choice=args.template_id)
else:
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)
#
if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/kpt_translate_verbalizer.{scriptformat}")

elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "trpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/adj_translate.{scriptformat}")
elif args.verbalizer == "kpt_plus":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/kpt++_verbalizer.{scriptformat}")

# (contextual) calibration
if args.verbalizer in ["kpt", "manual",'kpt_plus']:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler

        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                              decoder_max_length=3,
                                              batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="tail")

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# HP
# if args.calibration:
if args.verbalizer in ["kpt", "manual",'kpt_plus']:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from openprompt.utils.calibrate import calibrate

        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits)
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))

    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
  # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")


def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    pd.DataFrame({"alllabels":alllabels,"allpreds":allpreds}).to_csv('out_label.csv', header=0, index=False)
    print(allpreds)



    acc = accuracy_score(alllabels, allpreds)
    
    pre, recall, F1score, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
    cal_data = [acc, pre, recall, F1score]
    return cal_data




from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()




if args.verbalizer == "kpt" or args.verbalizer == "kpt_plus" :
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)


    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)


    scheduler2 = None


elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']


    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]



    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

tot_loss = 0
log_loss = 0
best_val_acc = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Total Loss: {tot_loss:.4f}")
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    cal_data = evaluate(prompt_model, validation_dataloader, desc="Valid")
    val_acc = cal_data[0]
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, Average Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
        epoch, tot_loss/len(train_dataloader), val_acc), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
data_set = evaluate(prompt_model, test_dataloader, desc="Test")
test_acc = data_set[0]
test_pre = data_set[1]
test_recall = data_set[2]
test_F1scall = data_set[3]

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"manual_or_soft {args.manual_or_soft}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"bt {args.batch_size}\t"
content_write += f"lr {args.learning_rate}\t"
content_write += f"eps {args.max_epochs}\t"
content_write += "\n"

content_write += f"Acc: {test_acc:.4f}\t"
content_write += f"Pre: {test_pre:.4f}\t"
content_write += f"Rec: {test_recall:.4f}\t"
content_write += f"F1s: {test_F1scall:.4f}\t"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os

os.remove(f"./ckpts/{this_run_unicode}.ckpt")