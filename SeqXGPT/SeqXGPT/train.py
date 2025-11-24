import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info
from dataloader import DataManager
from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier

import ast
import re
from typing import List, Dict



class SupervisedTrainer:
    # data: DataManager | model: ModelWiseTransformerClassifier
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label =id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='linear_en.pt'):
        # trange(N) 等同于 tqdm(range(N))：它会生成一个带有进度条的迭代器，适用于需要显示进度的循环。
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            print(f"---------------- Training epoch {epoch+1}/{self.num_train_epochs}")
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    # print(output)
                    logits = output['logits']
                    loss = output['loss']
                    # print(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            self.test()
            print('*' * 120)
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name,weights_only=False)
        self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self, content_level_eval=False):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']
                # print(f'preds: {preds.cpu().tolist()}')
                # print(f'true_labels: {labels.cpu().tolist()}')
                
                # print(f'inputs: {type(inputs["text"])}') # list
                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())
        
        # print(f'type of texts[0]: {type(texts[0])}') # str
        # with open("", 'w') as f:
        #     f.write(json.dumps(total_logits[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(texts[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(true_labels[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(pred_labels[3], ensure_ascii=False) + '\n')


        if content_level_eval:
            # content level evaluation
            print("*" * 8, "Content Level Evalation", "*" * 8)
            content_result = self.content_level_eval(texts, true_labels, pred_labels)
        else:
            # sent level evalation
            print("*" * 8, "Sentence Level Evalation", "*" * 8)
            sent_result = self.sent_level_eval(texts, true_labels, pred_labels)

        # word level evalation
        print("*" * 8, "Word Level Evalation", "*" * 8)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        true_labels_1d = true_labels.reshape(-1)
        pred_labels_1d = pred_labels.reshape(-1)
        mask = true_labels_1d != -1
        true_labels_1d = true_labels_1d[mask]
        pred_labels_1d = pred_labels_1d[mask]
        accuracy = (true_labels_1d == pred_labels_1d).astype(np.float32).mean().item()
        print("Accuracy: {:.1f}".format(accuracy*100))
        pass
    
    def content_level_eval(self, texts, true_labels, pred_labels):
        from collections import Counter

        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
        
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            print('\n\n##############################################################')
            print(text)
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)

            boundary_ix_real = []
            boundary_ix = []
            for idx, sent in enumerate(text.splitlines()):
                if idx < len(true_sent_label) and idx < len(pred_sent_label):

                    if(true_sent_label[idx] != pred_sent_label[idx]):
                        # print('==========预测错误示例=================\n')
                        # print(f'sent: {sent}')
                        # print(f'true_label: {true_sent_label[idx]}')
                        # print(f'pred_label: {pred_sent_label[idx]}')
                        # print('\n')
                        boundary_ix_real.append(idx)
                    else:
                        pass
                        # print(f'sent: {sent}')
                        # print(f'true_label: {true_sent_label[idx]}')
                        # print(f'pred_label: {pred_sent_label[idx]}')
                else:
                    # print('==========list index out of range=================\n')
                    # print(f'sent: {sent}')
                    print('\n')
            
            if(len(boundary_ix_real)!=0):
                current_group=[boundary_ix_real[0]]
                for i in range(1, len(boundary_ix_real)):
                    # 检查是否连续：当前元素 = 前一个元素 + 1
                    if boundary_ix_real[i] == boundary_ix_real[i-1] + 1:
                        current_group.append(boundary_ix_real[i])
                    else:
                        # 不连续时，保存当前分组的最大值
                        boundary_ix.append(current_group[0])
                        current_group = [boundary_ix_real[i]]  # 开始新分组
                
                # 处理最后一个分组
                boundary_ix.append(current_group[0])

                print(f'boundary_ix: {boundary_ix}')
                print(f'len(boundary_ix): {len(boundary_ix)}')

            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        true_sent_labels_cleaned = []
        for t in true_sent_labels:
            if t=='gpt2':
                t = 'llama'
            true_sent_labels_cleaned.append(t)
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels_cleaned]
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
        return result
    
    def get_sent_label(self, text, label):
        """
        按行（换行符 '\n'）切分 text，每一行作为一个“句子”。
        然后根据原始 token 级标签，找出每行最常见的标签。
        """
        # print(f'----text-----\n{text}')
        # print(f'-----labels -----\n{label}')
        # 1. 按换行分行，保留空行
        # sents = text.splitlines()
        sents = self.split_code_into_blocks(text)
        # print(f'-----sents-----\n{sents}')

        offset = 0
        sent_label = []
        for sent in sents:
            #print(f'----sent------\n{sent}')
            # 跳过空行，但仍需更新 offset
            if sent == "":
                offset += 1  # 仅跳过 '\n' 本身，占一个字符
                continue

            # 2. 在原文本中匹配本行的起止位置
            #    （从上次 offset 开始，找到本行第一次出现的位置）
            start = text.find(sent, offset)
            #print(f'----start----\n{start}')
            end = start + len(sent)
            #print(f'----end----\n{end}')
            offset = end + 1  # 末尾的 '\n' 也跳过

            # 3. 计算到本行末尾为止，总共多少 token
            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break  # 超出最大序列长度，停止

            # 4. 计算本行的 token 数量，以及本行在 label 列表中的起止索引
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            # print(f'----start_word_idx----\n{start_word_idx}')
            # print(f'----end_word_idx----\n{end_word_idx}')
            tags = label[start_word_idx:end_word_idx]
            if(tags==[]):
                print('----empty tags----')
                continue

            # 5. 选本行最常见的标签
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])

        if len(sent_label) == 0:
            print("empty sent label list")
        return sent_label
    
    def split_code_into_blocks(self, code: str) -> List[str]:
        """
        将代码分割成不同的逻辑块
        
        参数:
            code: 要分割的代码字符串
            
        返回:
            List[str]: 每个元素是一个代码块的字符串
        """
        blocks = []
        
        try:
            # 使用AST解析代码
            tree = ast.parse(code)
            
            # 处理顶级节点
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    # 函数定义块
                    block_content = ast.get_source_segment(code, node)
                    blocks.append(block_content)
                elif isinstance(node, ast.ClassDef):
                    # 类定义块
                    block_content = ast.get_source_segment(code, node)
                    blocks.append(block_content)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    # 控制流块
                    block_content = ast.get_source_segment(code, node)
                    blocks.append(block_content)
                else:
                    # 其他代码块（导入、赋值等）
                    block_content = ast.get_source_segment(code, node)
                    blocks.append(block_content)
        
        except SyntaxError:
            # 如果AST解析失败，使用基于行的简单分割
            blocks = self.split_code_by_lines(code)
        
        return blocks

    def split_code_by_lines(self, code: str) -> List[str]:
        """
        使用基于行的简单方法分割代码
        
        参数:
            code: 要分割的代码字符串
            
        返回:
            List[str]: 每个元素是一个代码块的字符串
        """
        blocks = []
        lines = code.split('\n')
        current_block = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # 检测空行作为块分隔符
            if not stripped_line and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
                continue
            
            current_block.append(line)
        
        # 添加最后一个块
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks

    
    # def split_code_into_blocks(self, text):
    #     """
    #     将代码文本按照空行分割成块，并保留所有换行符
        
    #     参数:
    #         text (str): 输入的代码文本
            
    #     返回:
    #         list: 包含代码块的列表，每个块保留原始换行符
    #     """
    #     lines = text.splitlines(True)  # 保留换行符分割为行
    #     blocks = []  # 存储所有代码块
    #     current_block = []  # 当前正在处理的代码块
        
    #     for line in lines:
    #         # 检查是否是空行（只包含空白字符和换行符）
    #         if line.strip() == '':
    #             # 如果当前块不为空，则将其加入blocks并重置
    #             if current_block:
    #                 blocks.append(''.join(current_block))
    #                 current_block = []
    #             # 空行本身也作为一个单独的块或保留在后续处理中
    #             current_block.append(line)
    #         else:
    #             # 非空行，添加到当前块
    #             current_block.append(line)
        
    #     # 添加最后一个块
    #     if current_block:
    #         blocks.append(''.join(current_block))
        
    #     return blocks

   

    # def get_sent_label(self, text, label):
    #     import nltk
    #     sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
    #     sents = sent_separator.tokenize(text)

    #     offset = 0
    #     sent_label = []
    #     for sent in sents:
    #         start = text[offset:].find(sent) + offset
    #         end = start + len(sent)
    #         offset = end
            
    #         split_sentence = self.data.split_sentence
    #         end_word_idx = len(split_sentence(text[:end]))
    #         if end_word_idx > self.seq_len:
    #             break
    #         word_num = len(split_sentence(text[start:end]))
    #         start_word_idx = end_word_idx - word_num
    #         tags = label[start_word_idx:end_word_idx]
    #         most_common_tag = self._get_most_common_tag(tags)
    #         sent_label.append(most_common_tag[0])
        
    #     if len(sent_label) == 0:
    #         print("empty sent label list")
    #     return sent_label
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter
        # print(f'----tags----\n{tags}') # block的时候，tag有空[]
        tags_cleaned = []
        for tag in tags:
            if tag==-1:
                continue
            tags_cleaned.append(tag)

        tags = [self.id2label[tag] for tag in tags_cleaned]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        # print(f'----tag counts----\n{tag_counts}')
        most_common_tag = tag_counts.most_common(1)[0]
        # print(f'----most common tag----\n{most_common_tag}')

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    # {0: 'B-gpt2', 1: 'M-gpt2', 2: 'E-gpt2', 3: 'S-gpt2', 4: 'B-gptneo', 5: 'M-gptneo', 6: 'E-gptneo', 7: 'S-gptneo', 8: 'B-gptj', 9: 'M-gptj', 10: 'E-gptj', 11: 'S-gptj', 12: 'B-llama', 13: 'M-llama', 14: 'E-llama', 15: 'S-llama', 16: 'B-gpt3re', 17: 'M-gpt3re', 18: 'E-gpt3re', 19: 'S-gpt3re', 20: 'B-human', 21: 'M-human', 22: 'E-human', 23: 'S-human'}
    # print(id2label)
    return id2label

def split_dataset(data_path, train_path, test_path, train_ratio=0.9):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    print('*'*32)
    print('The overall data sources:')
    print(file_names)
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)
    
    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    def save_dataset(fpath, data_samples):
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    print()
    print("The number of train dataset:", len(train_data))
    print("The number of test  dataset:", len(test_data))
    print('*'*32)
    pass

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=2048)

    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='./code_data_hybrid')
    # parser.add_argument('--train_path', type=str, default='./code_data_hybrid/code_data_hybrid_train.jsonl')
    # parser.add_argument('--test_path', type=str, default='./code_data_hybrid/code_data_hybrid_test.jsonl')
    parser.add_argument('--train_path', type=str, default='/data/wangmanyi/SeqXGPT/SeqXGPT/code_data_hybrid/code_data_hybrid_train.jsonl')
    parser.add_argument('--test_path', type=str, default='/data/wangmanyi/SeqXGPT/SeqXGPT/code_data_hybrid/code_data_hybrid_test.jsonl')

    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()

# python ./Seq_train/train.py --gpu=0 --split_dataset
# python ./Seq_train/train.py --gpu=0
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.split_dataset:
        print("Log INFO: split dataset...")
        split_dataset(data_path=args.data_path, train_path=args.train_path, test_path=args.test_path, train_ratio=args.train_ratio)

    # en_labels = backend_model_info.en_labels
    en_labels = {
        'gpt2': 0,
        'gptneo': 1,
        'gptj': 2,
        'llama': 3,
        'gpt3re': 4,
        # 'gpt3sum': 3,
        'human': 5
    }
    # en_labels = {'AI':0, 'human':1}

    # 为什么要把label construct成bmes标签？
    # Log INFO: id2label: {0: 'B-gpt2', 1: 'M-gpt2', 2: 'E-gpt2', 3: 'S-gpt2', 4: 'B-gptneo', 5: 'M-gptneo', 6: 'E-gptneo', 7: 'S-gptneo', 8: 'B-gptj', 9: 'M-gptj', 10: 'E-gptj', 11: 'S-gptj', 12: 'B-llama', 13: 'M-llama', 14: 'E-llama', 15: 'S-llama', 16: 'B-gpt3re', 17: 'M-gpt3re', 18: 'E-gpt3re', 19: 'S-gpt3re', 20: 'B-human', 21: 'M-human', 22: 'E-human', 23: 'S-human'}
    # Log INFO: label2id: {'B-gpt2': 0, 'M-gpt2': 1, 'E-gpt2': 2, 'S-gpt2': 3, 'B-gptneo': 4, 'M-gptneo': 5, 'E-gptneo': 6, 'S-gptneo': 7, 'B-gptj': 8, 'M-gptj': 9, 'E-gptj': 10, 'S-gptj': 11, 'B-llama': 12, 'M-llama': 13, 'E-llama': 14, 'S-llama': 15, 'B-gpt3re': 16, 'M-gpt3re': 17, 'E-gpt3re': 18, 'S-gpt3re': 19, 'B-human': 20, 'M-human': 21, 'E-human': 22, 'S-human': 23}
    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    data = DataManager(train_path=args.train_path, test_path=args.test_path, 
                       batch_size=args.batch_size, max_len=args.seq_len, 
                       human_label='human', id2label=id2label)
    
    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        if args.model == 'CNN':
            print('-' * 32 + "CNN" + '-' * 32)
            classifier = ModelWiseCNNClassifier(id2labels=id2label)
            ckpt_name = ''
        elif args.model == 'RNN':
            print('-' * 32 + "RNN" + '-' * 32)
            classifier = TransformerOnlyClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'original_text_model.ckpt'

        trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)

        if args.do_test:    
            print("Log INFO: do test...")
            saved_model = torch.load(ckpt_name,weights_only=False)
            trainer.model.load_state_dict(saved_model.state_dict())
            trainer.test(content_level_eval=args.test_content)
        else:
            print("Log INFO: do train...")
            # ckpt：中断点，保存模型权重与状态，断点重新训练
            trainer.train(ckpt_name=ckpt_name)

    """contrastive training"""
    if args.train_mode == 'contrastive_learning':
        print('-' * 32 + 'contrastive_learning' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''

        trainer = SupervisedTrainer(data, classifier, loss_criterion = 'ContrastiveLoss')
        trainer.train(ckpt_name=ckpt_name)

    """classify after contrastive"""
    if args.train_mode == 'contrastive_classify':
        print('-' * 32 + 'contrastive_classify' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''

        # trainer = SupervisedTrainer(data, classifier, train_mode='Contrastive_Classifier')
        trainer = SupervisedTrainer(data, classifier)
        trainer.train(ckpt_name=ckpt_name)
