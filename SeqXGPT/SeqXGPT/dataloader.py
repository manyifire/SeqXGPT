import numpy as np
import os
import random
import torch
import json
import pandas as pd
import pickle

from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize

import ast
import re
from typing import List, Dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataManager:

    def __init__(self, train_path, test_path, batch_size, max_len, human_label, id2label, word_pad_idx=0, label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        data = dict()

        if train_path:
            # {'features': [], 'prompt_len': [], 'label_int': [], 'text': []}
            train_dict = self.initialize_dataset(train_path)
            data["train"] = Dataset.from_dict(train_dict)
        
        if test_path:
            test_dict = self.initialize_dataset(test_path)
            data["test"] = Dataset.from_dict(test_dict)
        
        datasets = DatasetDict(data)
        #print(datasets["train"][0])
        if train_path:
            self.train_dataloader = self.get_train_dataloader(datasets["train"])
        if test_path:
            self.test_dataloader = self.get_eval_dataloader(datasets["test"])

    def initialize_dataset(self, data_path, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        print(f'{processed_data_filename}')
        # processed_data_path = os.path.join(save_dir, processed_data_filename)

        # if os.path.exists(processed_data_path):
        #     log_info = '*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4
        #     print('*' * len(log_info))
        #     print(log_info)
        #     print('*' * len(log_info))
        #     with open(processed_data_path, 'rb') as f:
        #         samples_dict = pickle.load(f)
        #     return samples_dict

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'features': [], 'prompt_len': [], 'label': [], 'text': [],'human_part': [], 'machine_part': [], 'prompt_pattern': [], 'labeled_statements': []}
        # print(len(samples)) #16113
        # print(type(samples[0])) #dict
        for item in tqdm(samples):
            if item['machine_part'] is None or len(item['machine_part']) == 0:
                continue
            text = item['text']
            label = item['label']
            prompt_len = item['prompt_len']
            # prompt_len改为了原数据集的boundary_ix，其余不变
            # prompt_len = 0

            # if label in ['gptj', 'gpt2', 'llama', 'gpt3re']:
            #     continue
            # if label == 'gpt3sum':
            #     label = 'gpt3re'
            # if label == 'gpt3re':
            #     continue

            label_int = item['label_int']
            begin_idx_list = item['begin_idx_list']
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
            max_begin_idx = np.max(begin_idx_list)
            # Truncate all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
            # Get the length of all vectors and take the minimum
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue
            ll_tokens_list = np.array(ll_tokens_list)
            # ll_tokens_list = normalize(ll_tokens_list, norm='l1')
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()

            samples_dict['features'].append(ll_tokens_list)
            samples_dict['prompt_len'].append(prompt_len)
            samples_dict['label'].append(label)
            samples_dict['text'].append(text)
            samples_dict['human_part'].append(item['human_part'])
            samples_dict['machine_part'].append(item['machine_part'])
            samples_dict['prompt_pattern'].append(item['prompt_pattern'])
            samples_dict['labeled_statements'].append(item['labeled_statements'])
        
        # with open(processed_data_path, 'wb') as f:
        #     pickle.dump(samples_dict, f)

        return samples_dict


    def get_train_dataloader(self, dataset):
        # print(type(dataset)) # dataset
        # print(np.array(dataset).shape) # (16113,)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)

    def get_eval_dataloader(self, dataset):
        
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    # data collator for block-level classification
    def data_collator(self, samples):
        # samples: {'features': [], 'prompt_len': [], 'label': [], 'text': []}
        # batch: {'features': [], 'labels': [], 'text': []}
        # labels是一个list，为每个token打上标记
        batch = {}
        #print(f'samples ------- {np.array(samples).shape}') # (32,)
        features = [sample['features'] for sample in samples]
        #print(type(features))
        
        prompt_len = [sample['prompt_len'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]

        features, masks = self.process_and_convert_to_tensor(features)
        # pad_masks = ~masks * -1
        pad_masks = (1 - masks) * self.label_pad_idx

        for idx, data in enumerate(samples):
            lines_with_nl = self.split_code_into_blocks(data['text'])
            if data['prompt_pattern'] == "H_M":
                # p_len = self.idx_after_x_newlines(data['text'], data['prompt_len'][0])
                p_len = data['text'].find(data['human_part'][-1])+len(data['human_part'][-1])
                prefix_len = len(self.split_sentence(data['text'][:p_len]))
                if prefix_len > self.max_len:
                    prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                    masks[idx][:] = prefix_ids[:]
                    continue
                total_len = len(self.split_sentence(data['text']))
                
                if prefix_len > 0:
                    prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                    masks[idx][:prefix_len] = prefix_ids[:]
                if total_len - prefix_len > 0:
                    if total_len > self.max_len:
                        # print(f'way 1 -------- {self.max_len - prefix_len}')
                        human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, data['label'])
                    else:
                        # print(f'way 2 --------------{total_len - prefix_len}')
                        human_ids = self.sequence_labels_to_ids(total_len - prefix_len, data['label'])
                    
                    # 把导致问题的样本信息都打印出来
                    if human_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                    masks[idx][prefix_len:total_len] = human_ids[:]
                masks[idx] += pad_masks[idx]

            if data['prompt_pattern'] == "M_H":
                first_x = lines_with_nl[:data['prompt_len'][0]]
                # p_len =  len("".join(first_x))
                p_len = data['text'].find(data['human_part'][0])
                prefix_len = len(self.split_sentence(text[idx][:p_len]))
                if prefix_len > self.max_len:
                    prefix_ids = self.sequence_labels_to_ids(self.max_len, data['label'])
                    masks[idx][:] = prefix_ids[:]
                    continue
                total_len = len(self.split_sentence(data['text']))
                
                if prefix_len > 0:
                    prefix_ids = self.sequence_labels_to_ids(prefix_len, data['label'])
                    masks[idx][:prefix_len] = prefix_ids[:]
                if total_len - prefix_len > 0:
                    if total_len > self.max_len:
                        human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, self.human_label)
                    else:
                        human_ids = self.sequence_labels_to_ids(total_len - prefix_len, self.human_label)
                    
                    # 把导致问题的样本信息都打印出来
                    if human_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                    masks[idx][prefix_len:total_len] = human_ids[:]
                masks[idx] += pad_masks[idx]

            if data['prompt_pattern'] == "H_M_H":
                human1 = lines_with_nl[:data['prompt_len'][0]]
                human2 = lines_with_nl[data['prompt_len'][1]:]
                machine_len = lines_with_nl[data['prompt_len'][0]:data['prompt_len'][1]]
                # human_len1 = len("".join(human1))
                human_len1 = data['text'].find(data['machine_part'][0])+len(data['human_part'][-1])
                # human_len2 = len("".join(human2))
                human_len2 = len(data['text'])-data['text'].find(data['machine_part'][-1])
                # machine_len = len("".join(machine_len))
                machine_len = data['text'].find(data['machine_part'][0]) + len(data['machine_part'][0]) - human_len1
                machine_len = len(self.split_sentence(data['text'][human_len1:human_len1+machine_len]))
                human_len2 = len(self.split_sentence(data['text'][human_len1+machine_len:]))

                prefix_len = len(self.split_sentence(data['text'][:human_len1]))
                if prefix_len > self.max_len:
                    prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                    masks[idx][:] = prefix_ids[:]
                    continue
                total_len = len(self.split_sentence(data['text']))
                
                if prefix_len > 0:
                    prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                    masks[idx][:prefix_len] = prefix_ids[:]

                if total_len - prefix_len - human_len2 > 0:
                    if total_len > self.max_len:
                        m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-human_len2, data['label'])
                        masks[idx][prefix_len:self.max_len - human_len2] = m_ids[:]
                    else:
                        m_ids = self.sequence_labels_to_ids(total_len - prefix_len-human_len2, data['label'])
                        masks[idx][prefix_len:total_len-human_len2] = m_ids[:]

                    # 把导致问题的样本信息都打印出来
                    if m_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                if total_len - prefix_len-machine_len > 0:
                    if total_len > self.max_len:
                        m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-machine_len, self.human_label)
                    else:
                        m_ids = self.sequence_labels_to_ids(total_len - prefix_len-machine_len, self.human_label)
                    
                    # 把导致问题的样本信息都打印出来
                    if m_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                    masks[idx][prefix_len+machine_len:total_len] = m_ids[:]

                masks[idx] += pad_masks[idx]

            if data['prompt_pattern'] == "M_H_M":
                human1 = lines_with_nl[:data['prompt_len'][0]]
                human2 = lines_with_nl[data['prompt_len'][1]:]
                machine_len = lines_with_nl[data['prompt_len'][0]:data['prompt_len'][1]]
                # human_len1 = len("".join(human1))
                # human_len2 = len("".join(human2))
                # machine_len = len("".join(machine_len))
                human_len1 = data['text'].find(data['machine_part'][0])+len(data['human_part'][-1])
                human_len2 = len(data['text'])-data['text'].find(data['machine_part'][-1])
                machine_len = data['text'].find(data['machine_part'][0]) + len(data['machine_part'][0]) - human_len1
                machine_len = len(self.split_sentence(data['text'][human_len1:human_len1+machine_len]))
                human_len2 = len(self.split_sentence(data['text'][human_len1+machine_len:]))

                prefix_len = len(self.split_sentence(data['text'][:human_len1]))
                if prefix_len > self.max_len:
                    prefix_ids = self.sequence_labels_to_ids(self.max_len, data['label'])
                    masks[idx][:] = prefix_ids[:]
                    continue
                total_len = len(self.split_sentence(data['text']))
                
                if prefix_len > 0:
                    prefix_ids = self.sequence_labels_to_ids(prefix_len, data['label'])
                    masks[idx][:prefix_len] = prefix_ids[:]

                if total_len - prefix_len - human_len2 > 0:
                    if total_len > self.max_len:
                        m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-human_len2, self.human_label)
                        masks[idx][prefix_len:self.max_len - human_len2] = m_ids[:]
                    else:
                        m_ids = self.sequence_labels_to_ids(total_len - prefix_len-human_len2, self.human_label)
                        masks[idx][prefix_len:total_len - human_len2] = m_ids[:]

                    # 把导致问题的样本信息都打印出来
                    if m_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                if total_len - prefix_len-machine_len > 0:
                    if total_len > self.max_len:
                        m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-machine_len, data['label'])
                    else:
                        m_ids = self.sequence_labels_to_ids(total_len - prefix_len-machine_len, data['label'])
                    
                    # 把导致问题的样本信息都打印出来
                    if m_ids is None:
                        # 把导致问题的样本信息都打印出来
                        print(f"[Error] human_ids is None at idx={idx}")
                        print(f"  text sample: {data['text']!r}")
                        print(f"  label[idx]: {data['label']!r}")
                        print(f"  prefix_len={prefix_len}, total_len={total_len}")
                        # 也可以直接抛异常，结束并定位
                        raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

                    masks[idx][prefix_len+machine_len:total_len] = m_ids[:]

                masks[idx] += pad_masks[idx]

        batch['features'] = features
        batch['labels'] = masks
        batch['text'] = text

        return batch
    

    def split_code_into_blocks(self, text):
        """
        将代码文本按照空行分割成块，并保留所有换行符
        
        参数:
            text (str): 输入的代码文本
            
        返回:
            list: 包含代码块的列表，每个块保留原始换行符
        """
        lines = text.splitlines(True)  # 保留换行符分割为行
        blocks = []  # 存储所有代码块
        current_block = []  # 当前正在处理的代码块
        
        for line in lines:
            # 检查是否是空行（只包含空白字符和换行符）
            if line.strip() == '':
                # 如果当前块不为空，则将其加入blocks并重置
                if current_block:
                    blocks.append(''.join(current_block))
                    current_block = []
                # 空行本身也作为一个单独的块或保留在后续处理中
                current_block.append(line)
            else:
                # 非空行，添加到当前块
                current_block.append(line)
        
        # 添加最后一个块
        if current_block:
            blocks.append(''.join(current_block))
        
        return blocks

     
    # def data_collator(self, samples):
    #     # samples: {'features': [], 'prompt_len': [], 'label': [], 'text': []}
    #     # batch: {'features': [], 'labels': [], 'text': []}
    #     # labels是一个list，为每个token打上标记
    #     batch = {}
    #     #print(f'samples ------- {np.array(samples).shape}') # (32,)
    #     features = [sample['features'] for sample in samples]
    #     #print(type(features))
        
    #     prompt_len = [sample['prompt_len'] for sample in samples]
    #     text = [sample['text'] for sample in samples]
    #     label = [sample['label'] for sample in samples]

    #     features, masks = self.process_and_convert_to_tensor(features)
    #     # pad_masks = ~masks * -1
    #     pad_masks = (1 - masks) * self.label_pad_idx

    #     for idx, data in enumerate(samples):
    #         lines_with_nl = data['text'].splitlines(True)
    #         if data['prompt_pattern'] == "H_M":
    #             # p_len = self.idx_after_x_newlines(data['text'], data['prompt_len'][0])
    #             p_len = data['text'].find(data['human_part'][-1])+len(data['human_part'][-1])
    #             prefix_len = len(self.split_sentence(data['text'][:p_len]))
    #             if prefix_len > self.max_len:
    #                 prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
    #                 masks[idx][:] = prefix_ids[:]
    #                 continue
    #             total_len = len(self.split_sentence(data['text']))
                
    #             if prefix_len > 0:
    #                 prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
    #                 masks[idx][:prefix_len] = prefix_ids[:]
    #             if total_len - prefix_len > 0:
    #                 if total_len > self.max_len:
    #                     # print(f'way 1 -------- {self.max_len - prefix_len}')
    #                     human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, data['label'])
    #                 else:
    #                     # print(f'way 2 --------------{total_len - prefix_len}')
    #                     human_ids = self.sequence_labels_to_ids(total_len - prefix_len, data['label'])
                    
    #                 # 把导致问题的样本信息都打印出来
    #                 if human_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #                 masks[idx][prefix_len:total_len] = human_ids[:]
    #             masks[idx] += pad_masks[idx]

    #         if data['prompt_pattern'] == "M_H":
    #             first_x = lines_with_nl[:data['prompt_len'][0]]
    #             # p_len =  len("".join(first_x))
    #             p_len = data['text'].find(data['human_part'][0])
    #             prefix_len = len(self.split_sentence(text[idx][:p_len]))
    #             if prefix_len > self.max_len:
    #                 prefix_ids = self.sequence_labels_to_ids(self.max_len, data['label'])
    #                 masks[idx][:] = prefix_ids[:]
    #                 continue
    #             total_len = len(self.split_sentence(data['text']))
                
    #             if prefix_len > 0:
    #                 prefix_ids = self.sequence_labels_to_ids(prefix_len, data['label'])
    #                 masks[idx][:prefix_len] = prefix_ids[:]
    #             if total_len - prefix_len > 0:
    #                 if total_len > self.max_len:
    #                     human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, self.human_label)
    #                 else:
    #                     human_ids = self.sequence_labels_to_ids(total_len - prefix_len, self.human_label)
                    
    #                 # 把导致问题的样本信息都打印出来
    #                 if human_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #                 masks[idx][prefix_len:total_len] = human_ids[:]
    #             masks[idx] += pad_masks[idx]

    #         if data['prompt_pattern'] == "H_M_H":
    #             human1 = lines_with_nl[:data['prompt_len'][0]]
    #             human2 = lines_with_nl[data['prompt_len'][1]:]
    #             machine_len = lines_with_nl[data['prompt_len'][0]:data['prompt_len'][1]]
    #             # human_len1 = len("".join(human1))
    #             human_len1 = data['text'].find(data['machine_part'][0])+len(data['human_part'][-1])
    #             # human_len2 = len("".join(human2))
    #             human_len2 = len(data['text'])-data['text'].find(data['machine_part'][-1])
    #             # machine_len = len("".join(machine_len))
    #             machine_len = data['text'].find(data['machine_part'][0]) + len(data['machine_part'][0]) - human_len1
    #             machine_len = len(self.split_sentence(data['text'][human_len1:human_len1+machine_len]))
    #             human_len2 = len(self.split_sentence(data['text'][human_len1+machine_len:]))

    #             prefix_len = len(self.split_sentence(data['text'][:human_len1]))
    #             if prefix_len > self.max_len:
    #                 prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
    #                 masks[idx][:] = prefix_ids[:]
    #                 continue
    #             total_len = len(self.split_sentence(data['text']))
                
    #             if prefix_len > 0:
    #                 prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
    #                 masks[idx][:prefix_len] = prefix_ids[:]

    #             if total_len - prefix_len - human_len2 > 0:
    #                 if total_len > self.max_len:
    #                     m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-human_len2, data['label'])
    #                     masks[idx][prefix_len:self.max_len - human_len2] = m_ids[:]
    #                 else:
    #                     m_ids = self.sequence_labels_to_ids(total_len - prefix_len-human_len2, data['label'])
    #                     masks[idx][prefix_len:total_len-human_len2] = m_ids[:]

    #                 # 把导致问题的样本信息都打印出来
    #                 if m_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #             if total_len - prefix_len-machine_len > 0:
    #                 if total_len > self.max_len:
    #                     m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-machine_len, self.human_label)
    #                 else:
    #                     m_ids = self.sequence_labels_to_ids(total_len - prefix_len-machine_len, self.human_label)
                    
    #                 # 把导致问题的样本信息都打印出来
    #                 if m_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #                 masks[idx][prefix_len+machine_len:total_len] = m_ids[:]

    #             masks[idx] += pad_masks[idx]

    #         if data['prompt_pattern'] == "M_H_M":
    #             human1 = lines_with_nl[:data['prompt_len'][0]]
    #             human2 = lines_with_nl[data['prompt_len'][1]:]
    #             machine_len = lines_with_nl[data['prompt_len'][0]:data['prompt_len'][1]]
    #             # human_len1 = len("".join(human1))
    #             # human_len2 = len("".join(human2))
    #             # machine_len = len("".join(machine_len))
    #             human_len1 = data['text'].find(data['machine_part'][0])+len(data['human_part'][-1])
    #             human_len2 = len(data['text'])-data['text'].find(data['machine_part'][-1])
    #             machine_len = data['text'].find(data['machine_part'][0]) + len(data['machine_part'][0]) - human_len1
    #             machine_len = len(self.split_sentence(data['text'][human_len1:human_len1+machine_len]))
    #             human_len2 = len(self.split_sentence(data['text'][human_len1+machine_len:]))

    #             prefix_len = len(self.split_sentence(data['text'][:human_len1]))
    #             if prefix_len > self.max_len:
    #                 prefix_ids = self.sequence_labels_to_ids(self.max_len, data['label'])
    #                 masks[idx][:] = prefix_ids[:]
    #                 continue
    #             total_len = len(self.split_sentence(data['text']))
                
    #             if prefix_len > 0:
    #                 prefix_ids = self.sequence_labels_to_ids(prefix_len, data['label'])
    #                 masks[idx][:prefix_len] = prefix_ids[:]

    #             if total_len - prefix_len - human_len2 > 0:
    #                 if total_len > self.max_len:
    #                     m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-human_len2, self.human_label)
    #                     masks[idx][prefix_len:self.max_len - human_len2] = m_ids[:]
    #                 else:
    #                     m_ids = self.sequence_labels_to_ids(total_len - prefix_len-human_len2, self.human_label)
    #                     masks[idx][prefix_len:total_len - human_len2] = m_ids[:]

    #                 # 把导致问题的样本信息都打印出来
    #                 if m_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #             if total_len - prefix_len-machine_len > 0:
    #                 if total_len > self.max_len:
    #                     m_ids = self.sequence_labels_to_ids(self.max_len - prefix_len-machine_len, data['label'])
    #                 else:
    #                     m_ids = self.sequence_labels_to_ids(total_len - prefix_len-machine_len, data['label'])
                    
    #                 # 把导致问题的样本信息都打印出来
    #                 if m_ids is None:
    #                     # 把导致问题的样本信息都打印出来
    #                     print(f"[Error] human_ids is None at idx={idx}")
    #                     print(f"  text sample: {data['text']!r}")
    #                     print(f"  label[idx]: {data['label']!r}")
    #                     print(f"  prefix_len={prefix_len}, total_len={total_len}")
    #                     # 也可以直接抛异常，结束并定位
    #                     raise ValueError(f"sequence_labels_to_ids returned None at sample {idx}")

    #                 masks[idx][prefix_len+machine_len:total_len] = m_ids[:]

    #             masks[idx] += pad_masks[idx]

    #     batch['features'] = features
    #     batch['labels'] = masks
    #     batch['text'] = text

    #     return batch

    
    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]
        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]

        # masks = [[False] * min(len(seq)+1, max_len) + [True] * (max_len - min(len(seq)+1, max_len)) for seq in data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)

        return tensor_data, tensor_mask


    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        pattern = re.compile(r'\S+|\s')
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    def _split_cn_sentence(self, sentence, use_sp=False):
        words = list(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words


    def split_sentence(self, sentence, use_sp=False, cn_percent=0.2):
        total_char_count = len(sentence)
        total_char_count += 1 if total_char_count == 0 else 0
        chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
        if chinese_char_count / total_char_count > cn_percent:
            return self._split_cn_sentence(sentence, use_sp)
        else:
            return self._split_en_sentence(sentence, use_sp)
        
    def idx_after_x_newlines(self,s: str, x: int) -> int:     
        lines_with_nl = s.splitlines(True)
        first_x = lines_with_nl[:x]
        return len("".join(first_x))

