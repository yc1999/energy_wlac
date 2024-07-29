# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import random
import math
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

def word_align(src_tokens, src_dict, tgt_dict, bi_dict, stop_word_list, pad_idx):
    """
    1. 筛选出停用词，过滤一下src_tokens；
    2. 进行映射，将src_tokens中的单词映射到对应的目标端；
    3. pad到一起传进去；
    """
    src_tokens = src_tokens.tolist()
    src_tokens = [ [src_dict.symbols[t] for t in sent] for sent in src_tokens ]    # convert id to symbol

    align_sents = []
    align_positions = []
    # 遍历每一个src_sent
    for src_sent in src_tokens:
        align_sent = []
        align_position = []

        # 遍历每一个src_word
        for idx, src_token in enumerate(src_sent):
            if src_token not in ["<s>", "</s>", "<unk>", "<pad>"] \
            and src_token in bi_dict:
                align_token = list( bi_dict[src_token].keys() )[0]
                # align_token = random.choice( list(bi_dict[src_token].keys()) )
                if align_token in tgt_dict.symbols:  #TODO: need sampling here
                    # 应该是一定会在bi_dict里面的，但是不一定在tgt_dict里面的
                    align_sent.append( tgt_dict.index(align_token) )
                    position_idx = idx + 2
                    align_position.append( position_idx )
            #TODO: and src_token not in stop_word_list \
        
        align_sents.append( torch.LongTensor(align_sent) )
        align_positions.append( torch.LongTensor(align_position) )

    # 转换成tensor，并使用pad_sequence进行padding
    align_sents = pad_sequence(align_sents, batch_first=True, padding_value=pad_idx)
    align_positions = pad_sequence(align_positions, batch_first=True, padding_value=pad_idx)    # <pad>的position idx一般设置为0

    return align_sents, align_positions

def gen_bi_context_span_from_sent(tgt_len, pos):
    max_span_length = int( math.ceil( tgt_len * 0.25 ) )
    left_shift = min( random.randint(0, pos), max_span_length )
    right_shift = min( random.randint(0, tgt_len -2 - pos), max_span_length )
    # lhs = max(1, pos - left_shift)  # TODO: change this to 0
    # rhs = min(pos + right_shift, tgt_len)
    lhs = pos - left_shift
    rhs = pos + right_shift
    return lhs, rhs

def gen_bi_context_mask(tgt_len, pos):
    if tgt_len <= 1:
        # Ideally, this should not happen. Because the input sentence should have at least 2 tokens.
        print("tgt_len <= 1".center(100, "*"))
        assert False
        return [0] * tgt_len
    span = gen_bi_context_span_from_sent(tgt_len, pos)
    masked = [0] * tgt_len
    # if span[1] == tgt_len-2:
        # print("here")
        # pass
    for word_id in range(span[0], span[1]+1):   # span[0] is inclusive, span[1] is inclusive
        masked[word_id] = 1
    return masked

def gen_bi_context_mask_span(tgt_len, pos):
    if tgt_len <= 1:
        # Ideally, this should not happen. Because the input sentence should have at least 2 tokens.
        print("tgt_len <= 1".center(100, "*"))
        assert False
        return [0] * tgt_len
    span = gen_bi_context_span_from_sent(tgt_len, pos)
    masked = [0] * tgt_len
    # if span[1] == tgt_len-2:
        # print("here")
        # pass
    for word_id in range(span[0], span[1]+1):   # span[0] is inclusive, span[1] is inclusive
        masked[word_id] = 1
    return masked, span

def gen_suffix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(0, pos+1):
        masked[word_id] = 1
    return masked

def gen_prefix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(pos, tgt_len):
        masked[word_id] = 1
    return masked

def gen_zero_context_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [1] * tgt_len
    masked[-1] = 0  # only </s> is not masked
    return masked   

def get_pos_with_mask(tgt_mask, cur_len, pad_idx):
    n = 1
    tgt_len = len(tgt_mask)
    row_pos = [0] * tgt_len
    for j in range(cur_len):
        if tgt_mask[j]:
            if j > 0 and tgt_mask[j - 1]:
                row_pos[j] = row_pos[j-1]
            else:
                row_pos[j] = n
                n += 1
        else:
            row_pos[j] = n
            n += 1
    row_pos = [ num+pad_idx for num in row_pos ]
    return row_pos

def add_word_suggestion_sample_v2(input_matrix, sugst_type, mask_idx, pad_idx, tgt_dict):
    batch_size = input_matrix.size(0)
    # This includes the eos token
    target_lengths = torch.sum( input_matrix.ne(pad_idx).int(), dim=-1 )
    min_tgt_len = torch.min(target_lengths).item()
    max_tgt_len = input_matrix.size(1)
    
    input_matrix_new = input_matrix.clone().detach()
    anchors = []
    tgt_mask_batch = []
    tgt_pos_batch = []

    len_limit = None
    if len(tgt_dict.symbols) == 50001:
        len_limit = 4
    elif len(tgt_dict.symbols) == 60001:
        len_limit = 2
    else:
        raise ValueError("len of tgt_dict is invalid!!! {}".format(len(tgt_dict.symbols)))

    if min_tgt_len > 1:
        for i in range(batch_size):
            cur_tgt_len = target_lengths[i].item()
            
            if sugst_type == "bi_context":

                anchor = random.randint(0, cur_tgt_len-2)

                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < len_limit and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, cur_tgt_len-2)
                    resample_cnt += 1

                anchors.append(anchor)

                tgt_mask = gen_bi_context_mask(cur_tgt_len, anchor)

                ###########################################################
                # Assuming there exits a freq list and the experiment results show that the freq list is not helpful.
                # cur_freq_list = [ tgt_dict.count[x] if x > 3 else 1e8 for x in input_matrix[i] ]
                # cur_freq_list = cur_freq_list[ : cur_tgt_len-1 ]
                # # 进行normalize，采用 x^2 * e^{-x} + 1函数
                # cur_freq_list = [ (math.pow(num, 2) * math.exp(-num * 0.001))  + 1 for num in cur_freq_list ]
                # # normalize cur_freq_list
                # cur_freq_list = [ num / sum(cur_freq_list) for num in cur_freq_list ]
                # # use cur_freq_list as probability to sample
                # anchor = np.random.choice(cur_tgt_len-1, 1, p=cur_freq_list)[0]
                
                # anchors.append(anchor)
                # tgt_mask = gen_bi_context_mask(cur_tgt_len, anchor)
                ###########################################################

            elif sugst_type == "prefix":

                pos = random.randint(1, cur_tgt_len-1)
                tgt_mask = gen_prefix_mask(cur_tgt_len, pos)

                anchor = random.randint(pos, cur_tgt_len-1)
                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < len_limit and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(pos, cur_tgt_len-1)
                    resample_cnt += 1
                anchors.append(anchor)

            elif sugst_type == "suffix":
                pos = random.randint(0, cur_tgt_len-2)
                tgt_mask = gen_suffix_mask(cur_tgt_len, pos)

                anchor = random.randint(0, pos)   # 由于gen_suffix_mask()用开区间，所以要减一
                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < len_limit and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, pos)
                    resample_cnt += 1
                anchors.append(anchor)
            elif sugst_type == "zero_context":
                anchor = random.randint(0, cur_tgt_len-2)

                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < len_limit and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, cur_tgt_len-2)
                    resample_cnt += 1

                anchors.append(anchor)
                tgt_mask = gen_zero_context_mask(cur_tgt_len, anchor)
            else:
                raise NotImplementedError

            if sugst_type == "bi_context": 
                new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
                tgt_mask_batch.append(new_tgt_mask)
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            elif sugst_type == "prefix":
                new_tgt_mask = tgt_mask + [1] * (cur_tgt_len - len(tgt_mask))
                new_tgt_mask = new_tgt_mask + [0] * (max_tgt_len - len(new_tgt_mask))
                tgt_mask_batch.append(new_tgt_mask)
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            elif sugst_type == "suffix":
                new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
                tgt_mask_batch.append(new_tgt_mask)
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            elif sugst_type == "zero_context":
                new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
                tgt_mask_batch.append(new_tgt_mask)
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            else:
                raise NotImplementedError

        for i in range(batch_size):
            for j in range(max_tgt_len):
                if tgt_mask_batch[i][j]:
                    input_matrix_new[i, j] = mask_idx
        tgt_pos_batch = input_matrix_new.new(tgt_pos_batch)
        anchors = input_matrix_new.new(anchors)
    
    return input_matrix_new, tgt_pos_batch, anchors

def add_word_suggestion_sample_v3(input_matrix, sugst_type, mask_idx, pad_idx, tgt_dict, target, src_tokens):
    batch_size = input_matrix.size(0)
    # This includes the eos token
    target_lengths = torch.sum( input_matrix.ne(pad_idx).int(), dim=-1 )
    min_tgt_len = torch.min(target_lengths).item()
    max_tgt_len = input_matrix.size(1)
    
    input_matrix_new = input_matrix.clone().detach()
    anchors = []
    tgt_mask_batch = []
    tgt_pos_batch = []
    stack_input_matrix = []
    stack_target = []
    stack_src_tokens = []
    
    if min_tgt_len > 1:
        for i in range(batch_size):
            cur_tgt_len = target_lengths[i].item()
            
            if sugst_type == "bi_context":

                anchor = random.randint(0, cur_tgt_len-2)

                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < 4 and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, cur_tgt_len-2)
                    resample_cnt += 1

                anchors.append(anchor)

                tgt_mask, span = gen_bi_context_mask_span(cur_tgt_len, anchor)
                
                new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
                tgt_mask_batch.append(new_tgt_mask)
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) ) 
                stack_input_matrix.append( input_matrix[i] )
                stack_target.append( target[i] )
                stack_src_tokens.append( src_tokens[i] )

                # 得到tgt_mask之后，在里面随机采其他的anchor，注意anchor的长度要好好控制
                # 1. 得到mask的长度
                span_len = span[1] - span[0] + 1
                sample_span = list( range(span[0], span[1]+1) )
                sample_span.remove(anchor)
                # 2. 在这个span里面再找一些其他的词语
                # 多采2个example
                sample_size = min(span_len-1, 20)
                for _ in range(sample_size):
                    anchor = random.sample( sample_span, k=1 )[0]
                    resample_cnt = 0
                    while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < 4 and resample_cnt < 3:
                        prob = random.random()
                        if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                            anchor = random.sample( sample_span, k=1 )[0]
                        resample_cnt += 1                
                    sample_span.remove(anchor)
                    anchors.append(anchor)
                    
                    new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
                    tgt_mask_batch.append(new_tgt_mask)
                    tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )  
                    stack_input_matrix.append( input_matrix[i] )
                    stack_target.append( target[i] )
                    stack_src_tokens.append( src_tokens[i] )


                ###########################################################
                # Assuming there exits a freq list and the experiment results show that the freq list is not helpful.
                # cur_freq_list = [ tgt_dict.count[x] if x > 3 else 1e8 for x in input_matrix[i] ]
                # cur_freq_list = cur_freq_list[ : cur_tgt_len-1 ]
                # # 进行normalize，采用 x^2 * e^{-x} + 1函数
                # cur_freq_list = [ (math.pow(num, 2) * math.exp(-num * 0.001))  + 1 for num in cur_freq_list ]
                # # normalize cur_freq_list
                # cur_freq_list = [ num / sum(cur_freq_list) for num in cur_freq_list ]
                # # use cur_freq_list as probability to sample
                # anchor = np.random.choice(cur_tgt_len-1, 1, p=cur_freq_list)[0]
                
                # anchors.append(anchor)
                # tgt_mask = gen_bi_context_mask(cur_tgt_len, anchor)
                ###########################################################

            elif sugst_type == "prefix":

                pos = random.randint(1, cur_tgt_len-1)
                tgt_mask = gen_prefix_mask(cur_tgt_len, pos)

                anchor = random.randint(pos, cur_tgt_len-1)
                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < 4 and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(pos, cur_tgt_len-1)
                    resample_cnt += 1
                anchors.append(anchor)

            elif sugst_type == "suffix":
                pos = random.randint(0, cur_tgt_len-2)
                tgt_mask = gen_suffix_mask(cur_tgt_len, pos)

                anchor = random.randint(0, pos)   # 由于gen_suffix_mask()用开区间，所以要减一
                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < 4 and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, pos)
                    resample_cnt += 1
                anchors.append(anchor)
            elif sugst_type == "zero_context":
                anchor = random.randint(0, cur_tgt_len-2)

                resample_cnt = 0
                while len(tgt_dict.symbols[ input_matrix[i][anchor] ]) < 4 and resample_cnt < 3:
                    prob = random.random()
                    if prob < 0.8:  #TODO: 将这个probability继续地调大一些，这样说不定可以选到更多长的单词。
                        anchor = random.randint(0, cur_tgt_len-2)
                    resample_cnt += 1

                anchors.append(anchor)
                tgt_mask = gen_zero_context_mask(cur_tgt_len, anchor)
            else:
                raise NotImplementedError

            # if sugst_type == "bi_context": 
            #     new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
            #     tgt_mask_batch.append(new_tgt_mask)
            #     tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            # elif sugst_type == "prefix":
            #     new_tgt_mask = tgt_mask + [1] * (cur_tgt_len - len(tgt_mask))
            #     new_tgt_mask = new_tgt_mask + [0] * (max_tgt_len - len(new_tgt_mask))
            #     tgt_mask_batch.append(new_tgt_mask)
            #     tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            # elif sugst_type == "suffix":
            #     new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
            #     tgt_mask_batch.append(new_tgt_mask)
            #     tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            # elif sugst_type == "zero_context":
            #     new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - cur_tgt_len, 0 )
            #     tgt_mask_batch.append(new_tgt_mask)
            #     tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_tgt_len, pad_idx) )
            # else:
            #     raise NotImplementedError
        stack_input_matrix = torch.stack(stack_input_matrix, dim=0)
        stack_target = torch.stack(stack_target, dim=0)
        stack_src_tokens = torch.stack(stack_src_tokens, dim=0)

        for i in range(stack_input_matrix.size(0)):
            for j in range(stack_input_matrix.size(1)):
                if tgt_mask_batch[i][j]:
                    stack_input_matrix[i, j] = mask_idx
        tgt_pos_batch = input_matrix_new.new(tgt_pos_batch)
        anchors = input_matrix_new.new(anchors)
    
    return stack_input_matrix, tgt_pos_batch, anchors, stack_target, stack_src_tokens 


def add_word_suggestion_sample(input_matrix, sugst_type, mask_idx, pad_idx):
    batch_size = input_matrix.size(0)
    target_lengths = torch.sum( input_matrix.ne(pad_idx).int(), dim=-1 )
    min_tgt_len = torch.min(target_lengths).item()
    max_tgt_len = input_matrix.size(1)
    tgt_len = min_tgt_len
    input_list = []
    position = []
    if tgt_len > 1:
        random_list = random.sample(range(0, tgt_len-1), k=1)     # 取 [0, tgt_len-1) 的原因在于不考虑最后一个token——"</s>"  
        for pos in random_list:
            input_matrix_new = input_matrix.clone().detach()
            if sugst_type == "bi_context":
                tgt_mask = gen_bi_context_mask(tgt_len, pos)
            else:
                raise NotImplementedError
            tgt_mask_batch = []
            tgt_pos_batch = []
            for b in range(batch_size):
                if sugst_type == "prefix":
                    raise NotImplementedError
                    new_tgt_mask = tgt_mask + [1] * (target_lengths[b].item() - len(tgt_mask))
                    new_tgt_mask = new_tgt_mask + [0] * (max_tgt_len - len(new_tgt_mask))
                else:
                    new_tgt_mask = tgt_mask + [0] * max( input_matrix_new.size(1) - min_tgt_len, 0 )
                tgt_mask_batch.append(new_tgt_mask)
                cur_len = target_lengths[b]
                tgt_pos_batch.append( get_pos_with_mask(new_tgt_mask, cur_len, pad_idx) ) # adpat this mask to fairseq positional encoding.
            for i in range(batch_size):
                for j in range(max_tgt_len):
                    if tgt_mask_batch[i][j]:
                        input_matrix_new[i, j] = mask_idx
            input_list.append(input_matrix_new)
            position.append(input_matrix_new.new(tgt_pos_batch))
    if input_list and position:
        x = input_list[0]
        y = position[0]
    else:
        x = None
        y = None
    # Until now, we have got a matrix of size (batch_size, max_tgt_len), each example is a positive example.
    # In rerank model, we should also construct K negative examples for every positive example. 
    target_position = random_list[0] 
    return x, y, target_position

def sample_types(tgt_tokens, tgt_dict, symbol2pinyin):
    """
    tgt_tokens: (bsz, ) Tensor of target tokens.#TODO :中文是拼音要进行修改
    """
    if symbol2pinyin == None: 
        tgt_tokens = tgt_tokens.tolist()
        tgt_tokens = [ tgt_dict.symbols[t] for t in tgt_tokens ]
        types = [ t[ : random.randint(0,int(len(t)*0.5))+1 ] for t in tgt_tokens ]
    else:
        tgt_tokens = tgt_tokens.tolist()
        tgt_tokens = [  symbol2pinyin[tgt_dict.symbols[t]] for t in tgt_tokens ]
        types = [ t[ : random.randint(0,int(len(t)*0.5))+1 ] for t in tgt_tokens ]    
    return types

def get_type_masks(types, tgt_dict, symbol2pinyin, target_indices=None):
    """
    在这一步中, 根据是否提供了target_indices, 来决定是否mask掉target token的概率。
    """
    if symbol2pinyin == None:
        type_masks = []
        if target_indices is None:
            for type in types:
                type_mask = [ not symbol.startswith(type) for symbol in tgt_dict.indices.keys()]
                type_masks.append(type_mask)        
        else:
            for type, target_idx in zip(types, target_indices):
                type_mask = [ not symbol.startswith(type) for symbol in tgt_dict.indices.keys()]
                # 将自己进行mask
                type_mask[target_idx] = True
                type_masks.append(type_mask)
        # type_masks : (bsz, vocab_size)
        type_masks = torch.tensor(type_masks, dtype=torch.bool)
    else:
        type_masks = []
        if target_indices is None:
            for type in types:
                type_mask = [ not symbol2pinyin[symbol].startswith(type) for symbol in tgt_dict.indices.keys()]
                type_masks.append(type_mask)        
        else:
            for type, target_idx in zip(types, target_indices):
                type_mask = [ not symbol2pinyin[symbol].startswith(type) for symbol in tgt_dict.indices.keys()]
                # 将自己进行mask
                type_mask[target_idx] = True
                type_masks.append(type_mask)
        # type_masks : (bsz, vocab_size)
        type_masks = torch.tensor(type_masks, dtype=torch.bool)       
    return type_masks

def read_synonyms():
    #TODO: Change hard code dictionary path
    dictionary_file = "/apdcephfs/share_916081/khalidcyang/project/gwlan-fairseq/dataset/bin/zh-en/bi_context/dict.en.txt"

    # # 建立词典
    symbols = dict()
    indices = dict()
    with open(dictionary_file, "r") as f:
        for idx, line in enumerate(f):
            word = line.strip().split(" ")[0]
            symbols[idx+4] = word
            indices[word] = idx + 4 # 必须要+4才行

    # 进行映射
    #TODO: change hard code length
    synonym_list = [ []  for _ in range(50001) ]
    with open("synonyms.txt", "r") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            idx = indices[word]
            for synonym in line.strip().split(" ")[1:]:
                synonym_list[idx].append(indices[synonym])

    # for i in range(4):
    #     synonym_list[i].append(i)
    # synonym_list[len(synonym_list)-1].append(len(synonym_list)-1)
    
    # print(synonym_list) 
    # print(synonym_list[1000])
    logger.info("len of synonym list: {}".format(len(synonym_list))) 
    return synonym_list

def get_synonyms(target, synonym_list, padding_value):
    #TODO: 有BUG，使用rerank_transformer.py里的版本
    lsts = []
    for tgt in target.tolist():
        lsts.append( [tgt] + synonym_list[tgt] )
    
    new_lsts = []
    for l in lsts:
        new_l = l[:1]
        if len(l) > 1:
            rest_l = l[1:]
            new_l += random.sample(rest_l, k=4 if len(rest_l) >= 4 else len(rest_l))
        new_lsts.append(new_l)

    tensor_lst = []
    for l in new_lsts:
        tensor_lst.append(torch.tensor(l))

    synonyms = pad_sequence(tensor_lst, batch_first=True, padding_value=padding_value) # pad value is two

    return synonyms

def collate(
    samples,
    pad_idx,
    eos_idx,
    mask_idx,
    suggestion_type,
    src_dict=None,
    tgt_dict=None,
    bi_dict=None,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    wordnet_enhanced=False,
    synonym_list=None,
    symbol2pinyin=None,
    stop_word_list=None,
    split=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
        else:
            # if input_feeding is False, we use `target` as the `prev_output_tokens`
            prev_output_tokens = target.clone().detach()    # TODO: learn from stackoverflow: https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
            bi_context_input_tokens = None
            bi_context_target = None
            prefix_input_tokens = None
            prefix_target = None
            suffix_input_tokens = None
            suffix_target = None
            zero_context_input_tokens = None
            zero_context_target = None
            synonyms = None
            if suggestion_type == "bi_context":
                # get input tokens , position ids and target position
                # bi_context_input_tokens: (bsz, tgt_len)
                # bi_context_position: (bsz, tgt_len)
                v2 = True   # use version 2 to do data sampling
                if v2 == False:
                    bi_context_input_tokens, bi_context_position, target_position = add_word_suggestion_sample(prev_output_tokens, "bi_context", mask_idx, pad_idx)
                    # target: (bsz, )
                    target = target[:, target_position]
                    types = sample_types(target, tgt_dict)
                    type_masks = get_type_masks(types, tgt_dict, target.tolist())
                else:             
                    v3 = False
                    if v3 == False and split == "train":
                        bi_context_input_tokens, bi_context_position, target_position, target, src_tokens = add_word_suggestion_sample_v3(
                            prev_output_tokens, "bi_context", mask_idx, pad_idx, tgt_dict, 
                            target, src_tokens, )
                        target = target.gather(1, target_position.unsqueeze(-1)).squeeze(-1)
                        types = sample_types(target, tgt_dict, symbol2pinyin)
                        type_masks = get_type_masks(types, tgt_dict, symbol2pinyin, target.tolist())
                        
                        # 根据GIZA++词典，得到对齐的结果。
                        # align_sents: (bsz, cutted_src_len)
                        # align_positions: (bsz, cutted_src_len)
                        align_sents = None
                        align_positions = None
                        if bi_dict is not None:
                            align_sents, align_positions = word_align(src_tokens,
                                                            src_dict=src_dict, 
                                                            tgt_dict=tgt_dict, 
                                                            bi_dict=bi_dict, 
                                                            stop_word_list=stop_word_list,
                                                            pad_idx=pad_idx,
                                                            )
                    else:
                        bi_context_input_tokens, bi_context_position, target_position = add_word_suggestion_sample_v2(prev_output_tokens, "bi_context", mask_idx, pad_idx, tgt_dict)
                        target = target.gather(1, target_position.unsqueeze(-1)).squeeze(-1)
                        types = sample_types(target, tgt_dict, symbol2pinyin)
                        type_masks = get_type_masks(types, tgt_dict, symbol2pinyin, target.tolist())
                        
                        # 根据GIZA++词典，得到对齐的结果。
                        # align_sents: (bsz, cutted_src_len)
                        # align_positions: (bsz, cutted_src_len)
                        align_sents = None
                        align_positions = None
                        if bi_dict is not None:
                            align_sents, align_positions = word_align(src_tokens,
                                                            src_dict=src_dict, 
                                                            tgt_dict=tgt_dict, 
                                                            bi_dict=bi_dict, 
                                                            stop_word_list=stop_word_list,
                                                            pad_idx=pad_idx,
                                                            )

                    # if wordnet_enhanced == True:
                        # if wordnet_enhanced == True, we should get synonyms for each target word
                        # gather synonyms
                        # synonyms = get_synonyms(target, synonym_list, padding_value=pad_idx)
            elif suggestion_type == "prefix":
                # get input tokens , position ids and target position
                prefix_input_tokens, prefix_position, target_position = add_word_suggestion_sample_v2(prev_output_tokens, "prefix", mask_idx, pad_idx, tgt_dict)
                target = target.gather(1, target_position.unsqueeze(-1)).squeeze(-1)
                types = sample_types(target, tgt_dict, symbol2pinyin)
                type_masks = get_type_masks(types, tgt_dict, symbol2pinyin, target.tolist())

                # 根据GIZA++词典，得到对齐的结果。
                # align_sents: (bsz, cutted_src_len)
                # align_positions: (bsz, cutted_src_len)
                align_sents = None
                align_positions = None
                if bi_dict is not None:
                    align_sents, align_positions = word_align(src_tokens,
                                                    src_dict=src_dict, 
                                                    tgt_dict=tgt_dict, 
                                                    bi_dict=bi_dict, 
                                                    stop_word_list=stop_word_list,
                                                    pad_idx=pad_idx,
                                                    )                    
            elif suggestion_type == "suffix":
                # get input tokens , position ids and target position
                suffix_input_tokens, suffix_position, target_position = add_word_suggestion_sample_v2(prev_output_tokens, "suffix", mask_idx, pad_idx, tgt_dict)
                target = target.gather(1, target_position.unsqueeze(-1)).squeeze(-1)
                types = sample_types(target, tgt_dict, symbol2pinyin)
                type_masks = get_type_masks(types, tgt_dict, symbol2pinyin, target.tolist())
                
                # 根据GIZA++词典，得到对齐的结果。
                # align_sents: (bsz, cutted_src_len)
                # align_positions: (bsz, cutted_src_len)
                align_sents = None
                align_positions = None
                if bi_dict is not None:
                    align_sents, align_positions = word_align(src_tokens,
                                                    src_dict=src_dict, 
                                                    tgt_dict=tgt_dict, 
                                                    bi_dict=bi_dict, 
                                                    stop_word_list=stop_word_list,
                                                    pad_idx=pad_idx,
                                                    ) 
            elif suggestion_type == "zero_context":
                # When we at the situation of "zero_context", we can not use add_word_suggestion_sample_v2() function
                align_sents = None
                align_positions = None
                zero_context_input_tokens, zero_context_position, target_position = add_word_suggestion_sample_v2(prev_output_tokens, "zero_context", mask_idx, pad_idx, tgt_dict)
                target = target.gather(1, target_position.unsqueeze(-1)).squeeze(-1)
                types = sample_types(target, tgt_dict, symbol2pinyin)
                type_masks = get_type_masks(types, tgt_dict, symbol2pinyin, target.tolist())
                # zero_context_input_tokens = torch.full((target.size(0), 1), mask_idx, dtype=torch.long)
                # zero_context_input_tokens = torch.cat( 
                #                             [ zero_context_input_tokens , torch.full((target.size(0), 1), eos_idx, dtype=torch.long) ], 
                #                             dim = -1
                # )
                # zero_context_position = torch.full((target.size(0), 1), 2, dtype=torch.long)
                # zero_context_position = torch.cat(
                #                         [ zero_context_position , torch.full((target.size(0), 1), 3, dtype=torch.long) ],
                #                         dim = -1
                # )
            else:
                raise NotImplementedError
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "type_masks": type_masks,
        "suggestion_type": suggestion_type,
        "target_position": target_position,
        "synonyms": synonyms,
        "align_sents": align_sents,
        "align_positions": align_positions,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens
        # batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
        #     0, sort_order
        # )
        if bi_context_input_tokens is not None:
            batch['bi_context_net_input'] = {
                "bi_context_position" : bi_context_position,
                "bi_context_input_tokens": bi_context_input_tokens,
                "bi_context_target": bi_context_target
            }
        if prefix_input_tokens is not None:
            batch["prefix_net_input"] = {
                "prefix_position": prefix_position,
                "prefix_input_tokens": prefix_input_tokens,
                "prefix_target": prefix_target
            }
        if suffix_input_tokens is not None:
            batch["suffix_net_input"] = {
                "suffix_position": suffix_position,
                "suffix_input_tokens": suffix_input_tokens,
                "suffix_target": suffix_target
            }
        if zero_context_input_tokens is not None:
            batch["zero_context_net_input"] = {
                "zero_context_position": zero_context_position,
                "zero_context_input_tokens": zero_context_input_tokens,
                "zero_context_target": zero_context_target
            }


    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch


class GWLANRerankMoreSignalDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        suggestion_type=None,   # customized parameter
        baseline=None,  # customized parameter
        symbol2pinyin=None, # customized parameter
        wordnet_enhanced=False, # customized parameter
        bi_dict=None,   # customized parameter
        stop_word_list=None,    # customized parameter
        split=None, # customized parameter
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.suggestion_type = suggestion_type
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.wordnet_enhanced = wordnet_enhanced
        self.synonym_list = read_synonyms()
        self.symbol2pinyin = symbol2pinyin
        self.baseline = baseline
        self.bi_dict = bi_dict
        self.stop_word_list = stop_word_list
        self.split = split

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            mask_idx=self.tgt_dict.mask(),
            tgt_dict=self.tgt_dict,
            suggestion_type=self.suggestion_type,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            wordnet_enhanced=self.wordnet_enhanced, # customized parameter
            synonym_list=self.synonym_list, # customized parameter
            symbol2pinyin=self.symbol2pinyin, # customized parameter
            src_dict=self.src_dict, # customized parameter
            bi_dict=self.bi_dict, # customized parameter
            stop_word_list=self.stop_word_list, # customized parameter
            split=self.split
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
