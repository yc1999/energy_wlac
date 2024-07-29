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

logger = logging.getLogger(__name__)

def gen_bi_context_span_from_sent(tgt_len, pos):
    max_span_length = int( math.ceil( tgt_len * 0.25 ) )
    left_shift = min( random.randint(0, pos), max_span_length )
    right_shift = min( random.randint(1, tgt_len - pos), max_span_length + 1 )
    lhs = max(1, pos - left_shift)  # TODO: change this to 0
    rhs = min(pos + right_shift, tgt_len)
    return lhs, rhs

def gen_bi_context_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    span = gen_bi_context_span_from_sent(tgt_len, pos)
    masked = [0] * tgt_len
    for word_id in range(span[0], span[1]):
        masked[word_id] = 1
    return masked

def gen_suffix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(0, pos):
        masked[word_id] = 1
    return masked

def gen_prefix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(pos, tgt_len):
        masked[word_id] = 1
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

def add_word_suggestion_sample(input_matrix, sugst_type, mask_idx, pad_idx):
    batch_size = input_matrix.size(0)
    target_lengths = torch.sum( input_matrix.ne(pad_idx).int(), dim=-1 )
    min_tgt_len = torch.min(target_lengths).item()
    max_tgt_len = input_matrix.size(1)
    tgt_len = min_tgt_len
    input_list = []
    position = []
    if tgt_len > 1:
        random_list = random.sample(range(1, tgt_len), k=1)     # 注意区间的选择   
        for pos in random_list:
            input_matrix_new = input_matrix.clone().detach()
            if sugst_type == "bi_context":
                tgt_mask = gen_bi_context_mask(tgt_len, pos)
            elif sugst_type == "suffix":
                tgt_mask = gen_suffix_mask(tgt_len, pos)
            elif sugst_type == "prefix":
                tgt_mask = gen_prefix_mask(tgt_len, pos)
            else:
                raise NotImplementedError
            tgt_mask_batch = []
            tgt_pos_batch = []
            for b in range(batch_size):
                if sugst_type == "prefix":
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
        
        # only mask the pos.
        # mlm_tgt_input_matrix: (bsz, tgt_len)
        mlm_tgt_input_matrix = input_matrix.clone().detach()
        for pos in random_list:
            # index需要保持在dim上可与input不同，其他维度上相同
            index = torch.full(size=(input_matrix.size(0), 1), fill_value=pos)
            src = torch.full_like(index, fill_value=mask_idx)
            mlm_tgt_input_matrix = torch.scatter(input=mlm_tgt_input_matrix, dim=1, index=index, src=src)

    if input_list and position:
        x = input_list[0]
        y = position[0]
    else:
        x = None
        y = None
    return x, y, mlm_tgt_input_matrix


def collate(
    samples,
    pad_idx,
    eos_idx,
    mask_idx,
    suggestion_type,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
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
            if suggestion_type == "bi_context":
                # get masked prev_output_tokens, attention mask and position ids
                bi_context_input_tokens, bi_context_position, bi_context_mlm_input_tokens = add_word_suggestion_sample(prev_output_tokens, "bi_context", mask_idx, pad_idx)
                # for target, indeed, we should mask what is not mask token.
                # taget_mask : (bsz, tgt_len)
                # try:

                # target_mask = bi_context_input_tokens.ne(mask_idx)
                target_mask = bi_context_mlm_input_tokens.ne(mask_idx)
                
                # except AttributeError:
                #     print("prev_output_tokens", prev_output_tokens)
                #     print("bi_context_input_tokens", bi_context_input_tokens)
                #     print(samples)
                target = target.masked_fill(target_mask, pad_idx)
            elif suggestion_type == "prefix":
                prefix_input_tokens, prefix_position = add_word_suggestion_sample(prev_output_tokens, "prefix", mask_idx, pad_idx)
                target_mask = prefix_input_tokens.ne(mask_idx)
                target_mask = torch.logical_or(target_mask, target.eq(eos_idx))
                target = target.masked_fill(target_mask, pad_idx)
            elif suggestion_type == "suffix":
                suffix_input_tokens, suffix_position = add_word_suggestion_sample(prev_output_tokens, "suffix", mask_idx, pad_idx)
                target_mask = suffix_input_tokens.ne(mask_idx)
                target = target.masked_fill(target_mask, pad_idx)    
            elif suggestion_type == "zero_context":
                zero_context_input_tokens = torch.full((target.size(0), 1), mask_idx)   #TODO: important change from <mask> to <mask> <eos>
                zero_context_input_tokens = torch.cat( 
                                            [ zero_context_input_tokens , torch.full((target.size(0), 1), eos_idx) ], 
                                            dim = -1
                )
                zero_context_position = torch.full((target.size(0), 1), 2)
                zero_context_position = torch.cat(
                                        [ zero_context_position , torch.full((target.size(0), 1), 3)],
                                        dim = -1
                )
                target_mask = torch.logical_or(target.eq(pad_idx), target.eq(eos_idx))   # eos_idx 和 pad_idx 就不需要贡献loss了。
                target = target.masked_fill(target_mask, pad_idx)
            elif suggestion_type == "joint":
                bi_context_input_tokens, bi_context_position = add_word_suggestion_sample(prev_output_tokens, "bi_context", mask_idx, pad_idx)
                target_mask = bi_context_input_tokens.ne(mask_idx)
                bi_context_target = target.masked_fill(target_mask, pad_idx)

                prefix_input_tokens, prefix_position = add_word_suggestion_sample(prev_output_tokens, "prefix", mask_idx, pad_idx)
                target_mask = prefix_input_tokens.ne(mask_idx)
                prefix_target = target.masked_fill(target_mask, pad_idx)

                
                suffix_input_tokens, suffix_position = add_word_suggestion_sample(prev_output_tokens, "suffix", mask_idx, pad_idx)
                target_mask = suffix_input_tokens.ne(mask_idx)
                suffix_target = target.masked_fill(target_mask, pad_idx)    

                zero_context_input_tokens = torch.full((target.size(0), 1), mask_idx)   #TODO: important change from <mask> to <mask> <eos>
                zero_context_input_tokens = torch.cat( 
                                            [ zero_context_input_tokens , torch.full((target.size(0), 1), eos_idx) ], 
                                            dim = -1
                )
                zero_context_position = torch.full((target.size(0), 1), 2)
                zero_context_position = torch.cat(
                                        [ zero_context_position , torch.full((target.size(0), 1), 3)],
                                        dim = -1
                )
                target_mask = torch.logical_or(target.eq(pad_idx), target.eq(eos_idx))   # eos_idx 和 pad_idx 就不需要贡献loss了。
                zero_context_target = target.masked_fill(target_mask, pad_idx)
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
        "suggestion_type": suggestion_type,
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
                "bi_context_target": bi_context_target,
                "bi_context_mlm_input_tokens": bi_context_mlm_input_tokens
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


class GWLANKDDataset(FairseqDataset):
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
        suggestion_type=None,
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
            suggestion_type=self.suggestion_type,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
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
