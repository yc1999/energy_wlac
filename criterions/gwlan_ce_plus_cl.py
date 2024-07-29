# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This loss is made for continual pretraining. 
"""

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")
from data.gwlan_rerank_dataset import get_type_masks

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    coef : float = field(
        default=0.5,
        metadata={"help": "coefficient for contrastive loss"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "gwlan_ce_plus_cl", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class GWLANCEPlusCLCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        coef = 0.5,
    ):
        """
        We use [MASK] to compute Cross Entropy Loss
        and then we use [MASK], positive, negative examples to compute contrastive loss. 
        """
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.coef = coef

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample["net_input"])
        if sample["suggestion_type"] == "bi_context":
            net_output = model(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                prev_output_tokens=sample["bi_context_net_input"]["bi_context_input_tokens"],
                tgt_position_ids = sample["bi_context_net_input"]["bi_context_position"],
                target_position = sample["target_position"],
                target = sample["target"],
                type_masks = sample["type_masks"],
                tgt_dict = self.task.target_dictionary
            )
        elif sample["suggestion_type"] == "prefix":
            pass
        elif sample["suggestion_type"] == "suffix":
            pass
        elif sample["suggestion_type"] == "zero_context":
            pass
        else:
            raise NotImplementedError           
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def generate(cls, task, models, sample):

        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        
        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], task.tgt_dict.pad()
        )

        # get new target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ task.tgt_dict.index(task.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        target_position = sample["net_input"]["prev_output_tokens"].eq(task.tgt_dict.mask()).nonzero()[:,-1]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [task.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, task.tgt_dict).to(tgt_position_ids.device)

        # logits: (bsz * (top_k+1), 1)
        # labels: (bsz * (top_k+1))
        net_output = model(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = sample["net_input"]["prev_output_tokens"] if task.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if task.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = task.tgt_dict,
                        mode="infer",
        )[0]
        
        # net_output: (bsz, tgt_vocab_size)
        net_output = net_output[(model.cfg.top_k) : : (model.cfg.top_k+1)]

        # get sample id
        ids = sample["id"].tolist()
        types = [task.datasets["types"][id] for id in ids]
        
        if task.cfg.target_lang != "zh":
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = [ not symbol.startswith(type) for symbol in task.tgt_dict.indices.keys()]
                type_masks.append(type_mask)
            # 变成tensor
            type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
        else:
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = []
                for symbol in task.tgt_dict.indices.keys():
                    symbol_pinyin = task.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
                    if symbol_pinyin.startswith(type):
                        type_mask.append(False)
                    else:
                        type_mask.append(True)
                type_masks.append(type_mask)
            # 变成tensor
            type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)            

        # 修改概率
        # net_output: (bsz, tgt_vocab_size)
        net_output = net_output.masked_fill(type_masks, -float('inf'))
        # 取最大的概率
        net_output = net_output.softmax(dim=-1)
        preds = torch.max(net_output, dim=-1).indices.cpu().numpy()
        probs = torch.max(net_output, dim=-1).values.cpu().numpy()
        
        # ground truth labels
        labels = [ task.tgt_dict.index(task.datasets["suggestions"][id]) for id in ids ]
        labels_probs = net_output.gather(1, torch.tensor(labels, dtype=torch.long).to(net_output.device).unsqueeze(1)).squeeze(-1).cpu().numpy()

        # 找到每个label的在概率中的位置
        # 对probs进行排序
        sorted_net_output = torch.argsort(net_output, dim=-1, descending=True)
        # 找到每个label在排序后的位置
        # 需要将sorted_net_output转换成cpu
        sorted_net_output = sorted_net_output.cpu().tolist()
        # 找到每个label在排序后的位置
        labels_probs_rk = []
        for i, label in enumerate(labels):
            labels_probs_rk.append(sorted_net_output[i].index(label))
        assert len(labels_probs_rk) == len(labels)

        # labels: (bsz, )
        labels = np.array(labels)
        
        n_correct = sum(labels == preds)
        total = preds.shape[0]

        return n_correct, total, labels, preds, probs, labels_probs, labels_probs_rk

    # def generate(self, models, sample):

    #     """
    #     This is the mask inner product generate function for RerankTransformer.
    #     """
    #     # get model from models
    #     model = models[0]

    #     ids = sample["id"].tolist()
        
    #     # get new position ids
    #     # tgt_position_ids: (bsz, seq_len)
    #     tgt_position_ids =  utils.make_positions(
    #             sample["net_input"]["prev_output_tokens"], self.task.tgt_dict.pad()
    #     )

    #     # get new target
    #     # (bsz,)
    #     # We should get this from .suggestion file
    #     target = torch.tensor([ self.task.tgt_dict.index(self.task.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)

    #     # get new target_position
    #     # (bsz,)
    #     # This time, it is a tensor, not a scalar
    #     # find every <mask>'s position in prev_output_tokens
    #     target_position = sample["net_input"]["prev_output_tokens"].eq(self.task.tgt_dict.mask()).nonzero()[:,-1]

    #     # get new type_masks
    #     # (bsz, vocab_size)
    #     # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
    #     types = [self.task.datasets["types"][id] for id in ids]
    #     type_masks = get_type_masks(types, self.task.tgt_dict).to(tgt_position_ids.device)

    #     # logits: (bsz * (top_k+1), 1)
    #     # labels: (bsz * (top_k+1))
    #     net_output = model(
    #                     src_tokens = sample["net_input"]["src_tokens"],
    #                     src_lengths = sample["net_input"]["src_lengths"],
    #                     prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.task.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
    #                     tgt_position_ids = tgt_position_ids if self.task.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
    #                     target_position = target_position,
    #                     target = target,
    #                     type_masks = type_masks,
    #                     tgt_dict = self.task.tgt_dict,
    #                     mode="infer",
    #     )[0]
    #     net_output = net_output[(model.cfg.top_k) : : (model.cfg.top_k+1)]

    #     # 根据prev_output_tokens里面是否为<mask>来决定当前掩码的目标
    #     # acc_mask : (bsz, seq_len)
    #     # acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.task.tgt_dict.mask()) if self.task.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"].eq(self.task.tgt_dict.eos()) # 只有是<mask>才会读取，之前写的是<eos>。见gwlan_dataset中的描述
    #     # net_output : (bsz, tgt_vocab_size)
    #     # net_output = net_output[acc_mask]
    #     # get sample id
    #     ids = sample["id"].tolist()
    #     types = [self.task.datasets["types"][id] for id in ids]
        
    #     if self.task.cfg.target_lang != "zh":
    #         type_masks = []
    #         # 查看词表进行概率掩码
    #         for type in types:
    #             type_mask = [ not symbol.startswith(type) for symbol in self.task.tgt_dict.indices.keys()]
    #             type_masks.append(type_mask)
    #         # 变成tensor
    #         type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
    #     else:
    #         type_masks = []
    #         # 查看词表进行概率掩码
    #         for type in types:
    #             type_mask = []
    #             for symbol in self.task.tgt_dict.indices.keys():
    #                 symbol_pinyin = self.task.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
    #                 if symbol_pinyin.startswith(type):
    #                     type_mask.append(False)
    #                 else:
    #                     type_mask.append(True)
    #             type_masks.append(type_mask)
    #         # 变成tensor
    #         type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)            

    #     # 修改概率
    #     # net_output: (bsz, tgt_vocab_size)
    #     net_output = net_output.masked_fill(type_masks, -float('inf'))
    #     # 取最大的概率
    #     preds = torch.max(net_output, dim=-1).indices.cpu().numpy()
        
    #     # ground truth labels
    #     labels = [ self.task.tgt_dict.index(self.task.datasets["suggestions"][id]) for id in ids ]
    #     # labels: (bsz, )
    #     labels = np.array(labels)
        
    #     n_correct = sum(labels == preds)
    #     total = preds.shape[0]

    #     return n_correct, total

    def eval_forward(self, model, sample, reduce=True):
        """Compute the accuracy for the given evaluation sample

        Note: loss and nll_loss are not computed here. Only the accuracy metric is computed.
        """
        n_correct, total = GWLANCEPlusCLCriterion.generate(self.task, [model], sample)[:2]
        
        loss = -1
        nll_loss = -1
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss, 
            "nll_loss": nll_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,  
            "n_correct": n_correct,
            "total": total,
        }

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = net_output[4]
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        # only probilities of [MASK] are used
        lprobs = lprobs[ (model.cfg.top_k) : : (model.cfg.top_k+1), :]
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def contrastive_learning_criterion(self, last_hidden_states, anchor_position):
        """
        contrastive learning loss
        last_hidden_states: ( bsz, top_k+1, tgt_len, hidden_dim )
        """
        if anchor_position.dim() == 1:
            # dynamic anchors

            # mask_anchor_position: ( bsz, ) -> (bsz, 1, 1, hidden_dim)
            mask_anchor_position = anchor_position.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            mask_anchor_position = mask_anchor_position.expand(-1, -1, -1, last_hidden_states.size(-1))
            # mask_hidden_states: (bsz, 1, hidden_dim)
            mask_hidden_states = last_hidden_states[:, -1:, :, :].gather(2, mask_anchor_position).squeeze(2)

            # top_k_anchor_position: ( bsz, ) -> (bsz, top_k, 1, hidden_dim)
            top_k_anchor_position = anchor_position.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            top_k_anchor_position = top_k_anchor_position.expand(-1, last_hidden_states.size(1)-1, -1, last_hidden_states.size(-1))
            # top_k_hidden_states: (bsz, top_k, hidden_dim)
            top_k_hidden_states = last_hidden_states[:, :-1, :, :].gather(2, top_k_anchor_position).squeeze(2)
        else:
            # mask_hidden_states: (bsz, 1, hidden_dim)
            mask_hidden_states = last_hidden_states[:, -1:, anchor_position , :]
            # top_k_hidden_states: (bsz, top_k, hidden_dim)
            top_k_hidden_states = last_hidden_states[:, :-1, anchor_position , :]

        # norm 2
        mask_hidden_states = mask_hidden_states / mask_hidden_states.norm(dim=2, keepdim=True)
        top_k_hidden_states = top_k_hidden_states / top_k_hidden_states.norm(dim=2, keepdim=True)

        temperature = 2e-2
        # (bsz, 1, top_k) -> (bsz, top_k)
        contrastive_score = torch.matmul(mask_hidden_states, top_k_hidden_states.transpose(1,2)) / temperature
        contrastive_score = contrastive_score.squeeze(1)
        
        # calculate cross entropy loss
        target = torch.full( size=(contrastive_score.size(0),), fill_value=0, device=contrastive_score.device )
        loss = F.cross_entropy(input=contrastive_score, target=target, reduction='sum')

        return loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        There is two part of loss:
        1) Cross Entropy Loss
        2) Contrastive Loss
        """
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        cross_entropy_loss, _ = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # (bsz * (top_k+1), tgt_len, hidden_dim) -> (bsz, top_k+1, tgt_len, hidden_dim)        
        last_hidden_states = net_output[1]["inner_states"][-1].transpose(0, 1)
        last_hidden_states = last_hidden_states.reshape( -1, model.cfg.top_k+1, last_hidden_states.size(1), last_hidden_states.size(2) )
        contrastive_loss = self.contrastive_learning_criterion(last_hidden_states, anchor_position = sample["target_position"])

        loss = (1 - self.coef) * cross_entropy_loss + (self.coef) * contrastive_loss
        nll_loss = loss # we simply use the loss as nll_loss
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

