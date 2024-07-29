# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

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
    softmax_normalization: bool = field(
        default=False,
        metadata={"help": "use softmax to normalize the output"},
    )
    coef: float = field(
        default=0.0,
        metadata={"help": "coefficient for the constrastive loss"},
    )
    eval_type: str = field(
        default="default",
        metadata={"help": "evaluation type"},
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
    "gwlan_weighted_binary_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class GWLANWeightedBinaryCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        softmax_normalization=False,
        coef=0.0,
        eval_type="default",
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.bce_criterion = BCEWithLogitsLoss(reduction="sum")
        self.softmax_criterion = CrossEntropyLoss(reduction="sum")
        logger.info("***************** softmax_normalization is {} *****************".format(softmax_normalization))
        self.softmax_normalization = softmax_normalization
        self.coef = coef
        self.eval_type = eval_type

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
                tgt_dict = self.task.target_dictionary,
                synonyms = sample["synonyms"],
                epoch_idx = self.task.epoch_idx,
                align_sents = sample["align_sents"],
                align_positions = sample["align_positions"]
            )
        elif sample["suggestion_type"] == "prefix":
            net_output = model(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                prev_output_tokens=sample["prefix_net_input"]["prefix_input_tokens"],
                tgt_position_ids = sample["prefix_net_input"]["prefix_position"],
                target_position = sample["target_position"],
                target = sample["target"],
                type_masks = sample["type_masks"],
                tgt_dict = self.task.target_dictionary,
                synonyms = sample["synonyms"],
                epoch_idx = self.task.epoch_idx,
                align_sents = sample["align_sents"],
                align_positions = sample["align_positions"]
            )
            # net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["prefix_net_input"]["prefix_input_tokens"], tgt_position_ids = sample["prefix_net_input"]["prefix_position"] )
        elif sample["suggestion_type"] == "suffix":
            net_output = model(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                prev_output_tokens=sample["suffix_net_input"]["suffix_input_tokens"],
                tgt_position_ids = sample["suffix_net_input"]["suffix_position"],
                target_position = sample["target_position"],
                target = sample["target"],
                type_masks = sample["type_masks"],
                tgt_dict = self.task.target_dictionary,
                synonyms = sample["synonyms"],                
                epoch_idx = self.task.epoch_idx,
            )
            # net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["suffix_net_input"]["suffix_input_tokens"], tgt_position_ids = sample["suffix_net_input"]["suffix_position"] )        
        elif sample["suggestion_type"] == "zero_context":
            net_output = model(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                prev_output_tokens=sample["zero_context_net_input"]["zero_context_input_tokens"],
                tgt_position_ids = sample["zero_context_net_input"]["zero_context_position"],
                target_position = sample["target_position"],
                target = sample["target"],
                type_masks = sample["type_masks"],
                tgt_dict = self.task.target_dictionary,
                synonyms = sample["synonyms"],                
                epoch_idx = self.task.epoch_idx,
            )            
            # net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["zero_context_net_input"]["zero_context_input_tokens"], tgt_position_ids = sample["zero_context_net_input"]["zero_context_position"] )        
            # # in zero_context, net_output is (batch_size, 1, hidden_dim), we should expands it
            # net_output = (net_output[0][:,:1].expand(-1, sample["target"].size(1) ,-1), net_output[1])
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

    def eval_forward(self, model, sample, reduce=True):
        """Compute the accuracy for the given evaluation sample

        Note: 
        1. loss and nll_loss are not computed here. Only the accuracy metric is computed in `self.task.rerank_generate()` function.
        2. The model is already in eval mode.
        """
        if self.eval_type == "inner_product":
            n_correct, total = self.task.rerank_mask_inner_product_generate([model], sample)[:2]
        elif self.eval_type == "default":
            # We only need (n_correct, total) for validation.
            n_correct, total = self.task.rerank_generate([model], sample)[:2]
        else:
            raise NotImplementedError("eval_type {} is not supported".format(self.eval_type))
        
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

    def get_logits_and_target(self, model, net_output, sample):
        logits = net_output[0]
        target = net_output[2]
        assert logits.size(0) == target.size(0)
        return logits.view(-1), target

    # def get_lprobs_and_target(self, model, net_output, sample):
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     target = model.get_targets(sample, net_output)
    #     if self.ignore_prefix_size > 0:
    #         # lprobs: B x T x C
    #         lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
    #         target = target[:, self.ignore_prefix_size :].contiguous()
    #     return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

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
        # logits, target：(bsz * top_k) or (bsz * (top_k+1))
        logits, target = self.get_logits_and_target(model, net_output, sample)
        if self.softmax_normalization == False:
            if model.cfg.contrastive_learning == True:
                # logits: (bsz, top_k+1) -> (bsz * top_k)
                logits = logits.view(-1, model.cfg.top_k+1)
                logits = logits[:, :-1]
                bsz = logits.size(0)
                logits = logits.reshape(-1)
                
                target = target.view(-1, model.cfg.top_k+1)
                target = target[:, :-1]
                assert bsz == target.size(0)
                target = target.reshape(-1)
                
                bce_loss = self.bce_criterion(logits, target.float())                               
                
                # (bsz * (top_k+1), tgt_len, hidden_dim) -> (bsz, top_k+1, tgt_len, hidden_dim)
                last_hidden_states = net_output[1]["inner_states"][-1].transpose(0, 1)
                last_hidden_states = last_hidden_states.reshape( bsz, model.cfg.top_k+1, last_hidden_states.size(1), last_hidden_states.size(2) )
                cl_loss = self.contrastive_learning_criterion(last_hidden_states, anchor_position = sample["target_position"])
                
                loss = (1-self.coef) * bce_loss + (self.coef) * cl_loss
            else:
                # we need to zoom in the positive's weight
                weight = torch.full_like(input=target, fill_value=1)
                for idx in range(weight.size(0)):
                    if idx % model.cfg.top_k == 0:
                        weight[idx] = model.cfg.top_k // 2
                loss = F.binary_cross_entropy_with_logits(input=logits, target=target.float(), weight=weight,reduction="sum")
        else:
            # If self.softmax_normalization is True, we need to normalize the logits with top_k
            # logits: (bsz, top_k)
            logits = logits.view( -1, model.cfg.top_k ) 
            # generate new target
            # target: (bsz,)
            target = torch.full((logits.size(0),), 0).to(logits.device) # Only 0 is the correct label
            loss = self.softmax_criterion(logits, target)
            
        nll_loss = loss
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        # original formula:
        # lprobs, target = self.get_logits_and_target(model, net_output, sample)
        # mask = target.ne(self.padding_idx)
        # n_correct = torch.sum(
        #     lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        # )
        # total = torch.sum(mask)

        # new formula: 
        logits, target = self.get_logits_and_target(model, net_output, sample)
        logits = logits.view( -1, model.cfg.top_k)
        target = torch.full( (logits.size(0),), 0 ).to(logits.device)
        preds = logits.argmax(dim=-1)
        n_correct = (target == preds).sum()
        total = torch.tensor(target.size(0)).to(target.device)

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

