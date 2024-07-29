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
    "gwlan_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class GWLANLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample["net_input"])
        if sample["suggestion_type"] == "bi_context":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["bi_context_net_input"]["bi_context_input_tokens"], tgt_position_ids = sample["bi_context_net_input"]["bi_context_position"] )
        elif sample["suggestion_type"] == "prefix":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["prefix_net_input"]["prefix_input_tokens"], tgt_position_ids = sample["prefix_net_input"]["prefix_position"] )
        elif sample["suggestion_type"] == "suffix":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["suffix_net_input"]["suffix_input_tokens"], tgt_position_ids = sample["suffix_net_input"]["suffix_position"] )        
        elif sample["suggestion_type"] == "zero_context":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["zero_context_net_input"]["zero_context_input_tokens"], tgt_position_ids = sample["zero_context_net_input"]["zero_context_position"] )        
            # in zero_context, net_output is (batch_size, 1, hidden_dim), we should expands it
            net_output = (net_output[0][:,:1].expand(-1, sample["target"].size(1) ,-1), net_output[1])
        elif sample["suggestion_type"] == "mlm":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["net_input"]["prev_output_tokens"], tgt_position_ids = None) # must set tgt_position_ids to None
        elif sample["suggestion_type"] == "imt":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["imt_net_input"]["imt_input_tokens"], tgt_position_ids = None)            
        elif sample["suggestion_type"] == "ts":
            net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["ts_net_input"]["ts_input_tokens"], tgt_position_ids = None)    
        else:
            raise NotImplementedError           
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        _, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        sample_size = torch.sum(mask)
        # sample_size = (
            # sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
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

        Note: loss and nll_loss are not computed here. Only the accuracy metric is computed.
        """
        labels, preds = self.task.generate([model], sample)
        n_correct = sum(labels == preds)
        total = preds.shape[0]
        
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
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
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

