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
import torch.nn.functional as F


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
    wc_loss_coef: float = field(
        default=0.5,
        metadata={"help": "coefficient for wc_loss"}
    )
    decoder_word_prediction_coef: float = field(
        default=0,
        metadata={"help": "coefficient for decoder_word_prediction"}
    )
    loss_per_sentence: bool = field(
        default=False,
        metadata={"help": "Whether to calculate loss per sentence for the autoregressive task."}
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

def label_smoothed_nll_loss_per_example(lprobs, target, epsilon, bsz, ignore_index=None, reduce=True):
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
        # do reshape here
        nll_loss = nll_loss.view(bsz, -1)
        smooth_loss = smooth_loss.view(bsz, -1)

        # average view dimension
        token_nums = target.ne(ignore_index).view(bsz, -1).sum(dim=-1)
        nll_loss = nll_loss.div(token_nums)
        smooth_loss = smooth_loss.div(token_nums)

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "gwlan_info_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class GWLANInfoLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        wc_loss_coef=0.5,
        decoder_word_prediction_coef=0,
        loss_per_sentence=False
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.wc_loss_coef = wc_loss_coef
        self.decoder_word_prediction_coef = decoder_word_prediction_coef
        self.loss_per_sentence = loss_per_sentence

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
                src_tokens = sample["net_input"]["src_tokens"],
                src_lengths = sample["net_input"]["src_lengths"],
                prev_output_tokens = sample["net_input"]["prev_output_tokens"],
                src_position_ids = sample["net_input"]["src_position_ids"],
                src_segment_ids = sample["net_input"]["src_segment_ids"]
            )
        elif sample["suggestion_type"] == "prefix":
            net_output = model(
                src_tokens = sample["net_input"]["src_tokens"],
                src_lengths = sample["net_input"]["src_lengths"],
                prev_output_tokens = sample["net_input"]["prev_output_tokens"],
                src_position_ids = sample["net_input"]["src_position_ids"],
                src_segment_ids = sample["net_input"]["src_segment_ids"]
            )
        elif sample["suggestion_type"] == "suffix":
            net_output = model(
                src_tokens = sample["net_input"]["src_tokens"],
                src_lengths = sample["net_input"]["src_lengths"],
                prev_output_tokens = sample["net_input"]["prev_output_tokens"],
                src_position_ids = sample["net_input"]["src_position_ids"],
                src_segment_ids = sample["net_input"]["src_segment_ids"]
            )
        elif sample["suggestion_type"] == "zero_context":
            net_output = model(
                src_tokens = sample["net_input"]["src_tokens"],
                src_lengths = sample["net_input"]["src_lengths"],
                prev_output_tokens = sample["net_input"]["prev_output_tokens"],
                src_position_ids = sample["net_input"]["src_position_ids"],
                src_segment_ids = sample["net_input"]["src_segment_ids"]
            )    
            # raise NotImplementedError
            # net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["zero_context_net_input"]["zero_context_input_tokens"], tgt_position_ids = sample["zero_context_net_input"]["zero_context_position"] )        
            # in zero_context, net_output is (batch_size, 1, hidden_dim), we should expands it
            # net_output = (net_output[0][:,:1].expand(-1, sample["target"].size(1) ,-1), net_output[1])
        elif sample["suggestion_type"] == "mlm":
            raise NotImplementedError
            # net_output = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["net_input"]["prev_output_tokens"], tgt_position_ids = None) # must set tgt_position_ids to None
        else:
            raise NotImplementedError           
        
        # There are two parts loss,
        # One is for word classification
        # Another is for auto-regressive generation
        if getattr(self.task.cfg, "no_ar_task", False) != True:
            ar_loss, ar_nll_loss = self.compute_ar_loss(model, net_output[1], sample, reduce=reduce)
        else:
            assert self.sentence_avg == True
            ar_loss = 0
            ar_nll_loss = 0
        wc_loss, wc_nll_loss = self.compute_wc_loss(model, net_output[0], sample, reduce=reduce)
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        if getattr(self, "decoder_word_prediction_coef", 0) != 0:
            dwp_loss, dwp_nll_loss = self.compute_dwp_loss(model, net_output[1], sample, reduce=reduce)
        else:
            dwp_loss, dwp_nll_loss =0, 0

        loss = (1-self.wc_loss_coef) * ar_loss + self.wc_loss_coef * wc_loss + self.wc_loss_coef * self.decoder_word_prediction_coef * dwp_loss
        nll_loss = (1-self.wc_loss_coef) * ar_nll_loss + self.wc_loss_coef * wc_nll_loss + self.wc_loss_coef * self.decoder_word_prediction_coef * dwp_nll_loss

        # _, target = self.get_lprobs_and_target(model, net_output, sample)
        # mask = target.ne(self.padding_idx)
        # sample_size = torch.sum(mask)
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
            n_correct, total = self.compute_accuracy(model, net_output[0], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def eval_forward(self, model, sample, reduce=True):
        """Compute the accuracy for the given evaluation sample

        Note: loss and nll_loss are not computed here. Only the accuracy metric is computed.
        """
        print("#############################################################################")
        mask_logits = self.task.generate([model], sample)
        n_correct, total = self.compute_accuracy(model, mask_logits, sample)
        loss = -1
        nll_loss = -1
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss, 
            "nll_loss": nll_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["id"].size(0),
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
        try:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1) 
        except IndexError:
            print("sample:", sample)
            print("net_output:", net_output)
            print("lprobs:", lprobs)
            print("target:", target)
            print("lprobs size:", lprobs.size())
            print("target size:", target.size())
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_dwp_loss(self, model, net_ouput, sample, reduce=True):
        dwp_loss = F.cross_entropy(input=net_ouput[:,0], target=sample["target_tokens"], reduction="sum")
        dwp_nll_loss = dwp_loss
        return dwp_loss, dwp_nll_loss

    def compute_wc_loss(self, model, net_output, sample, reduce=True):
        wc_loss = F.cross_entropy(input=net_output, target=sample["target_tokens"], reduction='sum')
        wc_nll_loss = wc_loss
        return wc_loss, wc_nll_loss

    def compute_ar_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if  getattr(self, "loss_per_sentence", False) == False:
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
        else:
            bsz = sample["target"].size(0)
            loss, nll_loss = label_smoothed_nll_loss_per_example(
                lprobs,
                target,
                self.eps,
                bsz,
                ignore_index=self.padding_idx,
                reduce=reduce,               
            )
        if (target == torch.full(size=(target.size(0),1), fill_value=2).to(target.device)).all():
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        # lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        target = sample["target_tokens"]
        mask = target.ne(self.padding_idx)
        lprobs = net_output
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

