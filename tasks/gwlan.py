# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

sys.path.append("./")
import json
from data.gwlan_rerank_dataset import get_type_masks
from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import List, Optional
from argparse import Namespace
from omegaconf import II
from fairseq import metrics, search, tokenizer, utils

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.dictionary import Dictionary
from data.gwlan_dictionary import GWLANDictionary
import torch
from torch.nn.utils.rnn import pad_sequence
from data.gwlan_dataset import GWLANDataset
from data.gwlan_mlm_dataset import GWLANMLMDataset
from data.gwlan_rerank_dataset import GWLANRerankDataset
from data.gwlan_kd_dataset import GWLANKDDataset
from data.gwlan_electra_dataset import GWLANElectraDataset
from data.gwlan_rerank_tag_dataset import GWLANRerankTagDataset
from data.gwlan_prerank_dataset import GWLANPRerankDataset
from data.gwlan_nat_dataset import GWLANNatDataset
from data.gwlan_single_signal_dataset import GWLANSingleSignalDataset
from data.gwlan_rerank_more_signal_dataset import GWLANRerankMoreSignalDataset
from data.gwlan_nat_sc_dataset import GWLANNatSCDataset
from data.gwlan_hw_dataset import GWLANHWDataset
from data.gwlan_ar_dataset import GWLANArDataset
from data.gwlan_traditional_dataset import GWLANTraditionalDataset
from data.gwlan_rerank_traditional_dataset import GWLANRerankTraditionalDataset
from data.gwlan_ar_traditional_dataset import GWLANArTraditionalDataset

from criterions.gwlan_label_smoothed_cross_entropy import GWLANLabelSmoothedCrossEntropyCriterion
from criterions.gwlan_binary_cross_entropy import GWLANBinaryCrossEntropyCriterion
from criterions.gwlan_ce_plus_cl import GWLANCEPlusCLCriterion
from criterions.gwlan_margin_ranking_loss import GWLANMarginRankingLoss
from criterions.gwlan_joint_label_smoothed_cross_entropy import GWLANJointLabelSmoothedCrossEntropyCriterion
from criterions.gwlan_kd_label_smoothed_cross_entropy import GWLANKDLabelSmoothedCrossEntropyCriterion
from criterions.gwlan_electra_loss import GWLANElectraLossCriterion
from criterions.gwlan_mc_loss import GWLANMCLoss
from criterions.gwlan_tag_binary_cross_entropy import GWLANTagBinaryCrossEntropyCriterion
from criterions.gwlan_weighted_binary_cross_entropy import GWLANWeightedBinaryCrossEntropyCriterion
from criterions.gwlan_prerank_binary_cross_entropy import GWLANPRerankBinaryCrossEntropyCriterion
from criterions.gwlan_nat_loss import GWLANNatLoss
from criterions.gwlan_nat_sc_loss import GWLANNatSCLoss
from criterions.gwlan_hw_loss import GWLANHWLoss
from criterions.gwlan_ar_loss import GWLANARLoss
from pypinyin import lazy_pinyin

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    suggestion_type,
    rerank,
    mlm,
    cfg,
    symbol2pinyin,
    wordnet_enhanced,
    baseline,
    bi_dict,
    stop_word_list,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    types=None,
    suggestions=None
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    if cfg.prerank == True:
        logger.info("This dataset is for Prerank")
        return GWLANPRerankDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            symbol2pinyin=symbol2pinyin,
            wordnet_enhanced=wordnet_enhanced,
            baseline=baseline,
            bi_dict=bi_dict,
            stop_word_list=stop_word_list,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
            split=split
        )
    elif cfg.auto_regressive == True:
        logger.info(f"This dataset is for autoregressive prediction.")
        return GWLANArDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            split=split,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False           
            types=types,
            suggestions=suggestions,
            symbol2pinyin=symbol2pinyin if cfg.target_lang == "zh" else None,
            src_lang = cfg.source_lang,
            target_lang = cfg.target_lang,
            full_target_sentences= getattr(cfg, "full_target_sentences", False),
            char_embedding = getattr(cfg, "char_embedding", False),
        )         
    elif cfg.ar_traditional_solution == True:
        logger.info("This dataset is for autoregressive traditional prediction.")
        return GWLANArTraditionalDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            split=split,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False           
            types=types,
            suggestions=suggestions,
            symbol2pinyin=symbol2pinyin if cfg.target_lang == "zh" else None,
            src_lang = cfg.source_lang,
            target_lang = cfg.target_lang,
            full_target_sentences= getattr(cfg, "full_target_sentences", False),
            char_embedding = getattr(cfg, "char_embedding", False),
        )        
    elif cfg.rerank_traditional_solution:
        logger.info(f"This dataset is for rerank traditional solution.")
        return GWLANRerankTraditionalDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            symbol2pinyin=symbol2pinyin,
            wordnet_enhanced=wordnet_enhanced,
            baseline=baseline,
            bi_dict=bi_dict,
            stop_word_list=stop_word_list,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
            split=split
        )        
    elif cfg.nat_sc == True:
        logger.info("This dataset is for NAT Soft Constrained paradigm.")
        return GWLANNatSCDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
            split=split
        )      
    elif cfg.nat == True:
        logger.info("This dataset is for NAT paradigm.")
        return GWLANNatDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False
        )         
    elif cfg.rerank_tag == True:
        logger.info("This dataset is for rerank tag")
        return GWLANRerankTagDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            suggestion_type=suggestion_type,
            symbol2pinyin=symbol2pinyin,
            wordnet_enhanced=wordnet_enhanced,
            baseline=baseline,
            bi_dict=bi_dict,
            stop_word_list=stop_word_list,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
            split=split
        )   
    elif rerank == False and cfg.multiple_choice == False:
        if mlm == True:
            logger.info(f"This dataset is for MLM")
            return GWLANMLMDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False
            )
        elif cfg.electra == True:
            logger.info("This dataset is for ELECTRA training strategy.")
            return GWLANElectraDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                symbol2pinyin=symbol2pinyin,
                wordnet_enhanced=wordnet_enhanced,
                baseline=baseline,
                bi_dict=bi_dict,
                stop_word_list=stop_word_list,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
                split=split 
            )
        elif cfg.knowledge_distillation == True:
            logger.info(f"This dataset is for knowledge distillation version Generator.")
            return GWLANKDDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False
            )            
        elif cfg.gwlan_single_signal_dataset == True:
            logger.info(f"This dataset is for single signal baseline") 
            return GWLANSingleSignalDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False
            )          
        elif cfg.hw_solution == True:
            logger.info("This dataset is for huawei solution")
            return GWLANHWDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                symbol2pinyin=symbol2pinyin,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False                
            ) 
        elif cfg.traditional_solution == True:
            logger.info("This dataset is for traditional solution")
            return GWLANTraditionalDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                symbol2pinyin=symbol2pinyin if cfg.target_lang == "zh" else None,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False                
            ) 
        else:
            logger.info(f"This dataset is for baseline")
            return GWLANDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False     # Since we don't use a teacher forcing manner, we set input_feeding to False
            )
    else:
        if cfg.gwlan_rerank_more_signal_dataset == True:
            logger.info(f"This dataset is for reranking with more signal")
            return GWLANRerankMoreSignalDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                symbol2pinyin=symbol2pinyin,
                wordnet_enhanced=wordnet_enhanced,
                baseline=baseline,
                bi_dict=bi_dict,
                stop_word_list=stop_word_list,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
                split=split
            )          
        else:
            logger.info(f"This dataset is for reranking")
            return GWLANRerankDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset_sizes,
                tgt_dict,
                suggestion_type=suggestion_type,
                symbol2pinyin=symbol2pinyin,
                wordnet_enhanced=wordnet_enhanced,
                baseline=baseline,
                bi_dict=bi_dict,
                stop_word_list=stop_word_list,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                eos=eos,
                num_buckets=num_buckets,
                shuffle=shuffle,
                pad_to_multiple=pad_to_multiple,
                input_feeding=False,     # Since we don't use a teacher forcing manner, we set input_feeding to False
                split=split
            )        


@dataclass
class GWLANConfig(FairseqDataclass):
    """This is task specific args"""
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    gen_subset: str = II("dataset.gen_subset")
    results_path: str = field(
        default="results",
        metadata={
            "help": "path to save results to"
        },
    )
    output_verbose: bool = field(
        default=False,
        metadata={
            "help": "if True, then output verbose results"
        }
    )

    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    suggestion_type: str = field(
        default="error",
        metadata={"help": "suggestion type for this task"}
    )

    rerank: bool = field(
        default=False,
        metadata={
            "help": "two stage model with baseline(first stage) and reranking(second stage)"
        }
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "If it is pretrained with mlm."
        }
    )

    baseline_path: Optional[str] = field(
        default=None,
    )

    contrastive_learning_only: bool = field(
        default=False,
        metadata={
            "help": "If this is true, it means that we only use contrastive learning in rerank model."
        }
    )

    continual_pretraining: bool = field(
        default=False,
        metadata={
            "help": "Whether to use continual pretraining for evaluating."
        }
    )

    wordnet_enhanced: bool = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use wordnet information to enhance the word embedding layer of Transformer target side."
        }
    )

    hypernym_enhanced: bool = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use hypernym information to enhance the word embedding layer of Transformer target side."
        }
    )

    wn_hybrid_fusion: bool = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use synonym AND hypernym information to enhance the word embedding layer of Transformer target side."
        }
    )

    local_fairseq: bool = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use fairseq version localinfogatherlayer to enhance the word embedding layer of Transformer target side."
        }
    )

    local_num_heads: int = field(
        default=1,
        metadata={
            "help": "Parameter for reranker: the number of attention heads in LocalInfoGatherLayer."
        }
    )

    supply_path: str = field(
        # default="/apdcephfs/share_916081/khalidcyang/project/dataset/raw",
        default=os.environ.get('dataset_root')+"/raw",
        metadata={
            "help": "path to the supply dataset"
        }
    )

    tgt_input_enhanced: bool = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use synonyms to enhance target inputs."
        }
    )

    enhanced_type: str = field(
        default="synonym",
        metadata={
            "help": "Parameter for reranker: the type of KG enhancement to use."
        }
    )

    k_folds: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Parameter for reranker: the number of folds to use for cross validation."
        }
    )

    k_folds_baselines_path: Optional[str] = field(
        default="None",
        metadata={
            "help": "Parameter for reranker: the path to the baselines to use for cross validation."
        }
    )

    word_alignment_enhanced: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: whether to use word alignment to enhance the word embedding layer of Transformer target side."
        }
    )

    mlm_baseline_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for baseline: the path to the mlm to use for initializing the baseline."
        }
    )

    reranker_init_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for reranker: the path to the reranker to use for initializing the reranker."
        }
    )

    knowledge_distillation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for generator: whether to use knowledge distillation in the training process of the generator."
        }
    )

    mlm_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for generator: the path to the masked language modeling model."
        }
    )

    electra: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for electra: whether to train like electra joint style."
        }
    )

    electra_mlm_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for electra: the path to the masked language modeling model."
        }
    )

    electra_initialize_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for electra: the strategy to initialize weights."
        }
    )

    electra_tied_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for electra: the strategy used to tied weights of generator and reranker."
        }
    )

    electra_cut_off: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for electra: whether to cut off the generator training after some epochs."
        }
    )

    electra_cut_off_epoch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Parameter for electra: if electra_cut_off is True, cut off the generator training after some epochs."
        }
    )

    electra_pretrained_generator_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Parameter for electra: if electra_cut_off is True, we should provide pretrained generator path."
        }
    )

    multiple_choice: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for multiple choice: if multiple_choice is true, we use multiple choice strategy."
        }
    )

    rerank_tag: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for rerank tag: if rerank_tag is True, we use rerank_tag version dataset."
        }
    )

    prerank: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for prerank tag: if prerank is True, we use prerank version dataset."
        }
    )

    gwlan_single_signal_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for generator: if gwlan_single_signal_dataset is True, we only use one signal."
        }
    )

    gwlan_rerank_more_signal_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: if gwlan_rerank_more_signal_dataset is True, we only use one signal."
        }
    ) 

    reranker_embed_init_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for reranker: if reranker_embed_init_only is True, we only init the embed layer."
        }
    )
    
    inference_save_attn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for inference: if inference_save_attn is True, we will save attn score during inferece/tesing stage."
        }
    )

    inference_combine: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for inference: if inference_combine is True, we will combine the score of generator and reranker."
        }
    )

    baseline_ensemble: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Parameter for baseline inference: if baseline_ensemble is True, we will use another baseline inference function."    
        }
    )

    #* For NAT paradigm
    nat: bool = field(
        default=False,
        metadata={
            "help": "whether to use nat training strategy to train model."
        }
    )

    lp_factor: float = field(
        default=0.1,
        metadata={
            "help": "weight factor of length prediction loss."
        }
    )
    
    len_beam: int = field(
        default=1,
        metadata={
            "help": "len span."
        }
    )
    # mlm_init_model: str = field(
    #     default=None,
    #     metadata={
    #         "help": "Parameter for mlm_init_model path."
    #     }
    # )

    #* For NAT SC paradigm
    nat_sc: bool = field(
        default=False,
        metadata={
            "help": "whether to use NAT soft constraints to train the model."
        }
    )

    #* For hw reimplementation
    hw_solution: bool = field(
        default = False,
        metadata={
            "help": "Whether this task is used to reimplement the solution of huawei."
        }
    )

    auto_regressive: bool = field(
        default = False,
        metadata={
            "help": "Whether to use autoregressive way to do prediction."
        }
    )

    char_embedding: bool = field(
        default=False,
        metadata={
            "help": "Whether to use char_embedding to the model."
        }
    )

    candidates_file: Optional[str] = field(
        default=None
    )

    traditional_solution: bool = field(
        default=False,
        metadata={
            "help": "Whether this task is used to reimplement the interactive machine translation or transaltion suggestions."
        }
    )

    rerank_traditional_solution: bool = field(
        default=False,
        metadata={
            "help": "Whether this task is used to reimplement the reranker imt or ts."
        }
    )

    ar_traditional_solution: bool = field(
        default=False,
        metadata={
            "help": "Whether this task is used to reimplement the ar traditional imt or ts."
        }
    )

@register_task("gwlan", dataclass=GWLANConfig)
class GWLANTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: GWLANConfig

    def __init__(self, cfg: GWLANConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.load_eval_dataset(self.cfg.valid_subset, self.cfg.supply_path)
        
        self.baseline = None  # This paparameter is abondoned. Now baseline is put into the RerankerTransformer class.
        self.bi_dict = None
        self.stop_word_list = None

        if hasattr(self.cfg, "word_alignment_enhanced") and self.cfg.word_alignment_enhanced:
            with open("/apdcephfs/share_916081/khalidcyang/project/gwlan-fairseq/dataset/raw/bi_dict/{}{}_merger.json".format(self.cfg.source_lang, self.cfg.target_lang), "r", encoding="utf-8") as f:
                self.bi_dict = json.load(f)

            with open("/apdcephfs/share_916081/khalidcyang/project/gwlan-fairseq/dataset/bin/{}-{}/{}/dict.{}.txt".format(self.cfg.source_lang, self.cfg.target_lang, self.cfg.suggestion_type, self.cfg.source_lang), "r", encoding="utf-8") as f:
                self.stop_word_list = []
                for idx, line in enumerate(f):
                    line = line.strip().split(" ")
                    self.stop_word_list.append(line[0])
                    if idx == 99:
                        break

        if hasattr(cfg, "tgt_input_enhanced"):
            logger.info("***************** tgt_input_enhanced is {} *****************".format(cfg.tgt_input_enhanced))
        if hasattr(cfg, "k_folds"):
            logger.info("***************** k_folds is {} *****************".format(cfg.k_folds))
        if hasattr(cfg, "word_alignment_enhanced"):
            logger.info("***************** word_alignment_enhanced is {} *****************".format(cfg.word_alignment_enhanced))
        if hasattr(cfg, "criterion"):
            logger.info("***************** criterion is {} *****************".format(cfg.criterion))
        if hasattr(cfg, "margin"):
            logger.info("***************** margin is {} *****************".format(cfg.margin))
        if hasattr(cfg, "dynamic_negative_sampling"):
            logger.info("***************** dynamic_negative_sampling is {} *****************".format(cfg.dynamic_negative_sampling))
        
        # if self.cfg.rerank == True:
            # self.baseline = self.load_baseline(self.cfg.baseline_path)

    def load_baseline(self, path):
        """
        Given the path to the baseline model, load the model and return it.
        """
        from fairseq.checkpoint_utils import load_checkpoint_to_cpu
        from fairseq import tasks
        if "/shared_info/" in path:
            path = path.replace("/shared_info/", "/")
        state = load_checkpoint_to_cpu(path)
        cfg = state['cfg']
        task = tasks.setup_task(cfg.task)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        baseline = task.build_model(cfg.model, from_checkpoint=True)
        baseline.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)
        baseline.to(device)
        baseline.eval()
        return baseline   

    @classmethod
    def setup_task(cls, cfg: GWLANConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        if "/shared_info/" in paths[0]:
            paths[0] = paths[0].replace("/shared_info/", "/")
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        # print(src_dict.mask())
        # print(tgt_dict.mask())
        assert src_dict.mask() == src_dict.mask_index
        assert tgt_dict.mask() == tgt_dict.mask_index
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)


    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if "/apdcephfs/share_916081/khalidcyang/project/" in filename:
            filename = filename.replace("/apdcephfs/share_916081/khalidcyang/project/", "/data/users/zhangjunlei/yangc/project/khalidcyang/wlac-fairseq/")
        dictionary = GWLANDictionary.load(filename)
        dictionary.add_symbol("<mask>")
        dictionary.mask_index = dictionary.indices["<mask>"]

        return dictionary

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # 记录下epoch index, 其是从1开始的。
        self.epoch_idx = epoch

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            suggestion_type=self.cfg.suggestion_type,
            rerank=self.cfg.rerank,
            mlm=self.cfg.mlm,
            cfg=self.cfg,
            symbol2pinyin=self.symbol2pinyin if hasattr(self, "symbol2pinyin") else None,
            wordnet_enhanced=self.cfg.wordnet_enhanced,
            baseline=self.baseline,
            bi_dict=self.bi_dict,
            stop_word_list=self.stop_word_list,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split == "train"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            types=self.datasets["types"] if split != "train" else None,
            suggestions=self.datasets["suggestions"] if split != "train" else None,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
    
    def inference_step(self, models, sample):
        with torch.no_grad():
            return self.infer_generate(
                models, sample
            )
    
    def baseline_ensemble_inference_step(self, models, sample):
        with torch.no_grad():
            return self.baseline_ensemble_infer_generate(
                models, sample
            )
    
    def nat_inference_step(self, models, sample):
        with torch.no_grad():
            return self.nat_length_ensemble_generate(
                models, sample
            )

    def nat_sc_inference_step(self, models, sample):
        with torch.no_grad():
            return self.nat_sc_generate(
                models, sample
            )

    def rerank_inference_step(self, models, sample, mode="infer"):
        with torch.no_grad():
            if self.cfg.inference_save_attn:
                return self.rerank_generate_save_attn(
                    models, sample, mode
                )
            elif self.cfg.inference_combine:
                return self.rerank_generate_combine(
                    models, sample, mode
                )
            else:
                return self.rerank_generate(
                    models, sample, mode
                )

    def rerank_tag_inference_step(self, models, sample, mode="infer"):
        with torch.no_grad():
            return self.rerank_tag_generate(
                models, sample, mode
            )

    def continual_pretraining_inference_step(self, models, sample):
        with torch.no_grad():
            return  GWLANCEPlusCLCriterion.generate(
                self,
                models,
                sample,
            )
    
    def valid_step(self, sample, model, criterion):
        """Do forward pass in evaluation mode which is quite similar to the original valid_step"""
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion.eval_forward(model, sample)
        return loss, sample_size, logging_output

    def infer_generate(self, models, sample):
        """用于生成fairseq-generate的结果, 比用于fairseq-train的 `generate()` 函数多了:
        1. 保存attention权重;
        2. 提取 预测token 和 目标token 的预测分数;
        """
        # get model from models
        model = models[0]
        
        # call model forward

        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # net_output: (bsz, seq_len, tgt_vocab_size)
        net_output, extra = model(
                        src_tokens=sample["net_input"]["src_tokens"],
                        src_lengths=sample["net_input"]["src_lengths"],
                        prev_output_tokens=sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"]
        )
        attn = extra["attn"][0]
        # logger.info(attn.size())

        # 根据prev_output_tokens里面是否为<mask>来决定当前掩码的目标
        # acc_mask : (bsz, seq_len)
        acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()) if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"].eq(self.tgt_dict.mask()) # 对于非zero_context是<mask>才会读取，而zero_context是<eos>。见gwlan_dataset中的描述
        # net_output : (bsz, tgt_vocab_size)
        net_output = net_output[acc_mask]
        # get sample id
        ids = sample["id"].tolist()
        types = [self.datasets["types"][id] for id in ids]
        
        if self.cfg.target_lang != "zh":
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = [ not symbol.startswith(type) for symbol in self.tgt_dict.indices.keys()]
                type_masks.append(type_mask)
            # 变成tensor
            type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
        else:
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = []
                for symbol in self.tgt_dict.indices.keys():
                    symbol_pinyin = self.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
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
        labels = [ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]
        labels_probs = net_output.gather(1, torch.tensor(labels, dtype=torch.long).to(net_output.device).unsqueeze(1)).squeeze(-1).cpu().numpy()

        # 找到每个label的在概率中的位置
        # 对probs进行排序
        sorted_net_output = torch.argsort(net_output, dim=-1, descending=True)
        k = 3
        topk_sorted_net_output = sorted_net_output[:,:k].cpu().tolist()
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
        
        true = sum(labels == preds)
        size = preds.shape[0]

        # 输出attention文件
        if self.cfg.inference_save_attn is True:
            self.save_attention(
                            ids=ids,
                            attn=attn,
                            src_tokens=sample["net_input"]["src_tokens"],
                            prev_output_tokens=sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
            )
        return labels, preds, probs, labels_probs, labels_probs_rk, topk_sorted_net_output
    
    def baseline_ensemble_infer_generate(self, models, sample):
        """用于生成fairseq-generate的结果, 比用于fairseq-train的 `generate()` 函数多了:
        1. 保存attention权重;
        2. 提取 预测token 和 目标token 的预测分数;
        """
        # get model from models
        model = models[0]
        
        # call model forward

        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # net_output: (bsz, seq_len, tgt_vocab_size)
        net_output, extra = model(
                        src_tokens=sample["net_input"]["src_tokens"],
                        src_lengths=sample["net_input"]["src_lengths"],
                        prev_output_tokens=sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"]
        )
        attn = extra["attn"][0]
        # logger.info(attn.size())

        # 根据prev_output_tokens里面是否为<mask>来决定当前掩码的目标
        # acc_mask : (bsz, seq_len)
        acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()) if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"].eq(self.tgt_dict.mask()) # 对于非zero_context是<mask>才会读取，而zero_context是<eos>。见gwlan_dataset中的描述
        # net_output : (bsz, tgt_vocab_size)
        net_output = net_output[acc_mask]
        # get sample id
        ids = sample["id"].tolist()
        types = [self.datasets["types"][id] for id in ids]
        
        if self.cfg.target_lang != "zh":
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = [ not symbol.startswith(type) for symbol in self.tgt_dict.indices.keys()]
                type_masks.append(type_mask)
            # 变成tensor
            type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
        else:
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = []
                for symbol in self.tgt_dict.indices.keys():
                    symbol_pinyin = self.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
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
        labels = [ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]
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
        
        true = sum(labels == preds)
        size = preds.shape[0]

        # 输出attention文件
        if self.cfg.inference_save_attn is True:
            self.save_attention(
                            ids=ids,
                            attn=attn,
                            src_tokens=sample["net_input"]["src_tokens"],
                            prev_output_tokens=sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
            )
        return labels, preds, probs, labels_probs, labels_probs_rk, net_output, types, [ self.datasets["suggestions"][id] for id in ids ]

    def rerank_generate(self, models, sample, mode="infer"):
        """
        This is the infer generate function for RerankTransformer used in testing stage.
        """
        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        # print(ids)

        # get position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # get target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)
        # print(target.tolist())
        # print(self.tgt_dict.symbols[target.tolist()[0]])

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        if self.cfg.suggestion_type != "zero_context":
            target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]
        else:
            target_position = sample["target_position"]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [self.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, self.tgt_dict, self.symbol2pinyin if hasattr(self, "symbol2pinyin") else None).to(tgt_position_ids.device)

        # logits: (bsz * top_k, 1)
        # labels: (bsz * top_k)
        # top_k_indices: (bsz, top_k)
        logits, _, labels, top_k_indices, _ = model(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = self.tgt_dict,
                        mode = mode,
                        align_sents = sample["align_sents"],
                        align_positions = sample["align_positions"],
                        reranker_target = target,
                        epoch_idx = self.epoch_idx,
                        ids=ids,
                        candidates_file=self.cfg.candidates_file
        )   # ! add a brand new parameter "reranker_target".

        # logits: (bsz * top_k, 1) -> (bsz, top_k)
        logits = logits.squeeze_(-1).view(-1, top_k_indices.size(-1))
        # 后处理logits，将不以前缀开头的置为-inf
        if self.cfg.target_lang != "zh":
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                        logits[i,j] = -float('inf')
        else:
            # in chinese case
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.symbol2pinyin[ self.tgt_dict.symbols[top_k_indices[i][j]] ].startswith(types[i]):
                        logits[i,j] = -float('inf')            
        # target: (bsz * top_k) -> (bsz, top_k)
        # labels = labels.view(-1, top_k_indices.size(-1))

        # pred: (bsz,)
        preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
        n_correct = (preds == target).sum().item()
        # n_correct = logits.argmax(dim=-1).eq(0).sum().item()
        total = logits.size(0)

        # transfer index to symbol
        top_k_indices = top_k_indices.tolist()
        top_k_symbols = [ [ self.tgt_dict.symbols[index] for index in indices ] for indices in top_k_indices]

        # need more to return:
        # 1. predicted word
        # 2. predicted word's probability
        # 3. gold word
        # 4. gold word after index
        # 5. gold word's probability
        # 6. rank of gold word
        # 7. top k candidates
        # 8. top k candidates after rerank
        probs = logits.softmax(dim=-1)
        ########
        # print(probs.tolist())
        # print("Predict Result is :", True if n_correct == 1 else False)
        # print("----------------------------------------------------")
        ########
        pred_values, pred_indices = torch.max(probs, dim=-1)   
        pred_words =  [ self.tgt_dict.symbols[ top_k_indices[i][j] ] for i, j in enumerate(pred_indices.tolist()) ]
        pred_words_prob = pred_values.tolist()
        gold_words = [ self.datasets["suggestions"][id] for id in ids ]
        gold_words_after_index = [ self.tgt_dict.symbols[t] for t in target ]
        # gold word can not in the list
        gold_words_prob = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_prob.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_prob.append(probs[i,idx].item())
        # gold word can not in the list
        gold_words_rank = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_rank.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_rank.append(torch.argsort(probs, dim=-1, descending=True).tolist()[i].index(idx))        
        # gold_words_rank =  [ rank_lst.index(0) for rank_lst in torch.argsort(probs, dim=-1, descending=True).tolist() ]
        top_k_symbols_rerank = [ [ top_k_symbols[i][j] for j in rank_lst ] for i, rank_lst in enumerate(torch.argsort(probs, dim=-1, descending=True).tolist()) ]
        return n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank


    def rerank_generate_combine(self, models, sample, mode="infer"):
        """
        This is the infer generate function for RerankTransformer used in testing stage.
        """
        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        # print(ids)

        # get position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # get target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)
        # print(target.tolist())
        # print(self.tgt_dict.symbols[target.tolist()[0]])

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        if self.cfg.suggestion_type != "zero_context":
            target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]
        else:
            target_position = sample["target_position"]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [self.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, self.tgt_dict, self.symbol2pinyin if hasattr(self, "symbol2pinyin") else None).to(tgt_position_ids.device)

        # logits: (bsz * top_k, 1)
        # labels: (bsz * top_k)
        # top_k_indices: (bsz, top_k)
        logits, _, labels, top_k_indices, _, generator_top_k_logits = model.forward_combine(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = self.tgt_dict,
                        mode = mode,
                        align_sents = sample["align_sents"],
                        align_positions = sample["align_positions"],
                        reranker_target = target,
                        epoch_idx = self.epoch_idx
        )   # ! add a brand new parameter "reranker_target".

        # logits: (bsz * top_k, 1) -> (bsz, top_k)
        logits = logits.squeeze_(-1).view(-1, top_k_indices.size(-1))
        # 后处理logits，将不以前缀开头的置为-inf
        if self.cfg.target_lang != "zh":
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                        logits[i,j] = -float('inf')
        else:
            # in chinese case
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.symbol2pinyin[ self.tgt_dict.symbols[top_k_indices[i][j]] ].startswith(types[i]):
                        logits[i,j] = -float('inf')


        # 后处理generator_top_k_logits，将不以前缀开头的置为-inf
        if self.cfg.target_lang != "zh":
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                        generator_top_k_logits[i,j] = -float('inf')
        else:
            # in chinese case
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.symbol2pinyin[ self.tgt_dict.symbols[top_k_indices[i][j]] ].startswith(types[i]):
                        generator_top_k_logits[i,j] = -float('inf')             
        # target: (bsz * top_k) -> (bsz, top_k)
        # labels = labels.view(-1, top_k_indices.size(-1))

        # pred: (bsz,)
        reranker_probs = logits.softmax(dim=-1)
        generator_top_k_probs = generator_top_k_logits.softmax(dim=-1)
        combine_probs = reranker_probs + generator_top_k_probs
        preds = top_k_indices.gather( 1, combine_probs.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
        n_correct = (preds == target).sum().item()
        # n_correct = logits.argmax(dim=-1).eq(0).sum().item()
        total = logits.size(0)

        # transfer index to symbol
        top_k_indices = top_k_indices.tolist()
        top_k_symbols = [ [ self.tgt_dict.symbols[index] for index in indices ] for indices in top_k_indices]

        # need more to return:
        # 1. predicted word
        # 2. predicted word's probability
        # 3. gold word
        # 4. gold word after index
        # 5. gold word's probability
        # 6. rank of gold word
        # 7. top k candidates
        # 8. top k candidates after rerank
        probs = logits.softmax(dim=-1)
        ########
        # print(probs.tolist())
        # print("Predict Result is :", True if n_correct == 1 else False)
        # print("----------------------------------------------------")
        ########
        pred_values, pred_indices = torch.max(probs, dim=-1)   
        pred_words =  [ self.tgt_dict.symbols[ top_k_indices[i][j] ] for i, j in enumerate(pred_indices.tolist()) ]
        pred_words_prob = pred_values.tolist()
        gold_words = [ self.datasets["suggestions"][id] for id in ids ]
        gold_words_after_index = [ self.tgt_dict.symbols[t] for t in target ]
        # gold word can not in the list
        gold_words_prob = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_prob.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_prob.append(probs[i,idx].item())
        # gold word can not in the list
        gold_words_rank = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_rank.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_rank.append(torch.argsort(probs, dim=-1, descending=True).tolist()[i].index(idx))        
        # gold_words_rank =  [ rank_lst.index(0) for rank_lst in torch.argsort(probs, dim=-1, descending=True).tolist() ]
        top_k_symbols_rerank = [ [ top_k_symbols[i][j] for j in rank_lst ] for i, rank_lst in enumerate(torch.argsort(probs, dim=-1, descending=True).tolist()) ]
        return n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank


    def rerank_generate_save_attn(self, models, sample, mode="infer"):
        """
        This is the infer generate function for RerankTransformer used in testing stage.
        """
        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        if 399 in ids:
            print(ids)
        # print(ids)

        # get position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # get target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)
        # print(target.tolist())
        # print(self.tgt_dict.symbols[target.tolist()[0]])

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        if self.cfg.suggestion_type != "zero_context":
            target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]
        else:
            target_position = sample["target_position"]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [self.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, self.tgt_dict, self.symbol2pinyin if hasattr(self, "symbol2pinyin") else None).to(tgt_position_ids.device)

        # logits: (bsz * top_k, 1)
        # labels: (bsz * top_k)
        # top_k_indices: (bsz, top_k)
        logits, extra, labels, top_k_indices, _ , prev_output_tokens= model.forward_save_attn(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = self.tgt_dict,
                        mode = mode,
                        align_sents = sample["align_sents"],
                        align_positions = sample["align_positions"],
                        reranker_target = target,
                        epoch_idx = self.epoch_idx
        )   # ! add a brand new parameter "reranker_target".

        # logits: (bsz * top_k, 1) -> (bsz, top_k)
        logits = logits.squeeze_(-1).view(-1, top_k_indices.size(-1))
        # 后处理logits，将不以前缀开头的置为-inf
        if self.cfg.target_lang != "zh":
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                        logits[i,j] = -float('inf')
        else:
            # in chinese case
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.symbol2pinyin[ self.tgt_dict.symbols[top_k_indices[i][j]] ].startswith(types[i]):
                        logits[i,j] = -float('inf')            
        # target: (bsz * top_k) -> (bsz, top_k)
        # labels = labels.view(-1, top_k_indices.size(-1))

        # pred: (bsz,)
        preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
        n_correct = (preds == target).sum().item()
        # n_correct = logits.argmax(dim=-1).eq(0).sum().item()
        total = logits.size(0)

        # transfer index to symbol
        top_k_indices = top_k_indices.tolist()
        top_k_symbols = [ [ self.tgt_dict.symbols[index] for index in indices ] for indices in top_k_indices]

        # need more to return:
        # 1. predicted word
        # 2. predicted word's probability
        # 3. gold word
        # 4. gold word after index
        # 5. gold word's probability
        # 6. rank of gold word
        # 7. top k candidates
        # 8. top k candidates after rerank
        probs = logits.softmax(dim=-1)
        ########
        # print(probs.tolist())
        # print("Predict Result is :", True if n_correct == 1 else False)
        # print("----------------------------------------------------")
        ########
        pred_values, pred_indices = torch.max(probs, dim=-1)   
        pred_words =  [ self.tgt_dict.symbols[ top_k_indices[i][j] ] for i, j in enumerate(pred_indices.tolist()) ]
        pred_words_prob = pred_values.tolist()
        gold_words = [ self.datasets["suggestions"][id] for id in ids ]
        gold_words_after_index = [ self.tgt_dict.symbols[t] for t in target ]
        # gold word can not in the list
        gold_words_prob = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_prob.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_prob.append(probs[i,idx].item())
        # gold word can not in the list
        gold_words_rank = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_rank.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_rank.append(torch.argsort(probs, dim=-1, descending=True).tolist()[i].index(idx))        
        # gold_words_rank =  [ rank_lst.index(0) for rank_lst in torch.argsort(probs, dim=-1, descending=True).tolist() ]
        top_k_symbols_rerank = [ [ top_k_symbols[i][j] for j in rank_lst ] for i, rank_lst in enumerate(torch.argsort(probs, dim=-1, descending=True).tolist()) ]

        # 输出attention文件
        attn = extra["attn"][0]
        if self.cfg.inference_save_attn is True:
            self.save_attention(
                            ids=ids,
                            attn=attn,
                            src_tokens=sample["net_input"]["src_tokens"],
                            prev_output_tokens=prev_output_tokens, # 采用ground_truth的输出结果
            )
        return n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank

    def rerank_tag_generate(self, models, sample, mode="infer"):
        """
        This is the infer generate function for RerankTransformer used in testing stage.
        """
        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        # print(ids)

        # get position ids
        # tgt_position_ids: (bsz, seq_len)
        # ! Modify here, we append <s> to sentence
        # sample["net_input"][]
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # get target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)
        # print(target.tolist())
        # print(self.tgt_dict.symbols[target.tolist()[0]])

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        if self.cfg.suggestion_type != "zero_context":
            target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]
        else:
            target_position = sample["target_position"]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [self.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, self.tgt_dict, self.symbol2pinyin if hasattr(self, "symbol2pinyin") else None).to(tgt_position_ids.device)

        generator_bi_context_input_tokens = sample["net_input"]["prev_output_tokens"]
        generator_bi_context_position = tgt_position_ids
        generator_target_position = target_position

        prev_output_tokens_list = []
        tgt_position_ids_list = []
        target_postion_list = (target_position+1).tolist()
        for i, (input_token_list, p) in enumerate(zip(sample["net_input"]["prev_output_tokens"].tolist(), target_position.tolist())):
            insert_input_token_list = input_token_list
            insert_input_token_list.insert(p, self.tgt_dict.index("<s>"))
            insert_input_token_list.insert(p+2, self.tgt_dict.index("<s>"))
            prev_output_tokens_list.append(insert_input_token_list)
            tgt_position_id = []
            for j, token_id in enumerate(insert_input_token_list):
                if token_id != self.tgt_dict.index("<pad>"):
                    tgt_position_id.append(j+2)
                else:
                    tgt_position_id.append(1)
            tgt_position_ids_list.append(tgt_position_id)
            # for input_token in input_token_list:
            #     if input_token != self.tgt_dict.mask():
            #         insert_input_token_list.append( self.tgt_dict.index("<s>") )
            #         insert_input_token_list.append( self.)
        prev_output_tokens = target.new(prev_output_tokens_list) 
        tgt_position_ids = target.new(tgt_position_ids_list)
        target_position = target.new(target_postion_list)

        # logits: (bsz * top_k, 1)
        # labels: (bsz * top_k)
        # top_k_indices: (bsz, top_k)
        # TODO: fix non-bi_zero_context error
        logits, _, labels, top_k_indices, _ = model(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = prev_output_tokens if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = self.tgt_dict,
                        mode = mode,
                        align_sents = sample["align_sents"],
                        align_positions = sample["align_positions"],
                        reranker_target = target,
                        epoch_idx = self.epoch_idx,
                        generator_bi_context_position = generator_bi_context_position,
                        generator_bi_context_input_tokens = generator_bi_context_input_tokens,
                        generator_target_position = generator_target_position
        )   # ! add a brand new parameter "reranker_target".

        # logits: (bsz * top_k, 1) -> (bsz, top_k)
        logits = logits.squeeze_(-1).view(-1, top_k_indices.size(-1))
        # 后处理logits，将不以前缀开头的置为-inf
        if self.cfg.target_lang != "zh":
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                        logits[i,j] = -float('inf')
        else:
            # in chinese case
            for i in range(len(top_k_indices.tolist())):
                for j in range(len(top_k_indices[i])):
                    if not self.symbol2pinyin[ self.tgt_dict.symbols[top_k_indices[i][j]] ].startswith(types[i]):
                        logits[i,j] = -float('inf')            
        # target: (bsz * top_k) -> (bsz, top_k)
        # labels = labels.view(-1, top_k_indices.size(-1))

        # pred: (bsz,)
        preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
        n_correct = (preds == target).sum().item()
        # n_correct = logits.argmax(dim=-1).eq(0).sum().item()
        total = logits.size(0)

        # transfer index to symbol
        top_k_indices = top_k_indices.tolist()
        top_k_symbols = [ [ self.tgt_dict.symbols[index] for index in indices ] for indices in top_k_indices]

        # need more to return:
        # 1. predicted word
        # 2. predicted word's probability
        # 3. gold word
        # 4. gold word after index
        # 5. gold word's probability
        # 6. rank of gold word
        # 7. top k candidates
        # 8. top k candidates after rerank
        probs = logits.softmax(dim=-1)
        ########
        # print(probs.tolist())
        # print("Predict Result is :", True if n_correct == 1 else False)
        # print("----------------------------------------------------")
        ########
        pred_values, pred_indices = torch.max(probs, dim=-1)   
        pred_words =  [ self.tgt_dict.symbols[ top_k_indices[i][j] ] for i, j in enumerate(pred_indices.tolist()) ]
        pred_words_prob = pred_values.tolist()
        gold_words = [ self.datasets["suggestions"][id] for id in ids ]
        gold_words_after_index = [ self.tgt_dict.symbols[t] for t in target ]
        # gold word can not in the list
        gold_words_prob = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_prob.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_prob.append(probs[i,idx].item())
        # gold word can not in the list
        gold_words_rank = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_rank.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_rank.append(torch.argsort(probs, dim=-1, descending=True).tolist()[i].index(idx))        
        # gold_words_rank =  [ rank_lst.index(0) for rank_lst in torch.argsort(probs, dim=-1, descending=True).tolist() ]
        top_k_symbols_rerank = [ [ top_k_symbols[i][j] for j in rank_lst ] for i, rank_lst in enumerate(torch.argsort(probs, dim=-1, descending=True).tolist()) ]
        return n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank
        

    # def rerank_generate(self, models, sample):
    #     """
    #     This is the generate function for RerankTransformer.
    #     """
    #     # get model from models
    #     model = models[0]

    #     ids = sample["id"].tolist()
        
    #     # get new position ids
    #     # tgt_position_ids: (bsz, seq_len)
    #     tgt_position_ids =  utils.make_positions(
    #             sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
    #     )

    #     # get new target
    #     # (bsz,)
    #     # We should get this from .suggestion file
    #     target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)

    #     # get new target_position
    #     # (bsz,)
    #     # This time, it is a tensor, not a scalar
    #     # find every <mask>'s position in prev_output_tokens
    #     target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]

    #     # get new type_masks
    #     # (bsz, vocab_size)
    #     # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
    #     types = [self.datasets["types"][id] for id in ids]
    #     type_masks = get_type_masks(types, self.tgt_dict).to(tgt_position_ids.device)

    #     # logits: (bsz * top_k, 1)
    #     # labels: (bsz * top_k)
    #     logits, _, labels, top_k_indices = model(
    #                     src_tokens = sample["net_input"]["src_tokens"],
    #                     src_lengths = sample["net_input"]["src_lengths"],
    #                     prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
    #                     tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
    #                     target_position = target_position,
    #                     target = target,
    #                     type_masks = type_masks,
    #                     tgt_dict = self.tgt_dict,
    #                     mode="infer",
    #     )

    #     # logits: (bsz * top_k, 1) -> (bsz, top_k)
    #     logits = logits.squeeze_(-1).view(-1, top_k_indices.size(-1))
    #     # 后处理logits，将不以前缀开头的置为-inf
    #     for i in range(len(top_k_indices.tolist())):
    #         for j in range(len(top_k_indices[i])):
    #             if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
    #                 logits[i,j] = -float('inf')
    #     # target: (bsz * top_k) -> (bsz, top_k)
    #     labels = labels.view(-1, top_k_indices.size(-1))

    #     # preds: (bsz,)
    #     preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
    #     n_correct = (preds == target).sum().item()
    #     total = logits.size(0)     
        
    #     return n_correct, total

    def rerank_mask_inner_product_generate(self, models, sample):
        """
        This is the mask inner product generate function for RerankTransformer.
        """
        # get model from models
        model = models[0]

        ids = sample["id"].tolist()
        
        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # get new target
        # (bsz,)
        # We should get this from .suggestion file
        target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)

        # get new target_position
        # (bsz,)
        # This time, it is a tensor, not a scalar
        # find every <mask>'s position in prev_output_tokens
        target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]

        # get new type_masks
        # (bsz, vocab_size)
        # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
        types = [self.datasets["types"][id] for id in ids]
        type_masks = get_type_masks(types, self.tgt_dict, self.symbol2pinyin).to(tgt_position_ids.device)

        # logits: (bsz * (top_k+1), 1)
        # labels: (bsz * (top_k+1))
        net_output = model(
                        src_tokens = sample["net_input"]["src_tokens"],
                        src_lengths = sample["net_input"]["src_lengths"],
                        prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
                        target_position = target_position,
                        target = target,
                        type_masks = type_masks,
                        tgt_dict = self.tgt_dict,
                        mode="infer",
        )

        # logits: (bsz * (top_k+1), 1) -> (bsz * (top_k+1)) -> (bsz, top_k+1)
        logits = net_output[0].squeeze_(-1)
        logits = logits.view(-1, model.cfg.top_k+1)
        bsz = logits.size(0)

        # (bsz * (top_k+1), tgt_len, hidden_dim) -> (bsz, top_k+1, tgt_len, hidden_dim)
        last_hidden_states = net_output[1]["inner_states"][-1].transpose(0, 1)
        last_hidden_states = last_hidden_states.reshape( bsz, model.cfg.top_k+1, last_hidden_states.size(1), last_hidden_states.size(2) )

        anchor_position = target_position

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

        # norm 2
        mask_hidden_states = mask_hidden_states / mask_hidden_states.norm(dim=2, keepdim=True)
        top_k_hidden_states = top_k_hidden_states / top_k_hidden_states.norm(dim=2, keepdim=True)

        temperature = 2e-2
        # (bsz, 1, top_k) -> (bsz, top_k)
        contrastive_score = torch.matmul(mask_hidden_states, top_k_hidden_states.transpose(1,2)) / temperature
        contrastive_score = contrastive_score.squeeze(1)
        logits = contrastive_score

        top_k_indices = net_output[3][:,:-1]    # don't take the last one

        # get the largest one
        # 后处理logits，将不以前缀开头的置为-inf
        for i in range(len(top_k_indices.tolist())):
            for j in range(len(top_k_indices[i])):
                if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
                    logits[i,j] = -float('inf')        
        
        # preds: (bsz, )
        preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
        n_correct = (preds == target).sum().item()
        total = logits.size(0)        

        # transfer index to symbol
        top_k_indices = top_k_indices.tolist()
        top_k_symbols = [ [ self.tgt_dict.symbols[index] for index in indices ] for indices in top_k_indices]

        # need more to return:
        # 1. predicted word
        # 2. predicted word's probability
        # 3. gold word
        # 4. gold word after index
        # 5. gold word's probability
        # 6. rank of gold word
        # 7. top k candidates
        # 8. top k candidates after rerank
        probs = logits.softmax(dim=-1)
        pred_values, pred_indices = torch.max(probs, dim=-1)   
        pred_words =  [ self.tgt_dict.symbols[ top_k_indices[i][j] ] for i, j in enumerate(pred_indices.tolist()) ]
        pred_words_prob = pred_values.tolist()

        gold_words = [ self.datasets["suggestions"][id] for id in ids ]
        gold_words_after_index = [ self.tgt_dict.symbols[t] for t in target ]
        # TODO: Change this bug
        # gold word can not in the list
        gold_words_prob = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_prob.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_prob.append(probs[i,idx].item())
        # gold word can not in the list
        gold_words_rank = []
        for i, (rank_lst, t) in enumerate(zip(top_k_indices, target.tolist())):
            if t not in rank_lst:
                gold_words_rank.append(-1)
            else:
                idx = rank_lst.index(t)
                gold_words_rank.append(torch.argsort(probs, dim=-1, descending=True).tolist()[i].index(idx))        
        
        top_k_symbols_rerank = [ [ top_k_symbols[i][j] for j in rank_lst ] for i, rank_lst in enumerate(torch.argsort(probs, dim=-1, descending=True).tolist()) ]
        
        return n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank           

    # def rerank_mask_inner_product_generate(self, models, sample):
    #     """
    #     This is the mask inner product generate function for RerankTransformer.
    #     """
    #     # get model from models
    #     model = models[0]

    #     ids = sample["id"].tolist()
        
    #     # get new position ids
    #     # tgt_position_ids: (bsz, seq_len)
    #     tgt_position_ids =  utils.make_positions(
    #             sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
    #     )

    #     # get new target
    #     # (bsz,)
    #     # We should get this from .suggestion file
    #     target = torch.tensor([ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]).to(tgt_position_ids.device)

    #     # get new target_position
    #     # (bsz,)
    #     # This time, it is a tensor, not a scalar
    #     # find every <mask>'s position in prev_output_tokens
    #     target_position = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()).nonzero()[:,-1]

    #     # get new type_masks
    #     # (bsz, vocab_size)
    #     # We can import `get_type_masks` from `gwlan_rerank_dataset.py` to do this.
    #     types = [self.datasets["types"][id] for id in ids]
    #     type_masks = get_type_masks(types, self.tgt_dict).to(tgt_position_ids.device)

    #     # logits: (bsz * (top_k+1), 1)
    #     # labels: (bsz * (top_k+1))
    #     net_output = model(
    #                     src_tokens = sample["net_input"]["src_tokens"],
    #                     src_lengths = sample["net_input"]["src_lengths"],
    #                     prev_output_tokens = sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
    #                     tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"],
    #                     target_position = target_position,
    #                     target = target,
    #                     type_masks = type_masks,
    #                     tgt_dict = self.tgt_dict,
    #                     mode="infer",
    #     )

    #     # logits: (bsz * (top_k+1), 1) -> (bsz * (top_k+1)) -> (bsz, top_k+1)
    #     logits = net_output[0].squeeze_(-1)
    #     logits = logits.view(-1, model.cfg.top_k+1)
    #     bsz = logits.size(0)

    #     # (bsz * (top_k+1), tgt_len, hidden_dim) -> (bsz, top_k+1, tgt_len, hidden_dim)
    #     last_hidden_states = net_output[1]["inner_states"][-1].transpose(0, 1)
    #     last_hidden_states = last_hidden_states.reshape( bsz, model.cfg.top_k+1, last_hidden_states.size(1), last_hidden_states.size(2) )

    #     anchor_position = target_position

    #     # mask_anchor_position: ( bsz, ) -> (bsz, 1, 1, hidden_dim)
    #     mask_anchor_position = anchor_position.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #     mask_anchor_position = mask_anchor_position.expand(-1, -1, -1, last_hidden_states.size(-1))
    #     # mask_hidden_states: (bsz, 1, hidden_dim)
    #     mask_hidden_states = last_hidden_states[:, -1:, :, :].gather(2, mask_anchor_position).squeeze(2)

    #     # top_k_anchor_position: ( bsz, ) -> (bsz, top_k, 1, hidden_dim)
    #     top_k_anchor_position = anchor_position.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #     top_k_anchor_position = top_k_anchor_position.expand(-1, last_hidden_states.size(1)-1, -1, last_hidden_states.size(-1))
    #     # top_k_hidden_states: (bsz, top_k, hidden_dim)
    #     top_k_hidden_states = last_hidden_states[:, :-1, :, :].gather(2, top_k_anchor_position).squeeze(2)

    #     # norm 2
    #     mask_hidden_states = mask_hidden_states / mask_hidden_states.norm(dim=2, keepdim=True)
    #     top_k_hidden_states = top_k_hidden_states / top_k_hidden_states.norm(dim=2, keepdim=True)

    #     temperature = 2e-2
    #     # (bsz, 1, top_k) -> (bsz, top_k)
    #     contrastive_score = torch.matmul(mask_hidden_states, top_k_hidden_states.transpose(1,2)) / temperature
    #     contrastive_score = contrastive_score.squeeze(1)
    #     logits = contrastive_score

    #     top_k_indices = net_output[3][:,:-1]    # don't take the last one

    #     # get the largest one
    #     # 后处理logits，将不以前缀开头的置为-inf
    #     for i in range(len(top_k_indices.tolist())):
    #         for j in range(len(top_k_indices[i])):
    #             if not self.tgt_dict.symbols[top_k_indices[i][j]].startswith(types[i]):
    #                 logits[i,j] = -float('inf')        
        
    #     # preds: (bsz, )
    #     preds = top_k_indices.gather( 1, logits.argmax(dim=-1).unsqueeze(-1) ).squeeze(-1)
    #     n_correct = (preds == target).sum().item()
    #     total = logits.size(0)

    #     return n_correct, total    

    def generate(self, models, sample):
        # get model from models
        model = models[0]
        
        # call model forward

        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        tgt_position_ids =  utils.make_positions(
                sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        )

        # net_output: (bsz, seq_len, tgt_vocab_size)
        net_output = model(
                        src_tokens=sample["net_input"]["src_tokens"],
                        src_lengths=sample["net_input"]["src_lengths"],
                        prev_output_tokens=sample["net_input"]["prev_output_tokens"] if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"]
        )[0]

        # 根据prev_output_tokens里面是否为<mask>来决定当前掩码的目标
        # acc_mask : (bsz, seq_len)
        acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()) if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"].eq(self.tgt_dict.mask()) # 只有是<mask>才会读取，之前写的是<eos>。见gwlan_dataset中的描述
        # net_output : (bsz, tgt_vocab_size)
        net_output = net_output[acc_mask]
        # get sample id
        ids = sample["id"].tolist()
        types = [self.datasets["types"][id] for id in ids]
        
        if self.cfg.target_lang != "zh":
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = [ not symbol.startswith(type) for symbol in self.tgt_dict.indices.keys()]
                type_masks.append(type_mask)
            # 变成tensor
            type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
        else:
            type_masks = []
            # 查看词表进行概率掩码
            for type in types:
                type_mask = []
                for symbol in self.tgt_dict.indices.keys():
                    symbol_pinyin = self.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
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
        preds = torch.max(net_output, dim=-1).indices.cpu().numpy()
        
        # ground truth labels
        labels = [ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]
        # labels: (bsz, )
        labels = np.array(labels)
        
        true = sum(labels == preds)
        size = preds.shape[0]
        return labels, preds

    def hw_generate(self, models, sample):
        # get model from models
        model = models[0]
        
        # call model forward

        # get new position ids
        # tgt_position_ids: (bsz, seq_len)
        # tgt_position_ids =  utils.make_positions(
        #         sample["net_input"]["prev_output_tokens"], self.tgt_dict.pad()
        # )
        # acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()) if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_input_tokens"].eq(self.tgt_dict.mask())             
        # 得到相应的<mask>的位置，然后插入前缀
        poses = []
        for i in range(sample["net_input"]["prev_output_tokens"].size(0)):
            for j in range(sample["net_input"]["prev_output_tokens"].size(1)):
                if sample["net_input"]["prev_output_tokens"][i][j] == self.tgt_dict.mask():
                    poses.append(j)
        
        # 插入相应的用户输入的前缀信息
        ids = sample["id"].tolist()
        types = [self.datasets["types"][id] for id in ids]
        new_prev_output_tokens = []
        old_prev_output_tokens = sample["net_input"]["prev_output_tokens"].tolist()
        for idx, typed_chars in enumerate(types):
            prev_output_token = old_prev_output_tokens[idx]
            prev_output_token.insert(poses[idx], self.tgt_dict.bos_index)
            
            for character in typed_chars[::-1]:
               prev_output_token.insert( poses[idx], self.tgt_dict.index(character) )
            prev_output_token.insert(poses[idx], self.tgt_dict.bos_index) 
            new_prev_output_tokens.append( torch.tensor(prev_output_token, dtype=torch.int64))
        
        new_prev_output_tokens = torch.nn.utils.rnn.pad_sequence(new_prev_output_tokens, batch_first=True, padding_value=self.tgt_dict.pad_index)
        sample["net_input"]["prev_output_tokens"] = new_prev_output_tokens.to(sample["net_input"]["prev_output_tokens"])

        # net_output: (bsz, seq_len, tgt_vocab_size)
        net_output = model(
                        src_tokens=sample["net_input"]["src_tokens"],
                        src_lengths=sample["net_input"]["src_lengths"],
                        prev_output_tokens=sample["net_input"]["prev_output_tokens"], # 采用ground_truth的输出结果
                        tgt_position_ids = None # tgt_position_ids if self.cfg.suggestion_type != "zero_context" else sample["zero_context_net_input"]["zero_context_position"]
        )[0]

        # 根据prev_output_tokens里面是否为<mask>来决定当前掩码的目标
        # acc_mask : (bsz, seq_len)
        acc_mask = sample["net_input"]["prev_output_tokens"].eq(self.tgt_dict.mask()) # 只有是<mask>才会读取，之前写的是<eos>。见gwlan_dataset中的描述
        # net_output : (bsz, tgt_vocab_size)
        net_output = net_output[acc_mask]
        # get sample id
        ids = sample["id"].tolist()
        types = [self.datasets["types"][id] for id in ids]
        
        # if self.cfg.target_lang != "zh":
        #     type_masks = []
        #     # 查看词表进行概率掩码
        #     for type in types:
        #         type_mask = [ not symbol.startswith(type) for symbol in self.tgt_dict.indices.keys()]
        #         type_masks.append(type_mask)
        #     # 变成tensor
        #     type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)
        # else:
        #     type_masks = []
        #     # 查看词表进行概率掩码
        #     for type in types:
        #         type_mask = []
        #         for symbol in self.tgt_dict.indices.keys():
        #             symbol_pinyin = self.symbol2pinyin[symbol] #"".join(lazy_pinyin(symbol))
        #             if symbol_pinyin.startswith(type):
        #                 type_mask.append(False)
        #             else:
        #                 type_mask.append(True)
        #         type_masks.append(type_mask)
        #     # 变成tensor
        #     type_masks = torch.tensor(type_masks, dtype=torch.bool).to(net_output.device)            

        # # 修改概率
        # # net_output: (bsz, tgt_vocab_size)
        # net_output = net_output.masked_fill(type_masks, -float('inf'))
        # 取最大的概率
        preds = torch.max(net_output, dim=-1).indices.cpu().numpy()
        
        # ground truth labels
        labels = [ self.tgt_dict.index(self.datasets["suggestions"][id]) for id in ids ]
        # labels: (bsz, )
        labels = np.array(labels)
        
        true = sum(labels == preds)
        size = preds.shape[0]
        return labels, preds


    def nat_generate(self, models, sample):
        model = models[0]
        if self.cfg.suggestion_type == "zero_context":
            sample["net_input"]["prev_output_tokens"] = torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.mask()).to(sample["net_input"]["prev_output_tokens"])
            sample["net_input"]["prev_output_tokens"] = torch.cat( 
                                        [ sample["net_input"]["prev_output_tokens"] , torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.eos()).to(sample["net_input"]["prev_output_tokens"]) ], 
                                        dim = -1
            )
        # Procedure 0: Encoder Forward
        encoder_out = model.encoder(
            src_tokens = sample["net_input"]["src_tokens"],
            src_lengths = sample["net_input"]["src_lengths"],
            return_all_hiddens = True
        )

        # Procedure 1: Length Prediction
        length_out = model.forward_length(encoder_out)

        # 推理出每个句子的长度。
        length_pred = (length_out.argmax(1) + sample["net_input"]["src_lengths"] - 128).tolist()

        # 重新组装prev_output_tokens
        new_prev_output_tokens = []
        for idx in range(sample["net_input"]["prev_output_tokens"].size(0)):
            sent = sample["net_input"]["prev_output_tokens"][idx].tolist()
            if self.tgt_dict.pad() in sent:
                pad_pos = sent.index( self.tgt_dict.pad() )
                sent = sent[:pad_pos]
            else:
                pass
            # 找到<mask>的位置
            mask_idx = sent.index(self.tgt_dict.mask())
            length = length_pred[idx]
            if length <= len(sent):
                pass
            else:
                for _ in range(length-len(sent)):
                    sent.insert(mask_idx, self.tgt_dict.mask()) 
            new_prev_output_tokens.append(torch.tensor(sent).to(sample["net_input"]["prev_output_tokens"]))
        prev_output_tokens = pad_sequence(new_prev_output_tokens, batch_first=True, padding_value=self.tgt_dict.pad())
        
        # Procedure 2: Decoder Forward
        decoder_out = model.decoder(
            prev_output_tokens,
            None,
            encoder_out=encoder_out,
            features_only=False,
            alignment_layer=None,
            alignment_heads=None,
            src_lengths=sample["net_input"]["src_lengths"],
            return_all_hiddens=True,
        )       

        # Procedure 3: Marginalize probablity
        # 得到所有mask位置概率，然后进行marginalized
        logits = decoder_out[0]
        correct = 0
        count = 0
        for idx in range(prev_output_tokens.size(0)):
            probs = []
            sample_id = sample["id"][idx]
            type = self.datasets["types"][sample_id]
            suggestion = self.datasets["suggestions"][sample_id]

            unobserved_token_logits = logits[idx][ prev_output_tokens[idx].eq(self.tgt_dict.mask()) ]
            # 调用函数，遍历所有的unoberved_tokens，以及整个词表的单词
            prob_matrix = unobserved_token_logits.softmax(-1).tolist()
            for cand_id, cand_word in enumerate(self.tgt_dict.symbols):
                if not cand_word.startswith(type):
                    probs.append(-float("inf"))
                else:
                    prob = self.naive_algo(word=cand_word, prob_matrix=prob_matrix)
                    probs.append(prob)
            # 进行排序
            probs = torch.tensor(probs)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)      
            if self.tgt_dict.symbols[sorted_indices[0]] == suggestion:
                correct += 1
            count += 1

        return correct, count 

    def nat_length_ensemble_generate(self, models, sample):
        model = models[0]
        if self.cfg.suggestion_type == "zero_context":
            sample["net_input"]["prev_output_tokens"] = torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.mask()).to(sample["net_input"]["prev_output_tokens"])
            sample["net_input"]["prev_output_tokens"] = torch.cat( 
                                        [ sample["net_input"]["prev_output_tokens"] , torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.eos()).to(sample["net_input"]["prev_output_tokens"]) ], 
                                        dim = -1
            )
        # Procedure 0: Encoder Forward
        encoder_out = model.encoder(
            src_tokens = sample["net_input"]["src_tokens"],
            src_lengths = sample["net_input"]["src_lengths"],
            return_all_hiddens = True
        )

        # Procedure 1: Length Prediction
        length_out = model.forward_length(encoder_out)

        # 推理出每个句子的长度。
        length_pred = (length_out.argmax(1) + sample["net_input"]["src_lengths"] - 128).tolist()

        # 重新组装prev_output_tokens
        len_beam = self.cfg.len_beam
        pred_probs = torch.full(size=(length_out.size(0),len(self.tgt_dict.symbols)), fill_value=0, dtype=torch.float).to(sample["net_input"]["prev_output_tokens"].device)
        
        for i in range(len_beam):
            len_cand = i-((len_beam-1)//2)
            self.nat_decoder_forward_with_len_cand(model, sample, len_cand, length_pred, encoder_out, pred_probs)

        sorted_probs, sorted_indices = torch.sort(pred_probs, descending=True)
        correct = 0
        count = 0
        for idx in range(sample["net_input"]["prev_output_tokens"].size(0)):
            sample_id = sample["id"][idx]
            type = self.datasets["types"][sample_id]
            suggestion = self.datasets["suggestions"][sample_id]
            if self.tgt_dict.symbols[ sorted_indices[idx][0] ] == suggestion:
                correct += 1
            count += 1
        return correct, count, sorted_probs, sorted_indices        

    def nat_decoder_forward_with_len_cand(self, model, sample, len_cand, length_pred, encoder_out, pred_probs):
        new_prev_output_tokens = []
        for idx in range( sample["net_input"]["prev_output_tokens"].size(0) ):
            sent = sample["net_input"]["prev_output_tokens"][idx].tolist()
            if self.tgt_dict.pad() in sent:
                pad_pos = sent.index( self.tgt_dict.pad() )
                sent = sent[:pad_pos]
            else:
                pass
            # 找到<mask>的位置
            mask_idx = sent.index(self.tgt_dict.mask())
            length = length_pred[idx]
            if (length+len_cand) <= len(sent):
                pass
            else:
                for _ in range(length+len_cand - len(sent)):
                    sent.insert(mask_idx, self.tgt_dict.mask())
            new_prev_output_tokens.append(torch.tensor(sent).to(sample["net_input"]["prev_output_tokens"]))
        prev_output_tokens = pad_sequence(new_prev_output_tokens, batch_first=True, padding_value=self.tgt_dict.pad())
        
        # Procedure 2: Decoder Forward
        decoder_out = model.decoder(
            prev_output_tokens,
            None,
            encoder_out=encoder_out,
            features_only=False,
            alignment_layer=None,
            alignment_heads=None,
            src_lengths=sample["net_input"]["src_lengths"],
            return_all_hiddens=True,
        )

        # Procedure 3: Marginalize probablity
        # 得到所有mask位置概率，然后进行marginalized
        logits = decoder_out[0]
        for idx in range(prev_output_tokens.size(0)):
            probs = []
            sample_id = sample["id"][idx]
            type = self.datasets["types"][sample_id]

            unobserved_token_logits = logits[idx][ prev_output_tokens[idx].eq(self.tgt_dict.mask()) ]
            # 调用函数，遍历所有的unoberved_tokens，以及整个词表的单词
            prob_matrx = unobserved_token_logits.softmax(-1).tolist()
            for cand_id, cand_word in enumerate(self.tgt_dict.symbols):
                if not cand_word.startswith(type):
                    probs.append(-float("inf"))
                else:
                    prob = self.naive_algo(word=cand_word, prob_matrix=prob_matrx)
                    probs.append(prob)
            probs = torch.tensor(probs).to(pred_probs)
            pred_probs[idx] += probs


    def naive_algo(self, word, prob_matrix):
        word_idx = self.tgt_dict.index(word)
        sum = 1
        for pos in range(len(prob_matrix)):
            sum *= (1 - prob_matrix[pos][word_idx])
        return 1 - sum

    def dp_algo_prefix(self, context, prob_matrix, tgt_dict):
        tgt_len = len(prob_matrix)

        context = context.split(" ")    # context is a list
        context = ["<s>"] + context
        context = [ tgt_dict.index(word) for word in context ]
        c_len = len(context) - 1

        # 初始化dp[0][j],在位置0上，包含前j个子序列，但是不包含后面的子序列的组合数。
        dp = [ [0 for _ in range(c_len+1)] for _ in range(tgt_len+1)]

        dp[0][0] = 1
        for i in range(1, tgt_len+1):
            for j in range(0, c_len+1):
                if j == 0:
                    dp[i][j] = 1
                    for pos in range(1, i+1):
                        dp[i][j] = dp[i][j] * (1 - prob_matrix[pos-1][context[1]]) # pow(vocab_size-1,i)
                elif i < j:
                    dp[i][j] = 0
                else:
                    if j == c_len:
                        # dp[i][j] = (dp[i-1][j-1] * 1) + dp[i-1][j] * (vocab_size) 
                        dp[i][j] = ( dp[i-1][j-1] * prob_matrix[i-1][context[j]] ) + dp[i-1][j]
                    else:
                        # dp[i][j] = (dp[i-1][j-1] * 1) + dp[i-1][j] * (vocab_size-1)
                        dp[i][j] = ( dp[i-1][j-1] * prob_matrix[i-1][context[j]] ) + dp[i-1][j] * (1 - prob_matrix[i-1][context[j+1]])

        return dp[tgt_len][c_len]

    def nat_sc_generate(self, models, sample):
        model = models[0]
        if self.cfg.suggestion_type == "zero_context":
            sample["net_input"]["prev_output_tokens"] = torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.mask()).to(sample["net_input"]["prev_output_tokens"])
            sample["net_input"]["prev_output_tokens"] = torch.cat( 
                                        [ sample["net_input"]["prev_output_tokens"] , torch.full((sample["net_input"]["prev_output_tokens"].size(0), 1), self.tgt_dict.eos()).to(sample["net_input"]["prev_output_tokens"]) ], 
                                        dim = -1
            )
        # Procedure 0: Encoder Forward
        encoder_out = model.encoder(
            src_tokens = sample["net_input"]["src_tokens"],
            src_lengths = sample["net_input"]["src_lengths"],
            return_all_hiddens = True,
            segment_labels = sample["segment_labels"]
        )

        # Procedure 1: Length Prediction
        length_out = model.forward_length(encoder_out)

        # 推理出每个句子的长度。
        length_pred = (length_out.argmax(1) + sample["net_input"]["src_lengths"] - 128).tolist()

        # 重新组装prev_output_tokens
        new_prev_output_tokens = []
        for idx in range(sample["net_input"]["prev_output_tokens"].size(0)):
            length = length_pred[idx]
            sent = [self.tgt_dict.mask()] * length + [self.tgt_dict.eos()]  # TODO: Bugs may exist.
            # sent = sample["net_input"]["prev_output_tokens"][idx].tolist()
            # if self.tgt_dict.pad() in sent:
            #     pad_pos = sent.index( self.tgt_dict.pad() )
            #     sent = sent[:pad_pos]
            # else:
            #     pass
            # # 找到<mask>的位置
            # mask_idx = sent.index(self.tgt_dict.mask())
            # length = length_pred[idx]
            # if length <= len(sent):
            #     pass
            # else:
            #     for _ in range(length-len(sent)):
            #         sent.insert(mask_idx, self.tgt_dict.mask()) 
            new_prev_output_tokens.append(torch.tensor(sent).to(sample["net_input"]["prev_output_tokens"]))
        prev_output_tokens = pad_sequence(new_prev_output_tokens, batch_first=True, padding_value=self.tgt_dict.pad())
        
        # Procedure 2: Decoder Forward
        decoder_out = model.decoder(
            prev_output_tokens,
            None,
            encoder_out=encoder_out,
            features_only=False,
            alignment_layer=None,
            alignment_heads=None,
            src_lengths=sample["net_input"]["src_lengths"],
            return_all_hiddens=True,
        )       

        # Procedure 3: Marginalize probablity
        # 得到所有mask位置概率，然后进行marginalized
        logits = decoder_out[0]
        correct = 0
        count = 0
        for idx in range(prev_output_tokens.size(0)):
            probs = []
            sample_id = sample["id"][idx]
            type = self.datasets["types"][sample_id]
            suggestion = self.datasets["suggestions"][sample_id]

            unobserved_token_logits = logits[idx][ prev_output_tokens[idx].eq(self.tgt_dict.mask()) ]
            # 调用函数，遍历所有的unoberved_tokens，以及整个词表的单词
            prob_matrix = unobserved_token_logits.softmax(-1).tolist()
            for cand_id, cand_word in enumerate(self.tgt_dict.symbols):
                if not cand_word.startswith(type):
                    probs.append(-float("inf"))
                else:
                    prob = self.naive_algo(word=cand_word, prob_matrix=prob_matrix)
                    probs.append(prob)
            # 进行排序
            probs = torch.tensor(probs)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)      
            if self.tgt_dict.symbols[sorted_indices[0]] == suggestion:
                correct += 1
            count += 1

        return correct, count

    def save_attention(self, ids, attn, src_tokens, prev_output_tokens):
        """该函数用于在inference_generate中保存attention的结果
        """
        attention_dir = os.path.join(self.cfg.results_path, "attention-{}.pt".format(self.cfg.gen_subset))
        if os.path.exists(attention_dir):
            attention_dict = torch.load(attention_dir)
        else:
            attention_dict = {}
        for i in range(len(ids)):
            id = ids[i]
            src_sent = [ self.src_dict.symbols[token] for token in src_tokens[i].tolist() ]
            tgt_sent = [ self.tgt_dict.symbols[token] for token in prev_output_tokens[i].tolist() ]
            cross_attn = attn[i]
            # constrained_ids = {"zero_context": 54, "bi_context": 299, "prefix": 756, "suffix": 687}
            # if id == constrained_ids[self.cfg.suggestion_type]:
            attention_dict[id+1] = (src_sent, tgt_sent, cross_attn) # id从1开始
        torch.save(attention_dict, attention_dir)
        return

    def load_eval_dataset(self, split, supply_path):
        """Load .suggestion and .type file for evaluation"""
        if "/shared_info/" in supply_path:
            supply_path = supply_path.replace("/shared_info/", "/")
        if self.cfg.suggestion_type == "mlm":
            path =  supply_path + "/{}-{}/{}/{}".format(self.cfg.source_lang, self.cfg.target_lang, "bi_context", split)
        elif self.cfg.suggestion_type == "joint":
            path =  supply_path + "/{}-{}/{}/{}".format(self.cfg.source_lang, self.cfg.target_lang, "bi_context", split)
        else:
            path =  supply_path + "/{}-{}/{}/{}".format(self.cfg.source_lang, self.cfg.target_lang, self.cfg.suggestion_type, split)
        suggestions = []
        types = []
        with open(path+".suggestion", "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                suggestions.append(line)
        with open(path+".type", "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                types.append(line)
        self.datasets["suggestions"] = suggestions
        self.datasets["types"] = types
        assert len(self.datasets["suggestions"]) == len(self.datasets["types"])

        # 建立词到拼音的映射
        if self.cfg.target_lang == "zh":
            self.symbol2pinyin = {}
            for symbol in self.tgt_dict.indices.keys():
                self.symbol2pinyin[symbol] = "".join(lazy_pinyin(symbol))
            assert len(self.symbol2pinyin.keys()) == len(self.tgt_dict.indices.keys())

    def generate_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
    
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        from modules.sequence_generator import(
            GWLANArSequenceGenerator
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = GWLANArSequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )