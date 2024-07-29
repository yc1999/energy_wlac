#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import json
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
import sys
sys.path.append("./")
from criterions.gwlan_ce_plus_cl import GWLANCEPlusCLCriterion
import time

RESULTS = dict()
TRUE_CNT = 0
SIZE_CNT = 0
G_TRUE_CNT = 0
G_SIZE_CNT = 0

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        if cfg.task.baseline_ensemble == True:
            output_path = os.path.join(
                cfg.common_eval.results_path,
                "generate-{}-ensemble.txt".format(cfg.dataset.gen_subset),
            )        
        else:
            output_path = os.path.join(
                cfg.common_eval.results_path,
                "generate-{}.txt".format(cfg.dataset.gen_subset),
            )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    global TRUE_CNT
    global SIZE_CNT
    global RESULTS
    global G_TRUE_CNT
    global G_SIZE_CNT
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    cfg.task.results_path = cfg.common_eval.results_path
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
    # load .suggestion and .type
    task.load_eval_dataset(cfg.dataset.gen_subset, cfg.task.supply_path)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    top_k_cnt = 0
    threshold = 8
    # print("threshold: ", threshold)
    not_in_top_k = []
    # models[0].cfg.top_k = 9
    # print("k:", models[0].cfg.top_k)

    time_start = time.time()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()

        if task.cfg.baseline_ensemble == True:
            # Case for Normal Baseline Evaluation
            labels, preds, probs,labels_probs, labels_probs_rk, net_output, types, golds = task.baseline_ensemble_inference_step(
                models,
                sample,
            )
            
            # 将预测结果加入统计辞典
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if labels_probs_rk[i] < threshold and labels_probs_rk[i] > 0:
                    top_k_cnt += 1
                elif labels_probs_rk[i] >= threshold:
                    not_in_top_k.append(id+1)
                if task.cfg.output_verbose:
                    RESULTS[id] = dict()
                    RESULTS[id]["gold"] = golds[i]
                    RESULTS[id]["gold_idx"] = labels[i]
                    RESULTS[id]["prefix"] = types[i]
                    RESULTS[id]["probs"] = net_output[i]
                    # "{:.6f}".format(probs[i]) + "\t" + task.tgt_dict[ preds[i] ] + "\t" + "{:.6f}".format(labels_probs[i]) + "\t" + str(labels_probs_rk[i])
                else:
                    RESULTS[id] = task.tgt_dict[ preds[i] ]
            TRUE_CNT += sum(labels == preds)
            SIZE_CNT += preds.shape[0]

        elif task.cfg.nat:
            # Case for NAT training
            n_correct, total, sorted_probs, sorted_indices = task.nat_inference_step(
                models,
                sample
            )
            TRUE_CNT += n_correct
            SIZE_CNT += total

            # 将预测结果加入统计字典
            ids = sample["id"].tolist()
            for idx, id in enumerate(ids):
                if task.cfg.output_verbose:
                    RESULTS[id] = "{}".format(id) \
                                + "\t" + task.datasets["types"][id] + "\t" + task.datasets["suggestions"][id] \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][0]] + "\t" + "{:.6f}".format(sorted_probs[idx][0]) \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][1]] + "\t" + "{:.6f}".format(sorted_probs[idx][1]) \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][2]] + "\t" + "{:.6f}".format(sorted_probs[idx][2])
        elif task.cfg.nat_sc:
            # Case for NAT training
            n_correct, total = task.nat_sc_inference_step(
                models,
                sample
            )
            TRUE_CNT += n_correct
            SIZE_CNT += total

            # 将预测结果加入统计字典
            ids = sample["id"].tolist()
            for idx, id in enumerate(ids):
                if task.cfg.output_verbose:
                    RESULTS[id] = "{}".format(id) \
                                + "\t" + task.datasets["types"][id] + "\t" + task.datasets["suggestions"][id] \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][0]] + "\t" + "{:.6f}".format(sorted_probs[idx][0]) \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][1]] + "\t" + "{:.6f}".format(sorted_probs[idx][1]) \
                                + "\t" + task.tgt_dict.symbols[sorted_indices[idx][2]] + "\t" + "{:.6f}".format(sorted_probs[idx][2])           
        elif task.cfg.continual_pretraining:
            # Case for Contiual Pretraining
            n_correct, total, labels, preds, probs,labels_probs, labels_probs_rk = task.continual_pretraining_inference_step(
                models,
                sample,
            )
            
            # 将预测结果加入统计辞典
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if labels_probs_rk[i] < threshold and labels_probs_rk[i] > 0:
                    top_k_cnt += 1
                elif labels_probs_rk[i] >= threshold:
                    not_in_top_k.append(id+1)
                if task.cfg.output_verbose:
                    RESULTS[id] = "{:.6f}".format(probs[i]) + "\t" + task.tgt_dict[ preds[i] ] + "\t" + "{:.6f}".format(labels_probs[i]) + "\t" + str(labels_probs_rk[i])
                else:
                    RESULTS[id] = task.tgt_dict[ preds[i] ]
            TRUE_CNT += n_correct
            SIZE_CNT += total

        elif task.cfg.rerank_tag == True:
            # Case for Rerank Model

            n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank = task.rerank_tag_inference_step(
                models,
                sample
            )
            
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if task.cfg.output_verbose:
                    RESULTS[id] = "{} {} {} {} {} {} {} {}".format(pred_words[i], pred_words_prob[i], gold_words[i], gold_words_after_index[i], gold_words_prob[i], gold_words_rank[i], top_k_symbols[i], top_k_symbols_rerank[i])
                else:
                    RESULTS[id] = "{} {} {}".format(pred_words[i], gold_words[i], gold_words_after_index[i], )
            TRUE_CNT += n_correct
            SIZE_CNT += total       
        elif task.cfg.electra == True:
            # Case for Electra Model
            n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank = task.rerank_inference_step(
                models,
                sample,
                mode="electra_cutoff_test"
            )
            
            labels, preds, _, _, _ = task.inference_step(
                [models[0].pretrained_generator],
                sample,
            )
            
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if task.cfg.output_verbose:
                    RESULTS[id] = "{} {} {} {} {} {} {} {}".format(pred_words[i], pred_words_prob[i], gold_words[i], gold_words_after_index[i], gold_words_prob[i], gold_words_rank[i], top_k_symbols[i], top_k_symbols_rerank[i])
                else:
                    RESULTS[id] = "{} {} {}".format(pred_words[i], gold_words[i], gold_words_after_index[i], )
            TRUE_CNT += n_correct
            SIZE_CNT += total

            G_TRUE_CNT += sum(labels == preds)
            G_SIZE_CNT += preds.shape[0]

        elif task.cfg.rerank == False:
            # Case for Normal Baseline Evaluation
            labels, preds, probs,labels_probs, labels_probs_rk, topk_sorted_net_output = task.inference_step(
                models,
                sample,
            )
            
            # 将预测结果加入统计辞典
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if labels_probs_rk[i] < threshold and labels_probs_rk[i] > 0:
                    top_k_cnt += 1
                elif labels_probs_rk[i] >= threshold:
                    not_in_top_k.append(id+1)
                if task.cfg.output_verbose:
                    # RESULTS[id] = "{:.6f}".format(probs[i]) + "\t" + task.tgt_dict[ preds[i] ] + "\t" + "{:.6f}".format(labels_probs[i]) + "\t" + str(labels_probs_rk[i])
                    # RESULTS[id] = "\t".join([task.tgt_dict.symbols[index] for index in topk_sorted_net_output[i] ])
                    RESULTS[id] = str([task.tgt_dict.symbols[index] for index in topk_sorted_net_output[i] ]) + "\t" + task.tgt_dict[ preds[i] ]
                else:
                    RESULTS[id] = task.tgt_dict[ preds[i] ] 
            TRUE_CNT += sum(labels == preds)
            SIZE_CNT += preds.shape[0]

        else:
            # Case for Rerank Model
            if task.cfg.contrastive_learning_only:
                n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank = task.rerank_mask_inner_product_generate(
                    models,
                    sample
                )
            
            else:
                n_correct, total, pred_words, pred_words_prob, gold_words, gold_words_after_index, gold_words_prob, gold_words_rank, top_k_symbols, top_k_symbols_rerank = task.rerank_inference_step(
                    models,
                    sample
                )
            
            ids = sample["id"].tolist()
            for i, id in enumerate(ids):
                if task.cfg.output_verbose:
                    RESULTS[id] = "{}\t{}\t{}\t{}".format(top_k_symbols_rerank[i][:3], pred_words[i],  gold_words[i], gold_words_after_index[i])
                    # RESULTS[id] = "{} {} {} {} {} {} {} {}".format(pred_words[i], pred_words_prob[i], gold_words[i], gold_words_after_index[i], gold_words_prob[i], gold_words_rank[i], top_k_symbols[i], top_k_symbols_rerank[i])
                else:
                    RESULTS[id] = "{} {} {}".format(pred_words[i], gold_words[i], gold_words_after_index[i], )
            TRUE_CNT += n_correct
            SIZE_CNT += total

    time_end = time.time()
    print("Total inference time: {}".format(time_end - time_start))

    if task.cfg.baseline_ensemble == True:
        # return scorer
        print("TRUE_CNT is {} , SIZE_CNT is {}, acc is {}".format(TRUE_CNT, SIZE_CNT,TRUE_CNT / SIZE_CNT), )
        print("There are {} false classified samples, and there are {} in top {} candidates, ratio is {}".format(SIZE_CNT - TRUE_CNT, top_k_cnt, threshold, top_k_cnt / (SIZE_CNT - TRUE_CNT)))
        print("Recall is {}".format( (TRUE_CNT+top_k_cnt) / SIZE_CNT) )
        print("Not is top {}: {}".format(threshold, not_in_top_k))
        RESULTS = dict( sorted( RESULTS.items(), key=lambda item: item[0] ) )
        output_path = os.path.join(
                task.cfg.results_path,
                "generate-{}-2.pt".format(task.cfg.gen_subset),
            )   
        torch.save(RESULTS,output_path) 
        # for id in RESULTS.keys():
            # print(id)
            # print(output_file)
            # print("{} {} {} {}".format(id+1, RESULTS[id], task.datasets["suggestions"][id], task.tgt_dict.symbols[task.tgt_dict.index((task.datasets["suggestions"][id]))] ), file=output_file)

    elif task.cfg.electra == True:
        # return scorer
        # rerank == True
        print("TRUE_CNT is {} , SIZE_CNT is {}, acc is {}".format(TRUE_CNT, SIZE_CNT,TRUE_CNT / SIZE_CNT), )
        print("G_TRUE_CNT is {} , G_SIZE_CNT is {}, acc is {}".format(G_TRUE_CNT, G_SIZE_CNT,G_TRUE_CNT / G_SIZE_CNT), )
        print("There are {} false classified samples, and there are {} in top {} candidates, ratio is {}".format(SIZE_CNT - TRUE_CNT, top_k_cnt, threshold, top_k_cnt / (SIZE_CNT - TRUE_CNT)))
        print("Recall is {}".format( (TRUE_CNT+top_k_cnt) / SIZE_CNT) )
        print("Not is top {}: {}".format(threshold, not_in_top_k))
        RESULTS = dict( sorted( RESULTS.items(), key=lambda item: item[0] ) )
        for id in RESULTS.keys():
            print("{} {}".format( id+1, RESULTS[id]  ), file=output_file)   
    elif task.cfg.rerank_tag == True:
        # return scorer
        print("TRUE_CNT is {} , SIZE_CNT is {}, acc is {}".format(TRUE_CNT, SIZE_CNT,TRUE_CNT / SIZE_CNT), )
        print("There are {} false classified samples, and there are {} in top {} candidates, ratio is {}".format(SIZE_CNT - TRUE_CNT, top_k_cnt, threshold, top_k_cnt / (SIZE_CNT - TRUE_CNT)))
        print("Recall is {}".format( (TRUE_CNT+top_k_cnt) / SIZE_CNT) )
        print("Not is top {}: {}".format(threshold, not_in_top_k))
        RESULTS = dict( sorted( RESULTS.items(), key=lambda item: item[0] ) )
        for id in RESULTS.keys():
            # print(id)
            # print(output_file)
            print("{} {} {} {}".format(id+1, RESULTS[id], task.datasets["suggestions"][id], task.tgt_dict.symbols[task.tgt_dict.index((task.datasets["suggestions"][id]))] ), file=output_file)     
    elif task.cfg.rerank == False or task.cfg.continual_pretraining:
        # return scorer
        if getattr(task.cfg, "len_beam", None) is not None:
            print("len_beam is {}".format(task.cfg.len_beam))
        print("TRUE_CNT is {} , SIZE_CNT is {}, acc is {}".format(TRUE_CNT, SIZE_CNT,TRUE_CNT / SIZE_CNT), )
        print("There are {} false classified samples, and there are {} in top {} candidates, ratio is {}".format(SIZE_CNT - TRUE_CNT, top_k_cnt, threshold, top_k_cnt / (SIZE_CNT - TRUE_CNT)))
        print("Recall is {}".format( (TRUE_CNT+top_k_cnt) / SIZE_CNT) )
        print("Not is top {}: {}".format(threshold, not_in_top_k))
        RESULTS = dict( sorted( RESULTS.items(), key=lambda item: item[0] ) )
        for id in RESULTS.keys():
            # print(id)
            # print(output_file)
            print("{}\t{}\t{}\t{}".format(id+1, RESULTS[id], task.datasets["suggestions"][id], task.tgt_dict.symbols[task.tgt_dict.index((task.datasets["suggestions"][id]))] ), file=output_file)
            # print("{}\t{}".format(id+1, RESULTS[id]), file=output_file)
    else:
        # return scorer
        # rerank == True
        print("TRUE_CNT is {} , SIZE_CNT is {}, acc is {}".format(TRUE_CNT, SIZE_CNT,TRUE_CNT / SIZE_CNT), )
        print("There are {} false classified samples, and there are {} in top {} candidates, ratio is {}".format(SIZE_CNT - TRUE_CNT, top_k_cnt, threshold, top_k_cnt / (SIZE_CNT - TRUE_CNT)))
        print("Recall is {}".format( (TRUE_CNT+top_k_cnt) / SIZE_CNT) )
        print("Not is top {}: {}".format(threshold, not_in_top_k))
        RESULTS = dict( sorted( RESULTS.items(), key=lambda item: item[0] ) )
        for id in RESULTS.keys():
            print("{}\t{}".format( id+1, RESULTS[id]  ), file=output_file)        

def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
