# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from models.rerank_transformer_decoder import RerankTransformerDecoder
from dataclasses import dataclass, field
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from typing import Optional
import sys
sys.path.append("./")
from modules.gwlan_sinusoidal_positional_embedding import GWLANSinusoidalPositionalEmbedding
from modules.local_info_gather_layer import LocalInfoGatherLayer, LocalInfoGatherLayerFairseq
#TODO: delete this
# from scratches.fairseq_attention import LocalInfoGatherLayer
from models.gwlan_transformer_decoder import GWLANTransformerDecoder
from models.btba_transformer_decoder import BTBATransformerDecoder
import logging
logger = logging.getLogger(__name__)

import random
from torch.nn.utils.rnn import pad_sequence

def read_synonyms(tgt_dict):
    #TODO: Change hard code dictionary path
    dictionary_file = "/apdcephfs/share_916081/khalidcyang/project/gwlan-fairseq/dataset/bin/zh-en/bi_context/dict.en.txt"

    # 建立词典
    symbols = dict()
    indices = dict()
    with open(dictionary_file, "r") as f:
        for idx, line in enumerate(f):
            word = line.strip().split(" ")[0]
            symbols[idx+4] = word
            indices[word] = idx + 4 # 必须要+4才行

    # compute frequency
    tot_num = float(sum(tgt_dict.count))
    freq_list = [ cnt / tot_num for cnt in tgt_dict.count ]
    
    # TODO: 手动设置为无穷大
    freq_list[0] = 1
    freq_list[1] = 1
    freq_list[2] = 1
    freq_list[3] = 1
    freq_list[-1] = 1

    # 进行映射
    #TODO: change hard code length
    # synonym_list = [ []  for _ in range(50001) ]
    # with open("synonyms.txt", "r") as f:
    #     for line in f:
    #         word = line.strip().split(" ")[0]
    #         idx = indices[word]
    #         for synonym in line.strip().split(" ")[1:]:
    #             synonym_list[idx].append(indices[synonym])

    ###### filter synonym by frequency ######
    synonym_list = [ []  for _ in range(50001) ]
    with open("synonyms.txt", "r") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            idx = indices[word]
            for synonym in line.strip().split(" ")[1:]:
                synonym_list[idx].append(indices[synonym])
            synonym_list[idx] = sorted(synonym_list[idx], key=lambda x: freq_list[x], reverse=True)
            synonym_list[idx] = synonym_list[idx][:5]
    ##########################################

    # for i in range(4):
    #     synonym_list[i].append(i)
    # synonym_list[len(synonym_list)-1].append(len(synonym_list)-1)
    
    # print(synonym_list) 
    # print(synonym_list[1000])
    logger.info("len of synonym list: {}".format(len(synonym_list))) 
    return synonym_list

def get_synonyms(target, synonym_list, padding_value):
    lsts = []
    for tgt in target.tolist(): 
        if tgt == padding_value:
            print("get_synonyms() is ", tgt)
            lsts.append( [0] + synonym_list[0] )
        else:
            lsts.append( [tgt] + synonym_list[tgt] )


    new_lsts = []
    for l in lsts:
        # new_l = l[:1]
        if len(l) > 1:
            rest_l = l[1:]
            new_l = random.sample(rest_l, k=5 if len(rest_l) >= 5 else len(rest_l))
            # new_l += random.sample(rest_l, k=4 if len(rest_l) >= 4 else len(rest_l))
            new_lsts.append(new_l)
        else:
            new_lsts.append([])
    tensor_lst = []
    for l in new_lsts:
        tensor_lst.append(torch.tensor(l, device=target.device, dtype=torch.long))

    synonyms = pad_sequence(tensor_lst, batch_first=True, padding_value=padding_value) # pad value is two

    return synonyms

def read_hypernyms(tgt_dict):
    #TODO: Change hard code dictionary path
    dictionary_file = "/apdcephfs/share_916081/khalidcyang/project/gwlan-fairseq/dataset/bin/zh-en/bi_context/dict.en.txt"

    # 建立词典
    symbols = dict()
    indices = dict()
    with open(dictionary_file, "r") as f:
        for idx, line in enumerate(f):
            word = line.strip().split(" ")[0]
            symbols[idx+4] = word
            indices[word] = idx + 4 # 必须要+4才行
    
    # 进行映射
    #TODO: change hard code length
    hypernym_list = [ []  for _ in range(50001) ]

    # compute frequency
    tot_num = float(sum(tgt_dict.count))
    freq_list = [ cnt / tot_num for cnt in tgt_dict.count ]
    
    # TODO: 手动设置为无穷大
    freq_list[0] = 0
    freq_list[1] = 0
    freq_list[2] = 0
    freq_list[3] = 0
    freq_list[-1] = 0

    freq_list = np.array(freq_list)
    ranked_freq_list = np.sort(freq_list)
    args = np.argsort(freq_list)
    
    # 累加到前10%
    tmp_cnt = 0.0
    least_5 = list()
    for idx, val in enumerate(ranked_freq_list):
        tmp_cnt += val
        least_5.append(args[idx])
        if tmp_cnt >= 0.05:
            break

    with open("word2hypernyms.txt", "r") as f:
       for line in f:
            word = line.strip().split(" ")[0]
            idx = indices[word]
            hypernyms = eval(line.strip().split(" ",1)[1])
            hypernyms = [ [ indices[word] for word in hypernym] for hypernym in hypernyms ]
            # 进行过滤，只取词频最大的那几个上位词token
            filtered_hypernyms = []
            for hypernym in hypernyms:
                hypernym = sorted(hypernym, key=lambda x:freq_list[x] , reverse=True)
                hypernym = hypernym[:1]
                filtered_hypernyms.append(hypernym)
            hypernym_list[idx] = filtered_hypernyms
            # hypernym_list[idx] = hypernyms

    logger.info("len of hypernym list: {}".format(len(hypernym_list)))

    return hypernym_list 

def get_hypernyms(target, hypernym_list, padding_value):
    lsts = []
    # 对于hypernym_list中的元素，我们首先取2个hypernyms集合，同时将自身也加进去。
    for tgt in target.tolist():
        hypernyms = hypernym_list[tgt]
        if len(hypernyms) == 0:
            if tgt == padding_value:
                # lsts.append( [0] )
                lsts.append([])
            else:
                # lsts.append( [tgt])
                lsts.append([])
        elif len(hypernyms) == 1:
            # lsts.append( [tgt] + random.sample( hypernyms[0], k=3 if len(hypernyms[0]) > 3 else len(hypernyms[0]) ))
            lsts.append( random.sample( hypernyms[0], k=2 if len(hypernyms[0]) > 2 else len(hypernyms[0]) ))
        elif len(hypernyms) > 1:
            two_hypernyms = random.sample( hypernyms, k=2 )
            word_hypernyms = []
            for h in two_hypernyms:
                word_hypernyms.extend( random.sample(h, k=2 if len(h) > 2 else len(h)) )
            # lsts.append( [tgt] +  word_hypernyms)
            lsts.append( word_hypernyms)
    
    tensor_lst = []
    for l in lsts:
        tensor_lst.append(torch.tensor(l, dtype=torch.long , device=target.device))

    hypernyms = pad_sequence(tensor_lst, batch_first=True, padding_value=padding_value) # pad value is two
    
    # assert hypernyms.size(1) <= 7
    assert hypernyms.size(1) <= 4

    return hypernyms

class LinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model("rerank_transformer_confidence_sampling")
class RerankTransformerConfidenceSamplingModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def load_baseline(self, path):
        """
        Given the path to the baseline model, load the model and return it.
        """
        from fairseq.checkpoint_utils import load_checkpoint_to_cpu
        from fairseq import tasks
        
        logger.info("loading baseline model from {}.".format(path))
        if "/shared_info/" in path:
            path = path.replace("/shared_info/", "/")
        state = load_checkpoint_to_cpu(path)
        cfg = state['cfg']
        task = tasks.setup_task(cfg.task)
        baseline = task.build_model(cfg.model, from_checkpoint=True)
        baseline.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)
        baseline.eval()
        
        if hasattr(self.cfg, "reranker_init_path") and self.cfg.reranker_init_path != None:
            logger.info("After loading baseline, we initialize reranker model from {}.".format(self.cfg.reranker_init_path))
            if "/shared_info/" in self.cfg.reranker_init_path:
                self.cfg.reranker_init_path = self.cfg.reranker_init_path.replace("/shared_info/", "/")
            state = load_checkpoint_to_cpu(self.cfg.reranker_init_path)
            
            rerank_transformer_dict = self.state_dict()
            pretrained_dict = state["model"]
            
            # filter operation
            if "rerank" not in self.cfg.reranker_init_path:
                logger.info("Use pretrained MLM to initialize reranker.")
                new_dict = { k:v for k,v in pretrained_dict.items() if k in rerank_transformer_dict.keys() and "decoder.output_projection" not in k }
            else:
                logger.info("Use pretrained Discriminator to initialize reranker.")
                new_dict = { k:v for k,v in pretrained_dict.items() if k in rerank_transformer_dict.keys() }


            rerank_transformer_dict.update(new_dict)
            logger.info("Toal : {}, update : {}".format(len(rerank_transformer_dict), len(new_dict)))
            self.load_state_dict(rerank_transformer_dict)
            logger.info("Finish loading rerank transformer.")
        else:
            # 除了使用state["model"]初始化baseline，还要用其初始化rerank model本身
            logger.info("After loading baseline, we don't use any weight to initialize reranker model.")
            if self.cfg.encoder.embed_dim == 1024:
                #TODO: We should modify this hard code.
                logger.info("Use Transformer-Big to initialize.")
                state = load_checkpoint_to_cpu("${project_root}save/zh-en/bi_context/lr-0.0007__max_tokens-4096__update_freq-16__gpus-6,7__share_decoder_input_output_embed-true__date-0427_2141/checkpoint.best_accuracy_69.5902.pt")
        
        return baseline

    def load_k_folds_baseline(self, paths, index):
        """
        Given the path to the baseline model, load the model and return it.
        """
        from fairseq.checkpoint_utils import load_checkpoint_to_cpu
        from fairseq import tasks

        paths = paths.split(":")
        path = paths[index]

        logger.info("loading {}-th k_folds baseline model from {}.".format(index, path))
        state = load_checkpoint_to_cpu(path)
        cfg = state['cfg']
        task = tasks.setup_task(cfg.task)
        baseline = task.build_model(cfg.model, from_checkpoint=True)
        baseline.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)
        baseline.eval()

        for param in baseline.parameters():
            param.requires_grad = False
        
        return baseline

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
        # as for the positional encoding of gwlan's decoder is very different, we should replace the original Positional Encoder
        self.decoder.embed_positions = GWLANSinusoidalPositionalEmbedding(embedding_dim=self.decoder.embed_positions.embedding_dim, padding_idx=self.decoder.embed_positions.padding_idx, init_size=self.decoder.embed_positions.weights.size(0))
        
        if hasattr(self.cfg, "word_alignment_enhanced") and self.cfg.word_alignment_enhanced == True:
            logger.info("parameter: word_alignment_enhanced is True, so we should use segment embedding.")
            self.decoder.segment_embeddings = nn.Embedding(
                2,
                self.decoder.embed_positions.embedding_dim
            )

        if args.wordnet_enhanced == True:
            # If "wordnet_enhanced" is True, we need to add the LocalInfoGatherLayer to the decoder.
            logger.info("wordnet_enhanced is True.")
            if hasattr(args, "local_fairseq") and args.local_fairseq == False:
                self.decoder.local_info_gather_layer = LocalInfoGatherLayer(
                                                        embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                        padding_idx = self.decoder.embed_tokens.padding_idx, 
                                                        dropout = args.dropout
                                                        )
            else:
                self.decoder.local_info_gather_layer = LocalInfoGatherLayerFairseq(
                                                        embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                        padding_idx = self.decoder.embed_tokens.padding_idx, 
                                                        dropout = args.dropout,
                                                        num_heads = args.local_num_heads if hasattr(args, "local_num_heads") else 1,
                                                        )

        if args.hypernym_enhanced == True:
            # If "hypernym_enhanced" is True, we need to add the LocalInfoGatherLayer to the decoder.
            logger.info("hypernym_enhanced is True.")
            if hasattr(args, "local_fairseq") and args.local_fairseq == False:
                self.decoder.hypernym_info_gather_layer = LocalInfoGatherLayer(
                                                            embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                            padding_idx = self.decoder.embed_tokens.padding_idx,
                                                            dropout = args.dropout
                                                            )
            else:
                self.decoder.hypernym_info_gather_layer = LocalInfoGatherLayerFairseq(
                                                            embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                            padding_idx = self.decoder.embed_tokens.padding_idx, 
                                                            dropout = args.dropout,
                                                            num_heads = args.local_num_heads if hasattr(args, "local_num_heads") else 1,
                                                            )
        if hasattr(args, "wn_hybrid_fusion") and args.wn_hybrid_fusion == True:
                self.decoder.synonym_info_gather_layer = LocalInfoGatherLayerFairseq(
                                                        embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                        padding_idx = self.decoder.embed_tokens.padding_idx, 
                                                        dropout = args.dropout,
                                                        num_heads = args.local_num_heads,
                                                        )
                self.decoder.hypernym_info_gather_layer = LocalInfoGatherLayerFairseq(
                                                            embed_dim = self.decoder.embed_tokens.embedding_dim, 
                                                            padding_idx = self.decoder.embed_tokens.padding_idx, 
                                                            dropout = args.dropout,
                                                            num_heads = args.local_num_heads,
                                                            )
                self.decoder.fusion_layer = nn.Linear(
                                                    in_features=1024,
                                                    out_features=512
                                            )   

        # 导入baseline model
        self.baseline = self.load_baseline(cfg.baseline_path)
        for param in self.baseline.parameters():
            param.requires_grad = False

        if hasattr(args, "k_folds") and args.k_folds > 1:
            # we should load different baselines
            self.k_folds_baselines = torch.nn.ModuleList()
            for index in range(args.k_folds):
                self.k_folds_baselines.append( self.load_k_folds_baseline(paths=cfg.k_folds_baselines_path, index=index ) )

        self.tgt_dict = self.decoder.dictionary 
        self.synonym_list = read_synonyms(self.tgt_dict)
        self.hypernym_list = read_hypernyms(self.tgt_dict)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        cfg = TransformerConfig.from_namespace(args)
        return RerankTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
            output_projection=cls.get_output_projection(cfg),
        )
        # return super().build_decoder(
        #     TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        # )

    def get_masked_probs(self, type_masks, logits):
        logits = logits.masked_fill(type_masks, -float('inf'))
        probs = logits.softmax(dim=-1)
        probs[torch.isnan(probs)] = 1e-6
        return probs

    # overwrite forward function
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_position_ids,
        target_position,
        target,
        type_masks,
        tgt_dict,
        synonyms: Optional[torch.Tensor] = None,
        epoch_idx: Optional[int] = None,
        mode: str = "fit",
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if mode == "fit":
            # 判断是否是k_folds模式
            if hasattr(self.cfg, "k_folds") and self.cfg.k_folds > 1:
                assert epoch_idx is not None

                # 如果是k_folds模式，则使用不同的baseline
                self.k_folds_baselines[ (epoch_idx - 1) % self.cfg.k_folds].eval()
                with torch.no_grad():
                    baseline_net_output = self.k_folds_baselines[ (epoch_idx - 1) % self.cfg.k_folds](
                        src_tokens,
                        src_lengths,
                        prev_output_tokens,
                        tgt_position_ids,
                    )
            else:
                # 首先运行self.baseline得到baseline的输出
                self.baseline.eval()
                with torch.no_grad():
                    baseline_net_output = self.baseline(
                        src_tokens,
                        src_lengths,
                        prev_output_tokens,
                        tgt_position_ids,
                    )
        elif mode == "infer":
            # 首先运行self.baseline得到baseline的输出
            self.baseline.eval()
            with torch.no_grad():
                baseline_net_output = self.baseline(
                    src_tokens,
                    src_lengths,
                    prev_output_tokens,
                    tgt_position_ids,
                )            

        # logits: (bsz, tgt_len, vocab_size) -> (bsz, vocab_size)
        logits = baseline_net_output[0]
        logits = logits.gather(1, target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, logits.size(-1))).squeeze(1)

        # when we have typed characters, we can use it to get top k negative examples
        # probs: (bsz, vocab_size)
        probs = self.get_masked_probs(type_masks, logits)

        sorted_indices = torch.argsort(probs, dim=1, descending=True)

        # if self.cfg.dynamic_negative_sampling == False:
        #     if mode == "fit":
        #         # fit mode needs groud truth labels
        #         # top_k_indices: (bsz, top_k)
        #         top_k_indices = sorted_indices[ : , :self.cfg.top_k-1 ]
        #         top_k_indices = torch.cat( (target.unsqueeze(1), top_k_indices), dim=1)
        #         if self.cfg.contrastive_learning == True:
        #             # contrastive learning
        #             top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )
        #     elif mode == "infer":
        #         # infer mode doesn't need ground truth labels
        #         # top_k_indices: (bsz, top_k)
        #         top_k_indices = sorted_indices[:, :self.cfg.top_k]
                
        #         ##### print the top k probabilities #####
        #         sorted_probs = torch.sort(probs, dim=1, descending=True)[0]
        #         top_k_probs = sorted_probs[:, :self.cfg.top_k]
        #         # do softmax
        #         print(top_k_indices.tolist())
        #         print([ tgt_dict.symbols[index] for index in top_k_indices.tolist()[0] ])
        #         print(top_k_probs.tolist()) 
        #         top_k_probs = top_k_probs.softmax(dim=1)
        #         print(top_k_probs.tolist()) 
        #         #########################################
                
        #         if self.cfg.need_mask == True:
        #             # When need mask, we need to append mask token.
        #             top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )        
        # else:
            # Here we use dynamic negative sampling.
        if mode == "fit":
            # fit mode needs groud truth labels
            # top_k_indices = sorted_indices[ :, :6]
            top_k_indices = sorted_indices

            # random sample top_k_indices, p is uniform distribution
            # p: (bsz, 4), index: (bsz, top_k-1)
            sorted_probs = torch.sort(probs, dim=1, descending=True)[0]
            top_k_probs = sorted_probs

            # p = torch.full_like( input=top_k_indices , fill_value=(1.0 / top_k_indices.size(1)), dtype=torch.float)
            p = top_k_probs
            try:
                index = p.multinomial(num_samples=self.cfg.top_k-1, replacement=False)
            except:
                print( [ tgt_dict.symbols[x] for x in target.tolist() ] )
                print([ [tgt_dict.symbols[index] for index in indices] for indices in top_k_indices.tolist() ])
                print("error is: ", p)
            # top_k_indices: (bsz, top_k)
            top_k_indices = top_k_indices.gather(1, index)
            top_k_indices = torch.cat( (target.unsqueeze(1), top_k_indices), dim=1)

            if self.cfg.contrastive_learning == True:
                # contrastive learning
                # top_k_indices: (bsz, top_k+1)
                top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )

        elif mode == "infer":
            # infer mode doesn't need ground truth labels
            # top_k_indices: (bsz, top_k)
            top_k_indices = sorted_indices[:, :self.cfg.top_k]  #TODO: change hard-coded value
            if self.cfg.need_mask == True:
                # When need mask, we need to append mask token.
                # top_k_indices: (bsz, top_k+1)
                top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )        

        # prev_output_tokens: (bsz, top_k, tgt_len)
        prev_output_tokens = prev_output_tokens.unsqueeze(1).expand(-1, top_k_indices.size(1), -1).clone()

        prev_output_tokens.scatter_(
            -1,
            target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, top_k_indices.size(1), -1),
            top_k_indices.unsqueeze(-1)
        )

        # target: (bsz, top_k) -> (bsz * top_k)
        cross_entropy_target = target
        target = torch.zeros_like(top_k_indices)
        target[:, 0] = 1
        target = target.view(-1)

        # tgt_position_ids: (bsz, top_k, tgt_len)
        tgt_position_ids = tgt_position_ids.unsqueeze(1).expand(-1, top_k_indices.size(1), -1)                 

        # view prev_output_tokens and tgt_position_ids as (bsz * top_k, tgt_len)
        prev_output_tokens = prev_output_tokens.view(-1, prev_output_tokens.size(-1))
        tgt_position_ids = tgt_position_ids.reshape(-1, tgt_position_ids.size(-1))
        
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        # expand encoder_out's encoder_out from (src_len, bsz, embed_dim) to (src_len, bsz * top_k, embed_dim)
        encoder_out["encoder_out"][0] = encoder_out["encoder_out"][0].unsqueeze(2).expand(-1, -1, top_k_indices.size(1), -1).reshape(encoder_out["encoder_out"][0].size(0), -1, encoder_out["encoder_out"][0].size(-1))

        # expand encoder_out's encoder_padding_mask from (bsz, src_len) to (bsz * top_k, src_len)
        encoder_out["encoder_padding_mask"][0] = encoder_out["encoder_padding_mask"][0].unsqueeze(1).expand(-1, top_k_indices.size(1), -1).reshape(-1, encoder_out["encoder_padding_mask"][0].size(-1))

        # target_position: (bsz * top_k, 1, embed_dim)
        target_position = target_position.unsqueeze(-1).expand(-1, top_k_indices.size(1)).reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.cfg.decoder.embed_dim)

        # we should use `top_k_indices` to generate synonyms
        synonyms = None
        hypernyms = None
        if hasattr(self.cfg, "wn_hybrid_fusion") and self.cfg.wn_hybrid_fusion == True:
            synonyms = get_synonyms(top_k_indices.reshape(-1), self.synonym_list, padding_value=tgt_dict.pad())
            hypernyms = get_hypernyms(top_k_indices.reshape(-1), self.hypernym_list, padding_value=tgt_dict.pad())
        elif self.cfg.wordnet_enhanced == True:
            synonyms = get_synonyms(top_k_indices.reshape(-1), self.synonym_list, padding_value=tgt_dict.pad())
        elif self.cfg.hypernym_enhanced == True:
            hypernyms = get_hypernyms(top_k_indices.reshape(-1), self.hypernym_list, padding_value=tgt_dict.pad())
        
        if hasattr(self.cfg, "tgt_input_enhanced") and self.cfg.tgt_input_enhanced == True:
            prev_output_tokens, tgt_position_ids, target_position = self.enhance_target_input(prev_output_tokens = prev_output_tokens, 
                                                                                            tgt_position_ids = tgt_position_ids, 
                                                                                            target_position = target_position, 
                                                                                            top_k_indices = top_k_indices, 
                                                                                            synonym_list = self.synonym_list,
                                                                                            hypernym_list = self.hypernym_list, 
                                                                                            padding_value = tgt_dict.pad(),
                                                                                            enhance_type = self.cfg.enhanced_type if hasattr(self.cfg, "enhanced_type") else "synonym")
        # print("top_k_indices: ")
        # print(top_k_indices.tolist())
        # top_k_symbols = [ [ tgt_dict.symbols[candidate] for candidate in candidates] for candidates in  top_k_indices.tolist() ]
        # print(top_k_symbols)

        # Feature: Word Alignment Enhanced Version
        segment_labels = None
        if "align_sents" in kwargs.keys() and "align_positions" in kwargs.keys() and kwargs["align_sents"] is not None and kwargs["align_positions"] is not None:
            # print(kwargs["align_sents"])
            # print(kwargs["align_positions"])
            prev_output_tokens, tgt_position_ids, segment_labels = self.enhance_word_alignment(
                                                                            prev_output_tokens = prev_output_tokens,
                                                                            tgt_position_ids = tgt_position_ids,
                                                                            top_k_indices = top_k_indices,
                                                                            align_sents = kwargs["align_sents"],
                                                                            align_positions = kwargs["align_positions"]
            )

        decoder_out = self.decoder(
            prev_output_tokens,
            tgt_position_ids,
            target_position,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            synonyms=synonyms,
            hypernyms=hypernyms,
            segment_labels=segment_labels
        )
        # if self.cfg.share_decoder_input_output_embed:
        #     origin_decoder_out_logits = decoder_out[0]
        #     # print("origin_decoder_out_logits: ", origin_decoder_out_logits.size())
        #     gather_decoder_out_logits = origin_decoder_out_logits.gather(1, top_k_indices.view(-1).unsqueeze(-1))
        #     # print("gather_decoder_out_logits: ", gather_decoder_out_logits.size())
        #     decoder_out = (gather_decoder_out_logits,) + decoder_out[1:]
        return (decoder_out + (target, top_k_indices, cross_entropy_target) )

    def enhance_word_alignment(self,
                                prev_output_tokens,
                                tgt_position_ids,
                                top_k_indices,
                                align_sents,
                                align_positions):
        # prev_output_tokens, tgt_position_ids : (bsz * top_k, tgt_len) 
        # target_position: (bsz * top_k, 1, embed_dim)
        # align_sents: (bsz, cutted_src_len)
        # align_positions: (bsz, cutted_src_len)

        # We should first extend align_sents to (bsz * top_k, tgt_len)
        align_sents = align_sents.unsqueeze(1).expand(-1, top_k_indices.size(1), -1).reshape(-1, align_sents.size(-1))
        align_positions = align_positions.unsqueeze(1).expand(-1, top_k_indices.size(1), -1).reshape(-1, align_sents.size(-1))

        assert align_sents.size() == align_positions.size()

        segment_labels = torch.cat( [torch.full_like(prev_output_tokens, fill_value=0, dtype=torch.long),
                                    torch.full_like(align_sents, fill_value=1, dtype=torch.long)],
                                    dim=-1)

        prev_output_tokens = torch.cat( [prev_output_tokens, align_sents], dim=-1)
        tgt_position_ids = torch.cat( [tgt_position_ids, align_positions], dim=-1)

        return prev_output_tokens, tgt_position_ids, segment_labels

    def enhance_target_input(self,
                            prev_output_tokens, 
                            tgt_position_ids, 
                            target_position, 
                            top_k_indices, 
                            synonym_list, 
                            hypernym_list,
                            padding_value,
                            enhance_type = "synonym"):
        if enhance_type == "synonym":
            # synonyms: (bsz * top_k, 5)
            synonyms = get_synonyms(top_k_indices.reshape(-1), synonym_list, padding_value=padding_value)

            # 初始化synonyms中的每一个token的position embedding为对应的target_position
            # we should first get target's position ids
            # target_position: (bsz * top_k, 1, embed_dim), tgt_position_ids: (bsz * top_k, tgt_len)
            # tgt_pos =   # (bsz * top_k, 1)
            # tgt_position_id = 

            # synonyms_postition_ids: (bsz * top_k, 1, 512) -> (bsz * top_k, 5)
            #synonyms_position_ids = target_position[:,:,0].clone().detach().to(target_position.device)   # 感觉其实无所谓，这一步。
            synonyms_position_ids = tgt_position_ids.gather(1, target_position[:,:,0]).expand(-1, synonyms.size(1))

            # 找到所有的pad的位置
            pad_masks = torch.full_like(input=synonyms, fill_value=True, dtype=torch.bool) # (synonyms == padding_value)
            # pad_masks = (synonyms == padding_value)
            # synonyms_postition_ids: (bsz * top_k, 5)
            synonyms_position_ids = torch.masked_fill(synonyms_position_ids, pad_masks, 1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            prev_output_tokens = torch.cat([prev_output_tokens, synonyms], dim=-1)
            tgt_position_ids = torch.cat([tgt_position_ids, synonyms_position_ids], dim=-1)
        elif enhance_type == "hypernym":
            hypernyms = get_hypernyms(top_k_indices.reshape(-1), hypernym_list, padding_value=padding_value)
            hypernyms_position_ids = tgt_position_ids.gather(1, target_position[:,:,0]).expand(-1, hypernyms.size(1))
            pad_masks = (hypernyms == padding_value)
            hypernyms_position_ids = torch.masked_fill(hypernyms_position_ids, pad_masks, 1)
            prev_output_tokens = torch.cat([prev_output_tokens, hypernyms], dim=-1)
            tgt_position_ids = torch.cat([tgt_position_ids, hypernyms_position_ids], dim=-1)            
        elif enhance_type == "hybrid":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return prev_output_tokens, tgt_position_ids, target_position

    @classmethod 
    def get_output_projection(cls, cfg):
        if cfg.share_decoder_input_output_embed:
            # When share_decoder_input_output_embed is True, fairseq will automatically use the same embedding matrix for input and output projection
            return None
        elif cfg.output_projection == "linear":
            # return nn.Linear(cfg.decoder.embed_dim, 1, bias=False)
            return LinearClassificationHead(
                cfg.decoder.embed_dim,
                cfg.decoder.embed_dim,
                1,
                "tanh",
                pooler_dropout=0.1)
        elif cfg.output_projection == "input_word_embedding":
            cfg.share_decoder_input_output_embed = True
            return None
        else:
            raise NotImplementedError

    def get_targets(self, sample, net_output):
        """This is different from FairseqEncoderDecoder's get_targets
        """
        return net_output[2]

    #TODO: add a new function
    def get_suggestion_targets(self, sample, net_output, suggestion_type):
        """Get targets from either the sample or the net's output."""
        return sample["{}_net_input".format(suggestion_type)]["{}_target".format(suggestion_type)]

# architectures

@register_model_architecture("rerank_transformer_confidence_sampling", "rerank_transformer_confidence_sampling")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("rerank_transformer_confidence_sampling", "rerank_transformer_confidence_sampling_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)
