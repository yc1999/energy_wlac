# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import random
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
from modules.gwlan_sinusoidal_positional_embedding import GWLANSinusoidalPositionalEmbedding
from models.gwlan_transformer_decoder import GWLANTransformerDecoder
from models.btba_transformer_decoder import BTBATransformerDecoder
from models.rerank_transformer_decoder_dynamic_anchors import RerankTransformerDecoderDynamicAnchors
from fairseq.models.roberta.model import RobertaClassificationHead
import logging
logger = logging.getLogger(__name__)

@register_model("rerank_transformer_dynamic_anchors")
class RerankTransformerDynamicAnchorsModel(TransformerModelBase):
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
        state = load_checkpoint_to_cpu(path)
        cfg = state['cfg']
        task = tasks.setup_task(cfg.task)
        baseline = task.build_model(cfg.model, from_checkpoint=True)
        baseline.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)
        baseline.eval()
        
        # 除了使用state["model"]初始化baseline，还要用其初始化rerank model本身
        logger.info("After loading baseline, use the same weight to initializing rerank transformer.")
        state = load_checkpoint_to_cpu(path)
        rerank_transformer_dict = self.state_dict()
        pretrained_dict = state["model"]
        
        # filter operation
        new_dict = { k:v for k,v in pretrained_dict.items() if k in rerank_transformer_dict.keys() and "decoder.output_projection" not in k }
        
        rerank_transformer_dict.update(new_dict)
        logger.info("Toal : {}, update : {}".format(len(rerank_transformer_dict), len(new_dict)))
        self.load_state_dict(rerank_transformer_dict)
        logger.info("Finish loading rerank transformer.")
        
        return baseline

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

        # as for the positional encoding of gwlan's decoder is very different, we should replace the original Positional Encoder
        self.decoder.embed_positions = GWLANSinusoidalPositionalEmbedding(embedding_dim=self.decoder.embed_positions.embedding_dim, padding_idx=self.decoder.embed_positions.padding_idx, init_size=self.decoder.embed_positions.weights.size(0))
        
        # 导入baseline model
        self.baseline = self.load_baseline(cfg.baseline_path)
        for param in self.baseline.parameters():
            param.requires_grad = False

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
        return RerankTransformerDecoderDynamicAnchors(
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
        mode: str = "fit",
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
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
        if self.training == True:
            logits = logits.gather(1, target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, logits.size(-1))).squeeze(1)
            # logits = logits[:, target_position, :]
        else:
            logits = logits.gather(1, target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, logits.size(-1))).squeeze(1)

        # when we have typed characters, we can use it to get top k negative examples
        # probs: (bsz, vocab_size)
        probs = self.get_masked_probs(type_masks, logits)

        sorted_indices = torch.argsort(probs, dim=1, descending=True)

        if mode == "fit":
            # fit mode needs groud truth labels
            # top_k_indices: (bsz, top_k)
            top_k_indices = sorted_indices[ : , :self.cfg.top_k-1 ]
            top_k_indices = torch.cat( (target.unsqueeze(1), top_k_indices), dim=1)
            if self.cfg.contrastive_learning == True:
                # contrastive learning
                top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )
        elif mode == "infer":
            # infer mode doesn't need ground truth labels
            # top_k_indices: (bsz, top_k)
            top_k_indices = sorted_indices[:, :self.cfg.top_k]
            if self.cfg.contrastive_learning == True:
                # contrastive learning
                top_k_indices = torch.cat( (top_k_indices, torch.full_like(target, tgt_dict.mask()).unsqueeze(1)), dim=1 )

        # prev_output_tokens: (bsz, top_k, tgt_len)
        prev_output_tokens = prev_output_tokens.unsqueeze(1).expand(-1, top_k_indices.size(1), -1).clone()

        if self.training == True:
            prev_output_tokens.scatter_(
                -1,
                target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, top_k_indices.size(1), -1),
                top_k_indices.unsqueeze(-1)
            )
        else:
            prev_output_tokens.scatter_(
                -1,
                target_position.unsqueeze(-1).unsqueeze(-1).expand(-1, top_k_indices.size(1), -1),
                top_k_indices.unsqueeze(-1)
            )

        # target: (bsz, top_k) -> (bsz * top_k)
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

        if self.training == False:
            # (bsz * top_k, 1, embed_dim)
            target_position = target_position.unsqueeze(-1).expand(-1, top_k_indices.size(1)).reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.cfg.decoder.embed_dim)
        else:
            target_position = target_position.unsqueeze(-1).expand(-1, top_k_indices.size(1)).reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.cfg.decoder.embed_dim)            

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
        )
        if self.cfg.share_decoder_input_output_embed:
            origin_decoder_out_logits = decoder_out[0]
            # print("origin_decoder_out_logits: ", origin_decoder_out_logits.size())
            gather_decoder_out_logits = origin_decoder_out_logits.gather(1, top_k_indices.view(-1).unsqueeze(-1))
            # print("gather_decoder_out_logits: ", gather_decoder_out_logits.size())
            decoder_out = (gather_decoder_out_logits,) + decoder_out[1:]
        return (decoder_out + (target, top_k_indices) )

    @classmethod 
    def get_output_projection(cls, cfg):
        if cfg.output_projection == "linear":
            return nn.Linear(cfg.decoder.embed_dim, 1, bias=False)
            # return RobertaClassificationHead(
            #     cfg.decoder.embed_dim,
            #     cfg.decoder.embed_dim,
            #     1,
            #     "tanh",
            #     pooler_dropout=0.1)
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

@register_model_architecture("rerank_transformer_dynamic_anchors", "rerank_transformer_dynamic_anchors")
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
