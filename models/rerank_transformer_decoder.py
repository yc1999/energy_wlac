from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase
from models.gwlan_transformer_decoder import GWLANTransformerDecoder
from typing import Any, Dict, List, Optional
from torch import Tensor
import torch

class RerankTransformerDecoder(GWLANTransformerDecoder):

    def forward(
        self,
        prev_output_tokens,
        tgt_position_ids,
        target_position,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        synonyms: Optional[torch.Tensor] = None,    # customized parameter
        hypernyms: Optional[torch.Tensor] = None,    # customized parameter
        segment_labels: Optional[torch.Tensor] = None,    # customized parameter
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            tgt_position_ids,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            synonyms=synonyms, # customized parameter
            target_position=target_position,    # customized parameter
            hypernyms=hypernyms,    # customized parameter
            segment_labels=segment_labels,    # customized parameter
        )

        if not features_only:
            # x: (bsz*topk, tgt_len, embed_dim) -> (bsz*topk, 1, embed_dim) -> (bsz*topk, embed_dim)
            # target_position: (bsz*topk, 1)
            x = x.gather(1, target_position)
            x.squeeze_(1)
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        tgt_position_ids,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        synonyms: Optional[torch.Tensor] = None,    # customized parameter
        target_position: Optional[torch.Tensor] = None,    # customized parameter
        hypernyms: Optional[torch.Tensor] = None,    # customized parameter
        segment_labels: Optional[torch.Tensor] = None,    # customized parameter
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            tgt_position_ids,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            synonyms,
            target_position,
            hypernyms,
            segment_labels,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        tgt_postion_ids,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        synonyms: Optional[torch.Tensor] = None,    # customized parameter
        target_position: Optional[torch.Tensor] = None,    # customized parameter
        hypernyms: Optional[torch.Tensor] = None,    # customized parameter
        segment_labels: Optional[torch.Tensor] = None,    # customized parameter
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        assert tgt_postion_ids is not None
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state, positions=tgt_postion_ids
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # After embed tokens, if synonyms is not None, then we need to do information gather:
        if synonyms is not None and hypernyms is not None:
            # x_synonyms: (bsz*topk, tgt_len, embed_dim)
            x_synonyms = self.synonym_info_gather_layer(
                input_tokens = synonyms,
                origin_embeddings = x,
                index = target_position,
                embed_tokens_layer = self.embed_tokens,
            )
            # x_hypernyms: (bsz*topk, tgt_len, embed_dim)
            x_hypernyms = self.hypernym_info_gather_layer(
                input_tokens = hypernyms,
                origin_embeddings = x,
                index = target_position,
                embed_tokens_layer = self.embed_tokens,
            )
            # We only get the embedding of the target word:
            # x_synonyms: (bsz*topk, 1, embed_dim)
            x_synonyms = torch.gather(input=x_synonyms, dim=1, index=target_position)
            x_hypernyms = torch.gather(input=x_hypernyms, dim=1, index=target_position)
            
            # x_hybrid: (bsz*topk, 1, embed_dim*2) -> (bsz*topk, 1, embed_dim)
            x_hybrid = torch.cat([x_synonyms, x_hypernyms], dim=-1)
            x_hybrid = self.fusion_layer(x_hybrid)
            
            x_hybrid = torch.scatter(input=x, dim=1, index=target_position, src=x_hybrid)

            x = x + x_hybrid
        
        elif synonyms is not None:
            # residual = x
            x = self.local_info_gather_layer(
                input_tokens = synonyms,
                origin_embeddings = x,
                index = target_position,
                embed_tokens_layer = self.embed_tokens,
            )
            # x = self.dropout_module(x)
            # x = x + residual
            # print("after attention:", torch.mean(torch.norm(x,dim=-1)).item())  
        elif hypernyms is not None:
            x = self.hypernym_info_gather_layer(
                input_tokens = hypernyms,
                origin_embeddings = x,
                index = target_position,
                embed_tokens_layer = self.embed_tokens,
            )
            # x = x + x_with_hypernyms
         
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if hasattr(self, "segment_embeddings"):
            x = x + self.segment_embeddings(segment_labels)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):

            # self_attn_mask_bool: (batch_size, tgt_len)
            self_attn_mask_bool = prev_output_tokens.eq(self.dictionary.mask())
            # self_attn_mask_bool: (batch_size, tgt_len, tgt_len)
            self_attn_mask_bool = self_attn_mask_bool.unsqueeze(1).expand(bs, slen, slen)
            # self_attn_mask_bool: (batch_size, num_heads, tgt_len, tgt_len)
            self_attn_mask_bool = self_attn_mask_bool.unsqueeze(1).expand(bs, self.cfg.decoder.attention_heads, slen, slen)
            # self_attn_mask_bool: (batch_size * num_heads, tgt_len, tgt_len)
            self_attn_mask_bool = self_attn_mask_bool.reshape(-1, slen, slen)
            self_attn_mask = torch.zeros(bs * self.cfg.decoder.attention_heads, slen, slen).to(prev_output_tokens.device).masked_fill(self_attn_mask_bool, -float('inf'))                

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}