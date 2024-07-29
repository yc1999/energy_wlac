"""
This file is used to gather information from the synonyms of a word. 
"""

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerNorm, MultiheadAttention

class LocalInfoGatherLayer(nn.Module):
    def __init__(self, embed_dim, padding_idx, dropout, num_heads=1):
        super(LocalInfoGatherLayer, self).__init__()

        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout_module = FairseqDropout(dropout)
        # TransformerDecoder貌似并没有使用layernorm_embedding，所以这里也不使用。
        # self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(self, input_tokens, origin_embeddings, index, embed_tokens_layer):
        """
        input_tokens: (batch_size, synonym_list_len)
        origin_embeddings: (batch_size, seq_len, embed_dim)
        index: (batch_size, 1, embed_dim)
        embed_tokens_layer: nn.Module (used to embed the input_tokens), which is the same as embed_tokens in the previous layer.
        """
        residual = origin_embeddings
          
        query = input_tokens[:, :1]
        # (batch_size, 1, embed_dim)
        query = embed_tokens_layer(query)
        # (batch_size, synonym_list_len, embed_dim)
        embeddings = embed_tokens_layer(input_tokens)

        # key_padding_mask: (batch_size, synonym_list_len)
        key_padding_mask = (input_tokens == self.padding_idx)
        attn_output, attn_output_weights = self.multihead_attn(query=query, key=embeddings, value=embeddings, key_padding_mask=key_padding_mask, attn_mask=None)

        # 得到attn_output之后，将attn_output置换之前的embedding
        new_embeddings = torch.scatter(input=origin_embeddings, dim=1, index=index, src=attn_output)

        new_embeddings = self.dropout_module(new_embeddings)
        new_embeddings = new_embeddings + residual

        return new_embeddings


class LocalInfoGatherLayerFairseq(nn.Module):
    def __init__(self, embed_dim, padding_idx, dropout, num_heads=1):
        super(LocalInfoGatherLayerFairseq, self).__init__()

        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        # fairseq的MultiHeadAttention是需要tgt_size在前面的，没有batch_first选项。
        self.multihead_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=dropout,
            encoder_decoder_attention=True,
        )
        self.dropout_module = FairseqDropout(dropout)
        # TransformerDecoder貌似并没有使用layernorm_embedding，所以这里也不使用。
        # self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(self, input_tokens, origin_embeddings, index, embed_tokens_layer):
        """
        input_tokens: (batch_size, synonym_list_len)
        origin_embeddings: (batch_size, seq_len, embed_dim)
        index: (batch_size, 1, embed_dim)
        embed_tokens_layer: nn.Module (used to embed the input_tokens), which is the same as embed_tokens in the previous layer.
        """
        residual = origin_embeddings

        query = input_tokens[:, :1]
        # (batch_size, 1, embed_dim)
        query = embed_tokens_layer(query)
        # (batch_size, synonym_list_len, embed_dim)
        embeddings = embed_tokens_layer(input_tokens)

        # key_padding_mask: (batch_size, synonym_list_len)
        key_padding_mask = (input_tokens == self.padding_idx)
        attn_output, attn_output_weights = self.multihead_attn(query=query.transpose(0,1), key=embeddings.transpose(0,1), value=embeddings.transpose(0,1), key_padding_mask=key_padding_mask, attn_mask=None)
        attn_output = attn_output.transpose(0,1)

        # 得到attn_output之后，将attn_output置换之前的embedding
        new_embeddings = torch.scatter(input=origin_embeddings, dim=1, index=index, src=attn_output)

        new_embeddings = self.dropout_module(new_embeddings)
        new_embeddings = new_embeddings + residual

        return new_embeddings