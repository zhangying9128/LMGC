# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from fairseq import utils
from fairseq.models.roberta import (
    RobertaModel, 
    RobertaLMHead,
    roberta_base_architecture,
    roberta_large_architecture,
)
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


@register_model('mpnet')
class MPNet(RobertaModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        #zhangying
        self.label_embedding = args.label_embedding if "label_embedding" in args else 'normal'
        self.discriminative_type = self.args.discriminative_type


    def task_compute(self, task='mlm', **kwargs):
        if task == 'mlm':
            return self.compute_mlm(**kwargs)
        elif task == 'plm':
            return self.compute_plm(**kwargs)
        else:
            return self.compute_mpnet(**kwargs)

    def compute_mlm(self, src_tokens, src_lengths, positions, pred_size, **kwargs):
        sz = src_tokens.size(1)
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions, self.label_embedding, self.discriminative_type)
        x = reverse_tensor(emb)
        positions_bias = self.encode_relative_emb(self.decoder.sentence_encoder, positions)
        for layer in self.decoder.sentence_encoder.layers:
            x, _ = layer(x, positions_bias=positions_bias)
        x = self.maybe_final_norm(self.decoder.sentence_encoder, x)
        x = reverse_tensor(x)
        x = self.output_layer(x[:, sz-pred_size:])
        return x

    def compute_plm(self, src_tokens, src_lengths, positions, pred_size, **kwargs):
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions, self.label_embedding, self.discriminative_type)
        x = reverse_tensor(emb)
        c, q = split_tensor(x, pred_size)
        content_position_bias = self.encode_relative_emb(
            self.decoder.sentence_encoder, positions[:, :-pred_size]
        )
        if content_position_bias is not None:
            query_position_bias = content_position_bias[:, -pred_size:].contiguous()
        else:
            query_position_bias = None

        sz = c.size(0)
        query_mask, content_mask = make_query_and_content_mask(src_tokens, sz, pred_size, kind='PLM')
        for i, layer in enumerate(self.decoder.sentence_encoder.layers):
            c, q = encode_two_stream_attn(
                layer, c, q, content_mask, query_mask, content_position_bias, query_position_bias,
            )

        q = self.maybe_final_norm(self.decoder.sentence_encoder, q)
        q = reverse_tensor(q)
        x = self.output_layer(q)
        return x

    def compute_mpnet(self, src_tokens, src_lengths, positions, pred_size, return_mlm=False, **kwargs):
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions, self.label_embedding, self.discriminative_type)
        x = reverse_tensor(emb)
        c, q = split_tensor(x, pred_size)
        content_position_bias = self.encode_relative_emb(self.decoder.sentence_encoder, positions[:, :-pred_size])
        if content_position_bias is not None:
            query_position_bias = content_position_bias[:, -pred_size:].contiguous()
        else:
            query_position_bias = None

        sz = c.size(0) - pred_size
        query_mask, content_mask = make_query_and_content_mask(src_tokens, sz, pred_size)

        for i, layer in enumerate(self.decoder.sentence_encoder.layers):
            c, q = encode_two_stream_attn(
                layer, c, q, content_mask, query_mask, content_position_bias, query_position_bias,
            )

        q = self.maybe_final_norm(self.decoder.sentence_encoder, q)
        q = reverse_tensor(q)

        x = self.output_layer(q)

        if return_mlm is True:
            c = c[-pred_size:]
            c = self.maybe_final_norm(self.decoder.sentence_encoder, c)
            c = reverse_tensor(c)
            c = self.output_layer(c)
            return x, c

        return x

    @staticmethod
    def encode_emb(self, src_tokens, positions=None, label_embedding='normal', discriminative_type=None):
        x = self.embed_tokens(src_tokens)

        #zhangying
        if label_embedding == 'enhance':
            bsz, tgt = src_tokens.size()
            x = x.view(bsz * tgt, -1)
            src_tokens = src_tokens.view(-1)
            label_defn = None
            if discriminative_type in ['edu', 'autoedu']:
                label_defn = {106:[ 4736, 15156, 3201, 2028, 2000, 10128, 2315, 5995, 2001, 1041, 15156, 3396]} #'elementary discourse units are the minimal building blocks of a discourse tree'

            elif discriminative_type in ['span', 'autospan']:
                label_defn = {5: [2000, 2711, 2001, 8491],
                              6: [2000, 2207, 2001, 8491]}

            elif discriminative_type in ['ns', 'autons']:
                label_defn = {5: [2000, 2711, 2001, 1041, 4641, 2034, 4285, 3542, 2001, 2596],  #the start of a supporting or background piece of information
                              6: [2000, 2207, 2001, 1041, 4641, 2034, 4285, 3542, 2001, 2596],  #the end of a supporting or background piece of information
                              7: [2000, 2711, 2001, 1041, 2066, 16187, 11642, 2034, 6831, 3542, 2001, 2596], #the start of a more sal ##ient or essential piece of information
                              8: [2000, 2207, 2001, 1041, 2066, 16187, 11642, 2034, 6831, 3542, 2001, 2596]} #the end of a more sal ##ient or essential piece of information

            elif discriminative_type in ['rela', 'autorela']:
                label_defn = {5: [2000, 2711, 2001, 2016, 18890, 29450, 1014, 2016, 18890, 29450, 5840, 2123, 3626, 2002, 14962, 12111, 2001, 2992, 4617],
                              6: [2000, 2207, 2001, 2016, 18890, 29450, 1014, 2016, 18890, 29450, 5840, 2123, 3626, 2002, 14962, 12111, 2001, 2992, 4617],
                              7: [2000, 2711, 2001, 4285, 2034, 25656],
                              8: [2000, 2207, 2001, 4285, 2034, 25656],
                              9: [2000, 2711, 2001, 3430, 2034, 2769],
                              10: [2000, 2207, 2001, 3430, 2034, 2769],
                              11: [2000, 2711, 2001, 7835, 1014, 12161, 1014, 23327, 2034, 10821],
                              12: [2000, 2207, 2001, 7835, 1014, 12161, 1014, 23327, 2034, 10821],
                              13: [2000, 2711, 2001, 4654, 1014, 25617, 1014, 9534, 3440, 11920, 2034, 4732],
                              14: [2000, 2207, 2001, 4654, 1014, 25617, 1014, 9534, 3440, 11920, 2034, 4732],
                              15: [2000, 2711, 2001, 5692, 7193, 1014, 14802, 5692, 2011, 2173, 2064, 2251, 2074, 9816, 1016, 4054, 1014, 2013, 2954, 1041, 5692, 3516, 15156, 16095, 1014, 2111, 2008, 2025, 1014, 1009, 2178, 1014, 2100],
                              16: [2000, 2207, 2001, 5692, 7193, 1014, 14802, 5692, 2011, 2173, 2064, 2251, 2074, 9816, 1016, 4054, 1014, 2013, 2954, 1041, 5692, 3516, 15156, 16095, 1014, 2111, 2008, 2025, 1014, 1009, 2178, 1014, 2100],
                              17: [2000, 2711, 2001, 3453, 7879, 21227, 1014, 3453, 7879, 21227, 3644, 3567, 2596, 2034, 4755, 2004, 2397, 9379, 1041, 2204, 2240, 4149],
                              18: [2000, 2207, 2001, 3453, 7879, 21227, 1014, 3453, 7879, 21227, 3644, 3567, 2596, 2034, 4755, 2004, 2397, 9379, 1041, 2204, 2240, 4149],
                              19: [2000, 2711, 2001, 9589, 3676, 1014, 9589, 3676, 2560, 2233, 2899, 2004, 3627, 2000, 9596, 2001, 2000, 4899, 22856, 3554, 3667, 2112, 3655],
                              20: [2000, 2207, 2001, 9589, 3676, 1014, 9589, 3676, 2560, 2233, 2899, 2004, 3627, 2000, 9596, 2001, 2000, 4899, 22856, 3554, 3667, 2112, 3655],
                              21: [2000, 2711, 2001, 9316, 1014, 7617, 1014, 7095, 2034, 7619],
                              22: [2000, 2207, 2001, 9316, 1014, 7617, 1014, 7095, 2034, 7619],
                              23: [2000, 2711, 2001, 3354, 1014, 7530, 2034, 3118],
                              24: [2000, 2207, 2001, 3354, 1014, 7530, 2034, 3118],
                              25: [2000, 2711, 2001, 2866, 1014, 2866, 3401, 2074, 4070, 2001, 5907, 3256, 2034, 2718, 4831, 2094, 2000, 3201],
                              26: [2000, 2207, 2001, 2866, 1014, 2866, 3401, 2074, 4070, 2001, 5907, 3256, 2034, 2718, 4831, 2094, 2000, 3201],
                              27: [2000, 2711, 2001, 9994, 2034, 20652, 2079, 1041, 4122, 1014, 7341, 1014, 6606, 1014, 3153, 2034, 9534, 8570, 4187, 2009, 14574, 2079, 2074, 3129],
                              28: [2000, 2207, 2001, 9994, 2034, 20652, 2079, 1041, 4122, 1014, 7341, 1014, 6606, 1014, 3153, 2034, 9534, 8570, 4187, 2009, 14574, 2079, 2074, 3129],
                              29: [2000, 2711, 2001, 3295, 5580, 1014, 3164, 3441, 1014, 4865, 3437, 1014, 8480, 7619, 2034, 17875, 2393, 3164],
                              30: [2000, 2207, 2001, 3295, 5580, 1014, 3164, 3441, 1014, 4865, 3437, 1014, 8480, 7619, 2034, 17875, 2393, 3164],
                              31: [2000, 2711, 2001, 12658, 2034, 2721, 3690, 3676],
                              32: [2000, 2207, 2001, 12658, 2034, 2721, 3690, 3676],
                              33: [2000, 2711, 2001, 8150, 2011, 15854, 2348, 1014, 2081, 1014, 2048, 2034, 2016, 2000, 2172, 2055],
                              34: [2000, 2207, 2001, 8150, 2011, 15854, 2348, 1014, 2081, 1014, 2048, 2034, 2016, 2000, 2172, 2055],
                              35: [2000, 2711, 2001, 8480, 2693],
                              36: [2000, 2207, 2001, 8480, 2693],
                              37: [2000, 2711, 2001, 6975, 2012, 2028, 4421, 2015, 8044, 28437, 2700, 10877],
                              38: [2000, 2207, 2001, 6975, 2012, 2028, 4421, 2015, 8044, 28437, 2700, 10877],
                              39: [2000, 2711, 2001, 8491],
                              40: [2000, 2207, 2001, 8491],
                              41: [2000, 2711, 2001, 6975, 2094, 2052, 2516, 1015, 5520, 3037, 2047, 5463, 2015, 2023, 26627, 5820, 11079, 2034, 6691, 20090, 2393],
                              42: [2000, 2207, 2001, 6975, 2094, 2052, 2516, 1015, 5520, 3037, 2047, 5463, 2015, 2023, 26627, 5820, 11079, 2034, 6691, 20090, 2393]}

            elif discriminative_type in ['full']:
                labels = {'Attribution': [ 2016, 18890, 29450, 1014, 2016, 18890, 29450, 5840, 2123, 3626, 2002, 14962, 12111, 2001, 2992, 4617],
                              'Background': [ 4285, 2034, 25656],
                              'Cause': [ 3430, 2034, 2769],
                              'Comparison': [ 7835, 1014, 12161, 1014, 23327, 2034, 10821],
                              'Condition': [ 4654, 1014, 25617, 1014, 9534, 3440, 11920, 2034, 4732],
                              'Contrast': [ 5692, 7193, 1014, 14802, 5692, 2011, 2173, 2064, 2251, 2074, 9816, 1016, 4054, 1014, 2013, 2954, 1041, 5692, 3516, 15156, 16095, 1014, 2111, 2008, 2025, 1014, 1009, 2178, 1014, 2100],
                              'Elaboration': [ 3453, 7879, 21227, 1014, 3453, 7879, 21227, 3644, 3567, 2596, 2034, 4755, 2004, 2397, 9379, 1041, 2204, 2240, 4149],
                              'Enablement': [ 9589, 3676, 1014, 9589, 3676, 2560, 2233, 2899, 2004, 3627, 2000, 9596, 2001, 2000, 4899, 22856, 3554, 3667, 2112, 3655],
                              'Evaluation': [ 9316, 1014, 7617, 1014, 7095, 2034, 7619],
                              'Explanation': [ 3354, 1014, 7530, 2034, 3118],
                              'Joint': [ 2866, 1014, 2866, 3401, 2074, 4070, 2001, 5907, 3256, 2034, 2718, 4831, 2094, 2000, 3201],
                              'Manner-Means': [ 9994, 2034, 20652, 2079, 1041, 4122, 1014, 7341, 1014, 6606, 1014, 3153, 2034, 9534, 8570, 4187, 2009, 14574, 2079, 2074, 3129],
                              'Topic-Comment': [ 3295, 5580, 1014, 3164, 3441, 1014, 4865, 3437, 1014, 8480, 7619, 2034, 17875, 2393, 3164],
                              'Summary': [ 12658, 2034, 2721, 3690, 3676],
                              'Temporal': [ 8150, 2011, 15854, 2348, 1014, 2081, 1014, 2048, 2034, 2016, 2000, 2172, 2055],
                              'Topic-Change': [ 8480, 2693],
                              'Textual-Organization': [ 6975, 2012, 2028, 4421, 2015, 8044, 28437, 2700, 10877],
                              'span': [ 8491],
                              'Same-Unit': [ 6975, 2094, 2052, 2516, 1015, 5520, 3037, 2047, 5463, 2015, 2023, 26627, 5820, 11079, 2034, 6691, 20090, 2393],
                              'satellite': [1041, 4641, 2034, 4285, 3542, 2001, 2596],
                              'nucleus': [1041, 2066, 16187, 11642, 2034, 6831, 3542, 2001, 2596],
                              'left': [2000, 2711, 2001],
                              'right': [2000, 2207, 2001]}

                ns = ['nucleus', 'satellite']
                relations = ['Attribution', 'Background', 'Cause', 'Comparison', 'Condition', 'Contrast', 'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',  'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization', 'span', 'Same-Unit']
                directions = ['left', 'right']
                label_defn = {}
                for n in ns:
                    for relation in relations:
                        for direction in directions:
                            label_defn[5+len(label_defn)] = labels[direction] + labels[n] + labels[relation]

            if label_defn is not None:
                for key, defn in label_defn.items():
                    EDU_TOKEN = src_tokens.new_tensor(defn) 
                    embed_EDU_TOKEN = self.embed_tokens(EDU_TOKEN).mean(dim=0)
                    batch_edu_index = torch.nonzero(src_tokens.eq(key),as_tuple=False) #'[SEP]'
                    x[batch_edu_index] = embed_EDU_TOKEN

            x = x.view(bsz, tgt, -1)

        if self.embed_scale is not None:
            x *= self.embed_scale
        if positions is not None:
            x += F.embedding(positions + 2, self.embed_positions.weight, self.padding_idx)

        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @staticmethod
    def maybe_final_norm(self, x):
        if self.emb_layer_norm is not None and self.normalize_before:
            return self.emb_layer_norm(x)
        return x

    @staticmethod
    def encode_relative_emb(self, positions):
        if not self.relative_attention_bias:
            return None
        qlen, klen = positions.size(1), positions.size(1)
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        
        relative_position = memory_position - context_position
        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(positions.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(0, 3, 1, 2).contiguous() # [bsz, head, qlen, klen]
        values = values.view(-1, qlen, klen)
        return values


def reverse_tensor(x):
    return x.transpose(0, 1)


def split_tensor(x, split_size):
    sz = x.size(0) - split_size
    return x[:sz].contiguous(), x[sz:].contiguous()

def encode_two_stream_attn(
    self, 
    c, 
    q, 
    content_mask: torch.Tensor = None, 
    query_mask: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
):
    def reuse_fn(x, residual):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x
    
    residual_c = c
    residual_q = q

    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)

    c, q = two_stream_self_attention(
        self.self_attn,
        query=[c, q],
        key=c,
        value=c,
        query_mask=query_mask,
        content_mask=content_mask,
        query_position_bias=query_position_bias,
        content_position_bias=content_position_bias,
    )

    c = reuse_fn(c, residual_c)
    q = reuse_fn(q, residual_q)

    return c, q


def two_stream_self_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    query_mask: torch.Tensor = None,
    content_mask: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
):
    c, q = query
    bsz, embed_dim = key.size(1), key.size(2)

    def transpose_fn(x):
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def fill_mask(attn_weights, attn_mask):
        return attn_weights.masked_fill(
            attn_mask.unsqueeze(0),
            float('-inf')
        )

    def attn_fn(_q, k, v, mask=None, bias=None):
        _q = transpose_fn(self.scaling * self.in_proj_q(_q))
        attn_weights = torch.bmm(_q, k.transpose(1, 2))
        if bias is not None:
            attn_weights += bias
        if mask is not None:
            attn_weights = fill_mask(attn_weights, mask)
        attn_weights = utils.softmax(
            attn_weights, dim=-1,        
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        return self.out_proj(attn)

    k = transpose_fn(self.in_proj_k(key))
    v = transpose_fn(self.in_proj_v(value))
    c = attn_fn(c, k, v, mask=content_mask, bias=content_position_bias)
    q = attn_fn(q, k, v, mask=query_mask, bias=query_position_bias)
    return c, q


def make_query_and_content_mask(tensor, a, b, kind='MPLM'):
    ''' 1 = inf

        Content Mask:                                                                       
        | <-   PLM   -> |    | <-      MPNet    -> |    | <-      MPNet    -> |             
                               x x x x x x x m m m        x x x x m m m x x x
                               1 2 3 4 5 6 7 5 6 7        1 2 3 4 5 6 7 5 6 7
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
        [ 0 0 0 0 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 1 0 0 0 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 1 1 0 0 0 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        Query Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |    | <-      MPNet    -> |
   m    [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
   m    [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]    [ 0 0 0 0 1 0 0 0 1 1 ]
   m    [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]    [ 0 0 0 0 1 1 0 0 0 1 ]
    '''

    def make_query_mask():
        mask = torch.triu(torch.ones(b, b), 0)
        mask = (torch.ones(b, a - b), 1 - mask) if kind is 'PLM' else (torch.ones(b, a - b), 1 - mask, mask)
        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask():
        mask = [torch.zeros(a - b, b), torch.tril(torch.ones(b, b), 0)]
        if kind is not 'PLM':
            mask.append(torch.zeros(b, b))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(a, a - b), mask) if kind is 'PLM' else (torch.ones(a + b, a - b), mask, 1 - mask)
        return torch.cat(mask, dim=-1).eq(0)

    return make_query_mask().to(tensor.device), make_content_mask().to(tensor.device)
  

    ''' 1 = inf

        Content Mask:                                                                  task_compute     
        | <-   PLM   -> |    | <- MPNet -> |    | <- MPNet -> |             
                               x x x m m m        m m m x x x
                               1 2 3 1 2 3        1 2 3 1 2 3
        [ 0 0 0 0 1 1 1 ]    [ 1 1 1 0 0 0 ]    [ 0 1 1 1 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 1 1 1 0 0 0 ]    [ 1 0 1 0 1 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 1 1 1 0 0 0 ]    [ 1 1 0 0 0 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 1 1 1 0 0 0 ]    [ 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 1 1 1 ]    [ 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 1 1 1 0 0 0 ]    [ 1 1 1 0 0 0 ]

        Query Mask:
        | <-   PLM   -> |    | <- MPNet -> |    | <- MPNet -> |
   m    [ 0 0 0 0 1 1 1 ]    [ 1 0 0 0 1 1 ]    [ 0 1 1 1 0 0 ]
   m    [ 0 0 0 0 0 1 1 ]    [ 0 1 0 1 0 1 ]    [ 1 0 1 0 1 0 ]
   m    [ 0 0 0 0 0 0 1 ]    [ 0 0 1 1 1 0 ]    [ 1 1 0 0 0 1 ]
    '''

    ''' 1 = inf

        Content Mask:                                                                       
        | <-   PLM   -> |    | <- MPNet -> |    | <- MPNet -> |             
                               x x x x x m         x x x x m x
                               1 2 3 4 5 5         1 2 3 4 5 5
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 0 ]    [ 0 0 0 0 0 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 0 ]    [ 0 0 0 0 0 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 0 ]    [ 0 0 0 0 0 1 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 0 ]    [ 0 0 0 0 0 1 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 ]    [ 0 0 0 0 0 1 ]
                             [ 0 0 0 0 1 0 ]    [ 0 0 0 0 1 0 ]
        Query Mask:
        | <-   PLM   -> |    | <- MPNet -> |    | <- MPNet -> |
   m    [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 0 ]    [ 0 0 0 0 0 1 ]

    '''


@register_model_architecture('mpnet', 'mpnet_base')
def mpnet_base_architecture(args):
    roberta_base_architecture(args)


@register_model_architecture('mpnet', 'mpnet_rel_base')
def mpnet_rel_base_architecture(args):
    args.use_relative_positions = getattr(args, 'use_relative_positions', True)
    mpnet_base_architecture(args)


@register_model_architecture('mpnet', 'mpnet_large')
def mpnet_large_architecture(args):
    roberta_large_architecture(args)

@register_model_architecture('mpnet', 'mpnet_rel_large')
def mpnet_rel_large_architecture(args):
    args.use_relative_positions = getattr(args, 'use_relative_positions', True)
    roberta_large_architecture(args)