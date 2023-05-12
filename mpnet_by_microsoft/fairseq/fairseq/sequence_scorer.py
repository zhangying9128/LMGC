# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import sys

from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None):
        self.pad = tgt_dict.pad()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0

    @torch.no_grad()
    #zhangying
    def generate(self, models, samples, evel_input_mode,  **kwargs):
        """Score a batch of translations."""

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        hypos = []
        #torch.set_printoptions(profile='full')

        #zhangying
        for i in range(len(samples)):
            avg_probs = None
            ref = None
            for j in range(len(samples[i])):
                sample = samples[i][j]
                net_input = sample['net_input']
                orig_target = sample['target']
                # compute scores for each model in the ensemble

                for model in models:
                    model.eval()
                    #decoder_out = model.forward(**net_input)
                    #zhangying
                    if evel_input_mode == 'mpnet':
                        decoder_out = model.task_compute(
                            #task='plm',
                            task='mpnet',
                            return_mlm=False,
                            **net_input, 
                        )
                    else:
                        decoder_out = model.task_compute(
                            task='plm',
                            **net_input, 
                        )
                    curr_prob = F.log_softmax(decoder_out, dim=-1)
                    probs = gather_target_probs(curr_prob, orig_target)
                    probs = probs.view(sample['target'].shape)

                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs = torch.cat((avg_probs,probs), dim=0)

                    if ref is None:
                        ref = orig_target
                    else:
                        ref = torch.cat((ref, orig_target), dim=0)

                    if len(models) > 1:
                        avg_probs.div_(len(models))
                        avg_probs.log_()

            score = avg_probs.sum() / avg_probs.size(0)
            hypos.append([{
                'tokens': ref.view(-1),
                'score': score,
                'positional_scores': avg_probs.view(-1),
            }])

        return hypos
