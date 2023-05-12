# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import json
import numpy as np
import torch
from multiprocessing import Pool

from collections import OrderedDict

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    RawLabelDataset, #zhangying
    ResamplingDataset, #zhangying
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.permutation_utils import make_span_perm


@register_task('masked_permutation_lm')
class MaskedPermutationLMTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--pred-prob', default=0.15, type=float,
                            help='probability for tokens prediction')
        parser.add_argument('--rand-prob', default=0.10, type=float,
                            help='probability for random input')
        parser.add_argument('--keep-prob', default=0.10, type=float,
                            help='probability for keep input')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--input-mode', default='mpnet', choices=['mlm', 'plm', 'mpnet'], 
                            help='Choose the input format for different tasks')
        parser.add_argument('--max-gram', default=1, type=int,
                            help='The maximum n-gram for whole word mask. It is setup with --max-whole-words')
        parser.add_argument('--eval-input-mode', default='mpnet', choices=['mlm', 'plm', 'mpnet'], 
                            help='Choose the input format for different tasks')

        #zhangying
        parser.add_argument('--discriminative-loss', default=False, action='store_true',
                            help='whether to minimize the unlikelihood with respect to negative samples')
        parser.add_argument('--discriminative-type', type=str, default="edu", choices=['edu', 'span', 'ns', 'rela', 'full', 'autoedu', 'autospan', 'autons', 'autorela'],
                            help='set specific task')
        parser.add_argument('--discriminative-size', default=5, type=int,
                            help='candidate size') 
        parser.add_argument('--label-embedding', type=str, default="normal", choices=['enhance', 'concat', 'normal'],
                            help='Using Enhance, Concat, or normal label embedding, as indicated in Section 5.1.3 of our paper')
        parser.add_argument('--train-cand-path', type=str, 
                            help='candidates for train data')
        parser.add_argument('--valid-cand-path', type=str, 
                            help='candidates for valid data')
        parser.add_argument('--reference', type=str, default='reference_raw.txt', help='reference file')
        parser.add_argument('--prediction', type=str, default='output_raw.txt', help='prediction file')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.mask_idx = dictionary.add_symbol('<mask>')
        self.dictionary = dictionary
        self.seed = args.seed
        self._max_position = None
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # ------------------------------------------------------#
        # zhang ying
        if self.args.discriminative_loss:
            if split == "train":
                bi_cand_file = self.args.train_cand_path + self.args.prediction
                discriminative_size = self.args.discriminative_size
            elif split == "valid":
                bi_cand_file = self.args.valid_cand_path + self.args.prediction
                discriminative_size = 5 #self.args.discriminative_size
            else:
                print("split error")
                exit()

        # ------------------------------------------------------#

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)
 
        dataset_impl = 'mmapcached' if self.args.discriminative_loss else self.args.dataset_impl
        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            dataset_impl, 
            combine=combine,
        )

        # zhang ying
        if self.args.discriminative_loss:
            print("loading candidate from ", bi_cand_file)
            train = []
            with open(bi_cand_file) as f:
                train = f.readlines()

            e = MultiprocessingEdit()
            pool = Pool(60,initializer=e.initializer(discriminative_size))
            encoded_lines = pool.imap(e.load_cands, zip(dataset.cache, train))
            for i, cands in enumerate(encoded_lines, start=1):
                for j, cand in enumerate(cands):
                    dataset.add_item(np.array(cand))
            
            pool.close()
            pool.join()
            cand_id = RawLabelDataset(dataset.cache_candid)

        else:
            cand_id = None

        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith('madeupword'):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(list(
                    map(is_beginning_of_word, range(len(self.source_dictionary)))
                ))
        else:
            mask_whole_words = None

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )

        print('| loaded {} batches from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        src_dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        self.datasets[split] = MaskedDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                    'cand_id': cand_id, #zhangying
                },
                sizes=[src_dataset.sizes],
                dictionary=self.dictionary,
                args=self.args,
                mask_whole_words=mask_whole_words,
            )
        #zhangying
        if True:
            with data_utils.numpy_seed(self.args.seed + epoch):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(self.datasets[split],
                sort_order=[
                    shuffle,
                    src_dataset.sizes,
                ],
            )
    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = MaskedDatasetInference(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_dataset, reduce=True),
            },
            sizes=src_lengths,
            dictionary=self.dictionary,
            args=self.args,
        )
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        return self._max_position

#zhangying
class MultiprocessingEdit(object):

    def __init__(self):
        pass

    def initializer(self, discriminative_size):
        self.discriminative_size = discriminative_size

    def load_cands(self, input):
        gold, cands = input
        gold = gold.tolist()
        new_cands = []
        pred = json.loads(cands)
        cands = pred["text_binary"] 
        new_cands += cands[:self.discriminative_size]
        return new_cands


class MaskedDataset(NestedDictionaryDataset):
    def __init__(self, defn, sizes=None, dictionary=None, args=None, mask_whole_words=None):
        super().__init__(defn, sizes)
        self.mask_idx = dictionary.add_symbol('<mask>')
        self.pred_prob  = args.pred_prob
        self.keep_prob  = args.keep_prob
        self.rand_prob  = args.rand_prob
        self.vocab = dictionary

        weights = np.ones(len(self.vocab))
        weights[:self.vocab.nspecial] = 0
        self.weights = weights / weights.sum()

        self.mask_whole_words = mask_whole_words
        self.input_mode = args.input_mode

        self.max_gram = args.max_gram
        
        # Generate a static n-gram template for faster training
        mask_ngram = dict()
        for i in range(1, args.tokens_per_sample + 1):
            template = []
            r = i
            for j in range(self.max_gram, 1, -1):
                cnt = int(i / self.max_gram / j)
                template.extend([j for _ in range(cnt)])
                r = r - cnt * j
            template.extend([1 for _ in range(r)])

            mask_ngram[i] = np.array(template)

        self.mask_ngram = mask_ngram
        #zhangying
        self.label_embedding = args.label_embedding
        self.discriminative_type = args.discriminative_type

    def collater(self, samples):
        samples = super().collater(samples)

        if len(samples) == 0:
            return {}

        torch.set_printoptions(profile='full')
        src_tokens = samples['net_input']['src_tokens']
        sz = src_tokens.size()

        pred_size = round(sz[1] * self.pred_prob)

        if pred_size == 0:
            pred_size = sz[1]
        
        if self.mask_whole_words is not None:
            positions = torch.stack([self.span_perm(src_tokens[i], pred_size) for i in range(sz[0])])
        else:
            positions = torch.stack([torch.randperm(sz[1]) for i in range(sz[0])])

        src_tokens, targets = self.permute_inputs(src_tokens, positions), None

        mask_range = range(sz[1] - pred_size, sz[1])
        if self.input_mode == 'mlm':
            targets = src_tokens[:, mask_range].contiguous()
            src_tokens[:, mask_range] = self.mask_perm(targets.clone(), self.mask_idx)
        elif self.input_mode == 'plm':
            # PLM does not use 8:1:1 ? 
            targets = src_tokens[:, mask_range].contiguous()
            src_tokens = torch.cat((src_tokens, torch.full_like(targets, self.mask_idx)), dim=1)
            positions = torch.cat((positions, positions[:, mask_range]), dim=1)
        else:
            targets = src_tokens[:, mask_range].contiguous()
            masked_tokens = self.mask_perm(targets.clone(), self.mask_idx)
            src_tokens = torch.cat((src_tokens, masked_tokens, masked_tokens), dim=1)
            positions = torch.cat((positions, positions[:, mask_range], positions[:, mask_range]), dim=1)
            #[0][1] is used for mpnet, [2] is used for mlm

        if self.label_embedding == 'concat':
            src_lengths = samples['net_input']['src_lengths'].clone()

            if self.discriminative_type == 'edu':
                defn = [106, 1028, 4736, 15156, 3201, 2028, 2000, 10128, 2315, 5995, 2001, 1041, 15156, 3396]
                defn_size = len(defn)
                defn = src_tokens.new_tensor(defn)
                defn_positions = torch.arange(defn_size).expand(sz[0], -1) + src_lengths.view(-1,1).expand(-1, defn_size)
                defn = defn.view(1, -1).expand(sz[0], -1)

            else:
                defn = None

            if defn is not None:
                sz = positions.size()
                pad_position = (src_tokens.view(-1) == 1).nonzero(as_tuple=False)
                positions.view(-1)[pad_position] += defn_size

                samples['net_input']['src_lengths'] += defn.size(1)
                positions = torch.cat((defn_positions, positions), dim=1)
                src_tokens = torch.cat((defn, src_tokens), dim=1)

            

        samples['target'] = targets
        samples['net_input']['positions'] = positions
        samples['net_input']['src_tokens'] = src_tokens
        samples['net_input']['pred_size'] = targets.size(1)
        return samples

    def span_perm(self, x, pred_size=None):
        # Permutation for span mask, faster than fairseq original implementation
        word_begins_mask = self.mask_whole_words.gather(0, x)
        word_begins_idx = word_begins_mask.nonzero(as_tuple=False).view(-1).tolist()
        
        if self.max_gram == 1: 
            # Only whole word mask, slightly faster than using n-gram and hardly affect accuracy
            ids = word_begins_idx
        else:
            sz = len(word_begins_idx)
            ngram = self.mask_ngram[sz].copy()
            np.random.shuffle(ngram)
            i, ids = 0, []

            for g in ngram:
                ids.append(word_begins_idx[i])
                i = i + g

        sz = len(ids)
        perm = np.random.permutation(sz)
        ids.append(x.size(0))
        
        span_perm = make_span_perm(perm, ids, x.size(0))
        if pred_size is not None:
            # Shuffle Predicted Part again for 
            np.random.shuffle(span_perm[-pred_size:])
        return torch.from_numpy(span_perm)

    def mask_perm(self, tokens, mask_idx=None):
        if mask_idx is None:
            mask_idx = self.mask_idx
        mask_prob = 1.0 - self.rand_prob - self.keep_prob
        mask_indices = torch.bernoulli(torch.full(tokens.shape, mask_prob)).bool()
        #random_indices = torch.bernoulli(torch.full(tokens.shape, self.rand_prob / (1.0 - mask_prob))).bool() & ~mask_indices
        tokens[mask_indices] = mask_idx
        #tokens[random_indices] = self.generate_random_tensor(random_indices.sum().tolist()).to(tokens.device)
        return tokens

    def make_perm(self, sz, pred_size):
        perm = torch.randperm(sz)
        perm[:sz - pred_size] = perm[:sz - pred_size].sort()[0]
        return perm

    def permute_inputs(self, inputs, positions):
        sz = inputs.size()
        offset = torch.arange(0, sz[0] * sz[1], sz[1])
        index = positions + offset.unsqueeze_(1)
        return inputs.reshape(-1)[index]

    def generate_random_tensor(self, sz):
        return torch.from_numpy(
            np.random.choice(len(self.vocab), sz, p=self.weights)        
        )

class MaskedDatasetInference(NestedDictionaryDataset):
    def __init__(self, defn, sizes=None, dictionary=None, args=None, mask_whole_words=None):
        super().__init__(defn, sizes)
        self.mask_idx = dictionary.add_symbol('<mask>')
        self.pred_prob  = args.pred_prob
        self.keep_prob  = args.keep_prob
        self.rand_prob  = args.rand_prob
        self.vocab = dictionary

        weights = np.ones(len(self.vocab))
        weights[:self.vocab.nspecial] = 0
        self.weights = weights / weights.sum()

        self.mask_whole_words = mask_whole_words
        self.input_mode = args.eval_input_mode if 'eval_input_mode' in args else args.input_mode

        self.max_gram = args.max_gram
        
        # Generate a static n-gram template for faster training
        mask_ngram = dict()
        for i in range(1, args.tokens_per_sample + 1):
            template = []
            r = i
            for j in range(self.max_gram, 1, -1):
                cnt = int(i / self.max_gram / j)
                template.extend([j for _ in range(cnt)])
                r = r - cnt * j
            template.extend([1 for _ in range(r)])

            mask_ngram[i] = np.array(template)

        self.mask_ngram = mask_ngram

        self.label_embedding = args.label_embedding if 'label_embedding' in args else 'normal'
        self.discriminative_type = args.discriminative_type

    def collater(self, samples):
        samples = super().collater(samples)
        bsz = 4096 * 10
        if len(samples) == 0:
            return {}

        new_samples = []
        for i in range(len(samples['id'])):
            id = samples['id'][i]
            src_lengths = samples['net_input']['src_lengths'][i]
            assert src_lengths < bsz
            src_tokens = samples['net_input']['src_tokens'][i][:src_lengths]

            positions = src_tokens.new_tensor(list(range(src_lengths)))
            sub_samples = []

            step = 0
            if self.label_embedding == 'concat':
                _src_lengths = src_lengths.clone()
                sz = positions.size()

                if self.discriminative_type == 'edu':
                    defn = [106, 1028, 4736, 15156, 3201, 2028, 2000, 10128, 2315, 5995, 2001, 1041, 15156, 3396]
                    defn_size = len(defn)
                    defn = src_tokens.new_tensor(defn)
                else:
                    defn = None

                if defn is not None:
                    defn_positions = torch.arange(defn_size) + _src_lengths

                    pad_position = (src_tokens.view(-1) == 1).nonzero(as_tuple=False)
                    positions.view(-1)[pad_position] += defn_size

                    src_lengths += defn_size
                    positions = torch.cat((defn_positions, positions), dim=0)
                    src_tokens = torch.cat((defn, src_tokens), dim=0)
                    step += defn_size

            while step < src_lengths - 1:# for token in the joint sentence (<pad> will influence results)
                d = OrderedDict()
                sub_tokens = []
                sub_targets = []
                sub_positions = []
                for j in range(0, min(bsz // src_lengths, src_lengths - step).item()):
                    s = torch.cat((src_tokens[step+1+j:], src_tokens[:step+1+j]))
                    t = s[-1]
                    p = torch.cat((positions[step+1+j:], positions[:step+1+j]))

                    if self.input_mode == 'mlm':
                        s[-1] = self.mask_perm(t.clone(), self.mask_idx)
                    elif self.input_mode == 'plm':
                        # PLM does not use 8:1:1 ? 
                        s = torch.cat((s, torch.full_like(t, self.mask_idx).view(1)), dim=0)
                        p = torch.cat((p, p[-1].view(1)), dim=0)
                    else:
                        masked_tokens = self.mask_perm(t.clone(), self.mask_idx).view(1)
                        s = torch.cat((s, masked_tokens, masked_tokens), dim=0)
                        p = torch.cat((p, p[-1].view(1), p[-1].view(1)), dim=0)

                    sub_tokens.append(s)
                    sub_targets.append(t)
                    sub_positions.append(p)

                step += bsz//src_lengths
                if len(sub_targets) !=0 :
                    d['target'] = torch.stack(sub_targets).unsqueeze(1)
                    d['net_input'] = OrderedDict()
                    d['net_input']['positions'] = torch.stack(sub_positions)
                    d['net_input']['src_tokens'] = torch.stack(sub_tokens)
                    d['net_input']['pred_size'] = d['target'].size(1)
                    d['net_input']['src_lengths'] = src_lengths.expand(d['target'].size(0))
                    sub_samples.append(d)


            new_samples.append(sub_samples)
        return new_samples

    def mask_perm(self, tokens, mask_idx=None):
        if mask_idx is None:
            mask_idx = self.mask_idx
        mask_prob = 1.0
        mask_indices = torch.bernoulli(torch.full(tokens.shape, mask_prob)).bool()
        tokens[mask_indices] = mask_idx
        return tokens

    def generate_random_tensor(self, sz):
        return torch.from_numpy(
            np.random.choice(len(self.vocab), sz, p=self.weights)        
        )