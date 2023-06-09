#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from transformers.data.processors.squad import SquadResult


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))


    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')


    if args.do_evaluate:
        validate(args, trainer, task, epoch_itr, valid_subsets)
        return 

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='tqdm', #'simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])
        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        preds, targets, all_results = [], [], []
        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size', 'targets', 'preds', 'starts', 'ends']:
                    continue
                extra_meters[k].update(v)
            if 'targets' in log_output:
                preds.append(log_output['preds'])
                targets.append(log_output['targets'])

            if 'starts' in log_output:
                for i in range(len(sample['id'])):
                    indice = sample['id'][i].tolist()
                    start  = log_output['starts'][i].cpu().tolist()
                    end    = log_output['ends'][i].cpu().tolist()
                    unique_id = task.features[indice].unique_id
                    result = SquadResult(unique_id, start, end)
                    all_results.append(result)

        if len(preds) > 0:
            preds = torch.cat(preds, 0).cpu().numpy()
            targets = torch.cat(targets, 0).cpu().numpy()
        else:
            preds = None
            targets = None
        
        if len(all_results) > 0:
            results = task.compute_predictions_logits(all_results)
            for k, v in results.items():
                print("({}, {})".format(k, v))
            exit()

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters, preds, targets)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
    
    #zhangying
    import os
    import copy
    import json
    from scripts.rsteval import ParsingMetrics
    from tqdm import tqdm 


    def _build_batchs(src_tokens, task):
        dataset = task.build_dataset_for_inference(src_tokens, [src.numel() for src in src_tokens])
        batch = dataset.collater([ dataset[i] for i in range(dataset.__len__())])
        return batch

    def evaluate_score(trainer, reference_dir):
        gen_args = copy.copy(args)
        gen_args.score_reference = True
        gold_path = os.path.join(reference_dir, gen_args.reference)
        pred_binary_path = os.path.join(reference_dir, gen_args.prediction)

        with open(gold_path) as f_gold, open(pred_binary_path) as f_pred:
            gold_data = f_gold.readlines()
            pred_data = f_pred.readlines()

        if gen_args.discriminative_type in ["edu", "autoedu"]:
            met_gold = ParsingMetrics(levels=['edu'])
        else:
            met_gold = ParsingMetrics()

        for i in tqdm(range(len(pred_data))):
            pred, gold = pred_data[i], gold_data[i]
            gold = json.loads(gold)
            pred = json.loads(pred)
            cand_texts = []
            cand_idxs = []
            for idx, cand in enumerate(pred["edu_strings_binary"]):
                cand_texts.append(cand)
                cand_idxs.append(idx)

            cand_texts = [torch.LongTensor(cand) for cand in cand_texts]
            
            if max([len(cand) for cand in cand_texts]) <= 512:
                #build batch
                batch = _build_batchs(cand_texts, task)
                generator = task.build_generator(gen_args)
                if torch.cuda.is_available() and not args.cpu:
                    batch = utils.move_to_cuda(batch)

                #predict
                evel_input_mode = args.evel_input_mode if 'evel_input_mode' in args else args.input_mode
                translations = task.inference_step(generator, [trainer.model], batch, evel_input_mode=evel_input_mode)
                hypos = translations
                scores = F.softmax(torch.FloatTensor([hypo[0]['positional_scores'].mean() for hypo in hypos]), dim=0)
                scores = scores.tolist() 
                
            else:
                scores = [1] + [0] * (len(cand_texts) - 1)

            indices = scores.index(max(scores))
            

            if gen_args.discriminative_type in ["edu", "autoedu"]:
                gold_edus = gold["edu_positions"][0]
                pred_edus = pred["edu_positions"][indices]
                if len(pred_edus) > 0 and pred_edus[0] == 0:
                    pred_edus.remove(0)
                met_gold.eval_edu(gold_edus, pred_edus)

            else:
                pred_bracket = pred["brackets"][indices]
                gold_bracket = gold["brackets"][0]
                met_gold.eval_brackets(gold_bracket, pred_bracket)


        evaluation_results = met_gold.report() 
        print(evaluation_results)
        return evaluation_results['all']

    valid_f1 = evaluate_score(trainer, args.valid_cand_path)
    valid_losses.append(-valid_f1)

    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None, preds=None, targets=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    
    if getattr(args, "regression_target", None) is not None and preds is not None:
        stats['pearson'] = pearsonr(preds, targets)[0]
        stats['spearman'] = spearmanr(preds, targets)[0]

    if getattr(args, "num_classes", None) == 2 and preds is not None:
        stats['mcc'] = matthews_corrcoef(targets, preds)
        stats['f1'] = f1_score(y_true=targets, y_pred=preds)

    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--do-evaluate', action='store_true', default=False,
                        help='Only do evaluation (for squad)')
    parser.add_argument('--do-layer-decay', action='store_true', default=False,
                        help='Do layer-wise learning rate decay ')
    parser.add_argument('--layer-decay', default=1.0, type=float,
                        help='The coefficient of layer decay')
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
