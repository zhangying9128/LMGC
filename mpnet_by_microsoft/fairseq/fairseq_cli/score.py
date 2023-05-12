#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils, options, progress_bar, utils
from tqdm import tqdm 
import os

def main(args, override_args=None):
    utils.import_user_module(args)

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    print(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    if True:
        #zhangying
        import copy
        import json
        from scripts.rsteval import ParsingMetrics

        def _build_batchs(src_tokens, task):
            dataset = task.build_dataset_for_inference(src_tokens, [src.numel() for src in src_tokens])
            batch = dataset.collater([ dataset[i] for i in range(dataset.__len__())])
            return batch

        def evaluate_score(model):
            gen_args = copy.copy(args)
            gen_args.score_reference = True
            gold_path = os.path.join(gen_args.reference_dir, gen_args.reference)
            pred_binary_path = os.path.join(gen_args.reference_dir, gen_args.prediction)

            with open(gold_path) as f_gold, open(pred_binary_path) as f_pred:
                gold_data = f_gold.readlines()
                pred_data = f_pred.readlines()

            if gen_args.discriminative_type in ["edu", "autoedu"]:
                met_gold = ParsingMetrics(levels=['edu'])
            else:
                met_gold = ParsingMetrics()

            all_scores = []
            for i in tqdm(range(len(pred_data))):
                pred, gold = pred_data[i], gold_data[i]
                gold = json.loads(gold)
                pred = json.loads(pred)
                cand_texts = []
                cand_idxs = []
                for idx, cand in enumerate(pred["text_binary"]):
                    cand_texts.append(cand)
                    cand_idxs.append(idx)
                
                cand_texts = [torch.LongTensor(cand) for cand in cand_texts]

                #build batch
                batch = _build_batchs(cand_texts, task)
                generator = task.build_generator(gen_args)
                if torch.cuda.is_available() and not args.cpu:
                    batch = utils.move_to_cuda(batch)

                #predict
                evel_input_mode = model_args.evel_input_mode if 'evel_input_mode' in model_args else model_args.input_mode
                translations = task.inference_step(generator, [model], batch, evel_input_mode=evel_input_mode)
                hypos = translations
                scores = F.softmax(torch.FloatTensor([hypo[0]['positional_scores'].mean() for hypo in hypos]), dim=0)
                scores = scores.tolist() 

                all_scores.append(scores)
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
            return evaluation_results, all_scores

        _, all_scores = evaluate_score(model)
        with open(args.score_file, 'w') as f:
            for scores in all_scores:
                f.write(json.dumps(scores) + "\n")

def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    main(args, override_args)


if __name__ == '__main__':
    cli_main()
