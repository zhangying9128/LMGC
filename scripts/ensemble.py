# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:50:47 2021

@author: Mac
"""
import os
import json
import numpy as np
from rsteval import ParsingMetrics

import argparse


def main(args):
    #load data
    reference_file = os.path.join(args.data_dir, args.reference)
    prediction_file = os.path.join(args.data_dir, args.prediction)
    with open(prediction_file) as f_pred, open(reference_file) as f_gold:
        predictions = [json.loads(preds) for preds in f_pred.readlines()]
        references = [json.loads(refs) for refs in f_gold.readlines()]

    #load model scores
    models = [args.model_name + str(seed) for seed in [1, 2, 3, 4, 5]] 
    models_scores = []
    for model in models:
        with open(os.path.join(model, args.score_file), "r") as f:
            all_scores = [json.loads(scores) for scores in f.readlines()]
        models_scores.append(all_scores)


    #eval model separetely
    if args.task == "segmentation":
        cum_scores = {"edu_precision":[], "edu_recall":[], "edu_f1":[]}
        task = "edu_positions"
    else:
        cum_scores = {"span":[], "ns":[],  "rela":[]}
        task = "brackets"

    for i in range(len(models)):
        if args.task == "segmentation":
            met_gold = ParsingMetrics(levels=['edu'])
        else:
            met_gold = ParsingMetrics()

        pred_batch_segs = []
        for j, (pred, ref) in enumerate(zip(predictions, references)):
            max_index = models_scores[i][j].index(max(models_scores[i][j]))
            pred = pred[task][max_index]
            if task == "edu_positions":
                if len(pred) > 0 and pred[0] == 0:
                    pred.remove(0)
                met_gold.eval_edu(ref[task][0], pred)
            else:
                met_gold.eval_brackets(ref[task][0], pred)


        evaluation_results = met_gold.report() 
        print(evaluation_results)
        for key in cum_scores.keys():
            cum_scores[key].append(evaluation_results[key])

    average_scores = {key: sum(values) / len(values) for key, values in cum_scores.items()}

    print('-----------------------average scores-----------------------------')
    print(average_scores)

    #ensemble scores
    if args.task == "segmentation":
        met_gold = ParsingMetrics(levels=['edu'])
    else:
        met_gold = ParsingMetrics()

    pred_batch_segs = []
    pred_batch = []

    for j in range(len(models_scores[0])):
        ref = references[j]
        scores = 0
        for i in range(len(models_scores)):
            scores += np.array(models_scores[i][j])
        _scores = [(s, len(scores) - i) for i, s in enumerate(list(scores))]
        _scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        max_indexs = [len(_scores) - i[1] for i in _scores]
        max_index = max_indexs[0]

        if task == "edu_positions":
            pred = predictions[j][task][max_index]
            if len(pred) > 0 and pred[0] == 0:
                pred.remove(0)
            met_gold.eval_edu(ref[task][0], pred)
        else:
            pred = predictions[j][task][max_index]
            met_gold.eval_brackets(ref[task][0], pred)


    evaluation_results = met_gold.report() 
    print('-----------------------ensemble scores-----------------------------')
    print(evaluation_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="segmentation", choices=['segmentation', 'parsing'])    
    parser.add_argument('--data-dir', type=str, default='data/segmentation/edu/test/', help='path to load reference and prediction files')
    parser.add_argument('--reference', type=str, default='reference_raw.txt', help='reference file')
    parser.add_argument('--prediction', type=str, default='output_raw.txt', help='prediction file')
    parser.add_argument('--model-name', type=str, default="discriminative_20_solvedata_lr0.00009_")
    parser.add_argument('--score-file', type=str, default=None,
                       help='path to loads scores for each pair of data')
    args = parser.parse_args()
    main(args)