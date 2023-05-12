#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午8:34

import numpy
import os
import json
import argparse

class Performance(object):
    def __init__(self, percision, recall, hit_num):
        self.percision = percision
        self.recall = recall
        self.hit_num = hit_num


class ParsingMetrics(object):
    def __init__(self, levels=['span', 'nuclearity', 'relation']):
        """ Initialization

        :type levels: list of string
        :param levels: eval levels, the possible values are only
                       'span','nuclearity','relation'
        """
        self.levels = levels
        self.span_perf = Performance([], [], 0)
        self.nuc_perf = Performance([], [], 0)
        self.rela_perf = Performance([], [], 0)
        self.gold_span_num = 0
        self.pred_span_num = 0
        self.hit_num_each_relation = {}
        self.pred_num_each_relation = {}
        self.gold_num_each_relation = {}
        #zhangying
        self.relation_confusion_matrix = {}
        self.edu_perf = Performance([], [], 0)

    def eval(self, goldtree, predtree):
        """ Evaluation performance on one pair of RST trees

        :type goldtree: RSTTree class
        :param goldtree: gold RST tree

        :type predtree: RSTTree class
        :param predtree: RST tree from the parsing algorithm
        """
        goldbrackets = goldtree.bracketing()
        predbrackets = predtree.bracketing()
        self.gold_span_num += len(goldbrackets)
        self.pred_span_num += len(predbrackets)
        for level in self.levels:
            if level == 'span':
                self._eval(goldbrackets, predbrackets, idx=1)
            elif level == 'nuclearity':
                self._eval(goldbrackets, predbrackets, idx=2)
            elif level == 'relation':
                self._eval(goldbrackets, predbrackets, idx=3)
            else:
                raise ValueError("Unrecognized eval level: {}".format(level))


    #zhangying
    def eval_brackets(self, goldbrackets, predbrackets):
        """ Evaluation performance on one pair of RST trees

        :type goldtree: RSTTree class
        :param goldtree: gold RST tree

        :type predtree: RSTTree class
        :param predtree: RST tree from the parsing algorithm
        """
        if goldbrackets is None and predbrackets is not None:
            self.pred_span_num += len(predbrackets)

        elif goldbrackets is not None and predbrackets is None:
            self.gold_span_num += len(goldbrackets)

            goldspan = [(item[0], item[2]) for item in goldbrackets]
            for span in goldspan:
                relation = span[-1]
                if relation in self.gold_num_each_relation:
                    self.gold_num_each_relation[relation] += 1
                else:
                    self.gold_num_each_relation[relation] = 1

        elif goldbrackets is not None and predbrackets is not None:
            self.gold_span_num += len(goldbrackets)
            self.pred_span_num += len(predbrackets)
            for level in self.levels:
                if level == 'span':
                    self._eval(goldbrackets, predbrackets, idx=1)
                elif level == 'nuclearity':
                    self._eval(goldbrackets, predbrackets, idx=2)
                elif level == 'relation':
                    self._eval(goldbrackets, predbrackets, idx=3)
                else:
                    raise ValueError("Unrecognized eval level: {}".format(level))

    def eval_edu(self, goldedus, prededus):
        if len(prededus) > 0 and prededus[0] == 0:
            prededus.remove(0)
        self.gold_span_num += len(goldedus)
        self.pred_span_num += len(prededus)
        self.edu_perf.hit_num += len(set(goldedus) & set(prededus))
    
    def _eval(self, goldbrackets, predbrackets, idx):
        """ Evaluation on each discourse span
        """
        if idx == 1 or idx == 2:
            goldspan = [item[:idx] for item in goldbrackets]
            predspan = [item[:idx] for item in predbrackets]
        elif idx == 3:
            goldspan = [(item[0], item[2]) for item in goldbrackets]
            predspan = [(item[0], item[2]) for item in predbrackets]
            #zhangying
            for gs, ps in zip(goldspan, predspan):
                g_relation, p_relation = gs[-1], ps[-1]
                if g_relation not in self.relation_confusion_matrix:
                    self.relation_confusion_matrix[g_relation] = {}
                if p_relation not in self.relation_confusion_matrix[g_relation]:
                    self.relation_confusion_matrix[g_relation][p_relation] = 0
                self.relation_confusion_matrix[g_relation][p_relation] += 1
        else:
            raise ValueError('Undefined idx for evaluation')
        hitspan = [span for span in goldspan if span in predspan]

        p, r = 0.0, 0.0
        for span in hitspan:
            if span in goldspan:
                p += 1.0
            if span in predspan:
                r += 1.0
        if idx == 1:
            self.span_perf.hit_num += p
        elif idx == 2:
            self.nuc_perf.hit_num += p
        elif idx == 3:
            self.rela_perf.hit_num += p
        p /= len(goldspan)
        r /= len(predspan)
        if idx == 1:
            self.span_perf.percision.append(p)
            self.span_perf.recall.append(r)
        elif idx == 2:
            self.nuc_perf.percision.append(p)
            self.nuc_perf.recall.append(r)
        elif idx == 3:
            self.rela_perf.percision.append(p)
            self.rela_perf.recall.append(r)
        if idx == 3:
            for span in hitspan:
                relation = span[-1]
                if relation in self.hit_num_each_relation:
                    self.hit_num_each_relation[relation] += 1
                else:
                    self.hit_num_each_relation[relation] = 1
            for span in goldspan:
                relation = span[-1]
                if relation in self.gold_num_each_relation:
                    self.gold_num_each_relation[relation] += 1
                else:
                    self.gold_num_each_relation[relation] = 1
            for span in predspan:
                relation = span[-1]
                if relation in self.pred_num_each_relation:
                    self.pred_num_each_relation[relation] += 1
                else:
                    self.pred_num_each_relation[relation] = 1

    def report(self):
        """ Compute the F1 score for different eval levels
            and print it out
        """
        cum_f1 = 0
        return_value = {}

        for level in self.levels:
            if 'span' == level:
                #zhangying 
                p = self.span_perf.hit_num / self.pred_span_num 
                r = self.span_perf.hit_num / self.gold_span_num
                f1 = (2 * p * r) / (p + r)
                return_value['span'] = f1
                cum_f1 += f1
                #print('Span level :global precision {0:.4f}, recall {1:.4f}, F1 {2:.4f}'.format(p, r, f1))

            elif 'nuclearity' == level:
                p = self.nuc_perf.hit_num / self.pred_span_num 
                r = self.nuc_perf.hit_num / self.gold_span_num
                f1 = (2 * p * r) / (p + r)
                return_value['ns'] = f1
                cum_f1 += f1
                #print('Nuclearity level :global precision {0:.4f}, recall {1:.4f}, F1 {2:.4f}'.format(p, r, f1))

            elif 'relation' == level:
                p = self.rela_perf.hit_num / self.pred_span_num 
                r = self.rela_perf.hit_num / self.gold_span_num
                f1 = (2 * p * r) / (p + r)
                return_value['rela'] = f1
                cum_f1 += f1
                #print('Relation level :global precision {0:.4f}, recall {1:.4f}, F1 {2:.4f}'.format(p, r, f1))

            elif 'edu' == level:
                #zhangying
                p = 1.0 * self.edu_perf.hit_num / self.pred_span_num if self.pred_span_num > 0 else 0.0
                r = 1.0 * self.edu_perf.hit_num / self.gold_span_num if self.gold_span_num > 0 else 0.0
                if p > 0 and r > 0:
                    f1 = (2.0 * p * r) / (p + r)
                else:
                    f1 = 0                
                return_value['edu_f1'] = f1
                return_value['edu_precision'] = p
                return_value['edu_recall'] = r
                cum_f1 += f1

            else:
                raise ValueError("Unrecognized eval level")

        return_value['all'] = cum_f1
        return return_value

def compute_oracle(predictions, references, task, met_gold):
    _references, _predictions = [], []
    for preds, ref in zip(predictions, references):
        ref = json.loads(ref)[task][0]
        preds = json.loads(preds)[task]

        if ref in preds:
            pred = ref
        else:
            if task == "edu_positions":
                pred = preds[0]
            else:
                if ref is None:
                    shortest_span = 100000
                    for pred in preds:
                        if len(pred) < shortest_span:
                            selected_pred_rst = pred
                            shortest_span = len(pred)

                else:
                    gold_ns_span = [item[:2] for item in ref]
                    gold_rela_span = [(item[0], item[2])for item in ref]
                    longest_span = 0
                    for pred in preds:
                        if pred is not None:
                            pred_rela_span = [(item[0], item[2]) for item in pred]
                            hit_rela_span = [span for span in gold_rela_span if span in pred_rela_span]

                            pred_ns_span = [item[:2] for item in pred]
                            hit_ns_span = [span for span in gold_ns_span if span in pred_ns_span]

                            if len(hit_rela_span) + len(hit_ns_span) >= longest_span:
                                selected_pred_rst = pred
                                longest_span = len(hit_ns_span) + len(hit_rela_span) 

                pred = selected_pred_rst

        if task == "edu_positions":
            met_gold.eval_edu(ref, pred)

        else:
            met_gold.eval_brackets(ref, pred)

    evaluation_results = met_gold.report() 
    print(evaluation_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="segmentation", choices=['segmentation', 'parsing'])    
    parser.add_argument('--data-dir', type=str, default='data/segmentation/edu/test/', help='path to load reference and prediction files')
    parser.add_argument('--reference', type=str, default='reference_raw.txt', help='reference file')
    parser.add_argument('--prediction', type=str, default='output_raw.txt', help='prediction file')
    args = parser.parse_args()


    reference_file = os.path.join(args.data_dir, args.reference)
    prediction_file = os.path.join(args.data_dir, args.prediction)
    with open(prediction_file) as f_pred, open(reference_file) as f_gold:
        predictions = f_pred.readlines()
        references = f_gold.readlines()

    if args.task == "segmentation":
        task = "edu_positions"
        met_gold = ParsingMetrics(levels=['edu'])
    else:
        task = "brackets"
        met_gold = ParsingMetrics()

    compute_oracle(predictions, references, task, met_gold)
