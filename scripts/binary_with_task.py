# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:48:29 2020

@author: zhang
"""
import os
import json
import torch
from collections import OrderedDict
from fairseq.models.masked_permutation_net import MPNet
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import time

def replace_token(sentence, map, bpe):
    temp_text = []
    new_sentence = []
    for token in sentence.split():
        if token.lower() in map:
            if len(temp_text) > 0:
                bpe_text = bpe.encode(' '.join(temp_text)) #encode method contains lowercase
                temp_text = []
                new_sentence.append(bpe_text)
            new_sentence.append(map[token.lower()])
        else:
            temp_text.append(token)
    if len(temp_text) > 0:
        bpe_text = bpe.encode(' '.join(temp_text)) #encode method contains lowercase
        temp_text = []
        new_sentence.append(bpe_text)
    bpe_sentence = " ".join(new_sentence)
    return bpe_sentence

class MultiprocessingEdit(object):

    def __init__(self):
        pass

    def initializer(self, bpe, source_dictionary, task):
        self.bpe = bpe
        self.task = task
        self.source_dictionary = source_dictionary

    def BPE_tokenize(self, sentence):
        if self.task in ["edu", "autoedu"]:
            sentence = sentence + " [SEP]"
            bpe_sentence = self.bpe.encode(sentence) 

        elif self.task in ["span", "autospan"]:
            sentence = sentence.split("(ROOT_left")[1]
            sentence = sentence.split("ROOT_right)")[0]
            span_map = {"(satellite_left": "[unused0]",
                        "satellite_right)": "[unused1]",
                        "(nucleus_left": "[unused0]",
                        "nucleus_right)": "[unused1]"}

            ns = ['nucleus', 'satellite']
            relations = ['Attribution', 'Background', 'Cause', 'Comparison', 'Condition', 'Contrast', 'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',  'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization', 'span', 'Same-Unit']
            directions = ['left', 'right']
            _span_map = {}
            for n in ns:
                for relation in relations:
                    for direction in directions:
                        if direction == 'left':
                            key_span =  '(' + n + '_' + direction
                            key = '(' + n + ':' + relation + '_' + direction
                        else:
                            key_span =  n + '_' + direction + ')'
                            key = n + ':' + relation + '_' + direction + ')'
                        _span_map[key.lower()] = span_map[key_span]

            span_map = _span_map
            bpe_sentence = replace_token(sentence, span_map, self.bpe)

        elif self.task in ["ns", "autons"]:
            sentence = sentence.split("(ROOT_left")[1]
            sentence = sentence.split("ROOT_right)")[0]

            ns_map = {"(satellite_left": "[unused0]",
                      "satellite_right)": "[unused1]",
                      "(nucleus_left": "[unused2]",
                      "nucleus_right)": "[unused3]"}

            ns = ['nucleus', 'satellite']
            relations = ['Attribution', 'Background', 'Cause', 'Comparison', 'Condition', 'Contrast', 'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',  'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization', 'span', 'Same-Unit']
            directions = ['left', 'right']
            _ns_map = {}
            for n in ns:
                for relation in relations:
                    for direction in directions:
                        if direction == 'left':
                            key_ns =  '(' + n + '_' + direction
                            key = '(' + n + ':' + relation + '_' + direction
                        else:
                            key_ns =  n + '_' + direction + ')'
                            key = n + ':' + relation + '_' + direction + ')'
                        _ns_map[key.lower()] = ns_map[key_ns]
            ns_map = _ns_map

            bpe_sentence = replace_token(sentence, ns_map, self.bpe)

        elif self.task in ["rela", "autorela"]:
            sentence = sentence.split("(ROOT_left")[1]
            sentence = sentence.split("ROOT_right)")[0]

            ns = ['nucleus', 'satellite']
            relations = ['Attribution', 'Background', 'Cause', 'Comparison', 'Condition', 'Contrast', 'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',  'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization', 'span', 'Same-Unit']
            directions = ['left', 'right']
            rela_map = {}
            for relation in relations:
                for direction in directions:
                    if direction == 'left':
                        key = '(' + relation + '_' + direction
                    else:
                        key =  relation + '_' + direction + ')'
                    rela_map[key.lower()] = "[unused" + str(len(rela_map)) + "]"


            _rela_map = {}
            for n in ns:
                for relation in relations:
                    for direction in directions:
                        if direction == 'left':
                            key_rela = '(' + relation + '_' + direction
                            key = '(' + n + ':' + relation + '_' + direction
                        else:
                            key_rela = relation + '_' + direction + ')'
                            key = n + ':' + relation + '_' + direction + ')'
                        _rela_map[key.lower()] = rela_map[key_rela.lower()]
            rela_map = _rela_map

            bpe_sentence = replace_token(sentence, rela_map, self.bpe)


        elif self.task == "full":
            sentence = sentence.split("(ROOT_left")[1]
            sentence = sentence.split("ROOT_right)")[0]

            ns = ['nucleus', 'satellite']
            relations = ['Attribution', 'Background', 'Cause', 'Comparison', 'Condition', 'Contrast', 'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',  'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization', 'span', 'Same-Unit']
            directions = ['left', 'right']
            full_map = {}
            for n in ns:
               for relation in relations:
                    for direction in directions:
                        if direction == 'left':
                            key = '(' + n + ':' + relation + '_' + direction
                        else:
                            key = n + ':' + relation + '_' + direction + ')'
                        full_map[key.lower()] = "[unused" + str(len(full_map)) + "]"

            bpe_sentence = replace_token(sentence, full_map, self.bpe)

        return bpe_sentence

    def encode_lines(self, line):
        line = line[0]
        texts_binary = []
        for text in line["text"]:
            tokenized_text = [self.BPE_tokenize(subtext) for subtext in text]
            tokenized_text = " ".join(tokenized_text) + ' </s>'
            texts_binary.append(self.source_dictionary.encode_line(tokenized_text, append_eos=False).tolist())
        line["text_binary"] = texts_binary
        return ["PASS", [line]]

def preprocess_text(model, args):
    with open(args.text_file, "r") as f:
        inputs = [json.loads(line) for line in f.readlines()]

    encoder = MultiprocessingEdit()
    pool = Pool(60, initializer=encoder.initializer(model.bpe, model.task.source_dictionary, args.task))

    time1=time.time()
    outputs = []
    encoded_lines = pool.imap(encoder.encode_lines, zip(inputs))
    for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
        if filt == "PASS":
            outputs += enc_lines
        if i % 10000 == 0:
            print("processed {} lines".format(i))
    time2=time.time()
    print('Time cost:' + str(time2 - time1) + 's')


    binary_file = args.text_file[:-7] + args.task + "_binary.txt"
    with open(binary_file, "w") as f:
        for line in outputs:
            f.write(json.dumps(line) + "\n")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-file', type=str, default="output_raw.txt")
    parser.add_argument('--pretrained-mpnet-path', type=str, default="mpnet.base")
    parser.add_argument('--task', type=str, default="edu", choices=['edu', 'span', 'ns', 'rela', "full", "autoedu", "autospan", "autons", "autorela"])

    args = parser.parse_args()
    model = MPNet.from_pretrained(args.pretrained_mpnet_path, 'mpnet.pt', bpe='bert')
    preprocess_text(model, args)


