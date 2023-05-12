import argparse
import contextlib
import sys
import os
import json

from collections import Counter
from multiprocessing import Pool

from transformers import BertTokenizer


def replace_token(sentence, map, bpe):
    temp_text = []
    new_sentence = []
    for token in sentence.split():
        if token.lower() in map:
            if len(temp_text) > 0:
                bpe_text = bpe._tokenize(' '.join(temp_text)) #encode method contains lowercase
                temp_text = []
                new_sentence.append(" ".join(bpe_text))
            new_sentence.append(map[token.lower()])
        else:
            temp_text.append(token)
    if len(temp_text) > 0:
        bpe_text = bpe._tokenize(' '.join(temp_text)) #encode method contains lowercase
        temp_text = []
        new_sentence.append(" ".join(bpe_text))
    bpe_sentence = " ".join(new_sentence)
    return bpe_sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--binary-method",
        type=str, default="edu", choices=['edu', 'span', 'ns', 'rela', 'full', 'segspan', 'segns', 'segrela', 'nli', 'gec', 'gec_concat','conspa', 'conspacomb', 'ner', 'nerspan'],
    )
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        if args.binary_method == 'edu':
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        elif args.binary_method in ['span', 'segspan']:
            encoded_lines = pool.imap(encoder.encode_span_lines, zip(*inputs), 100)
        elif args.binary_method in ['ns', 'segns']:
            encoded_lines = pool.imap(encoder.encode_ns_lines, zip(*inputs), 100)
        elif args.binary_method in ['rela', 'segrela']:
            encoded_lines = pool.imap(encoder.encode_rela_lines, zip(*inputs), 100)
        elif args.binary_method == 'full':
            encoded_lines = pool.imap(encoder.encode_full_lines, zip(*inputs), 100)
        elif args.binary_method == 'nli':
            encoded_lines = pool.imap(encoder.encode_nli_lines, zip(*inputs), 100)
        elif args.binary_method == 'gec':
            encoded_lines = pool.imap(encoder.encode_gec_lines, zip(*inputs), 100)
        elif args.binary_method == 'gec_concat':
            encoded_lines = pool.imap(encoder.encode_gec_concat_lines, zip(*inputs), 100)
        elif args.binary_method == 'conspa':
            encoded_lines = pool.imap(encoder.encode_conspa_lines, zip(*inputs), 100)
        elif args.binary_method == 'conspa_comb':
            encoded_lines = pool.imap(encoder.encode_conspa_comb_lines, zip(*inputs), 100)
        elif args.binary_method == 'ner':
            encoded_lines = pool.imap(encoder.encode_ner_lines, zip(*inputs), 100)
        elif args.binary_method == 'nerspan':
            encoded_lines = pool.imap(encoder.encode_nerspan_lines, zip(*inputs), 100)
        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode(self, line):
        global bpe
        subword = bpe._tokenize(line)
        return subword

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0]
            line = [part + " [SEP]" for part in line ]
            line = " ".join(line)
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def encode_ner_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            '''
            gec_map = {"(loc_left": "[unused0]",
                      "loc_right)": "[unused1]",
                      "(misc_left": "[unused2]",
                      "misc_right)": "[unused3]",
                      "(per_left": "[unused4]",
                      "per_right)": "[unused5]",
                      "(org_left": "[unused6]",
                      "org_right)": "[unused7]"}
            '''
            gec_map = {"(loc_left": "[unused0]",
                      "loc_right)": "[unused1]",
                      "(fac_left": "[unused2]",
                      "fac_right)": "[unused3]",
                      "(per_left": "[unused4]",
                      "per_right)": "[unused5]",
                      "(org_left": "[unused6]",
                      "org_right)": "[unused7]",
                      "(wea_left": "[unused8]",
                      "wea_right)": "[unused9]",
                      "(veh_left": "[unused10]",
                      "veh_right)": "[unused11]",
                      "(gpe_left": "[unused12]",
                      "gpe_right)": "[unused13]"}


            line = replace_token(line, gec_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_nerspan_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]

            gec_map = {"(span_left": "[unused0]",
                      "span_right)": "[unused1]"}

            line = replace_token(line, gec_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_conspa_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]

            label_vocabs =[(), ('ADJP',), ('ADJP', 'ADJP'), ('ADJP', 'ADVP'), ('ADJP', 'NP'), ('ADJP', 'QP'), ('ADVP',), ('ADVP', 'ADJP'), ('ADVP', 'ADVP'), ('ADVP', 'PRT'), ('CONJP',), ('FRAG',), ('FRAG', 'ADJP'), ('FRAG', 'ADVP'), ('FRAG', 'INTJ'), ('FRAG', 'NP'), ('FRAG', 'PP'), ('FRAG', 'S'), ('FRAG', 'S', 'ADJP'), ('FRAG', 'S', 'VP'), ('FRAG', 'SBAR'), ('FRAG', 'SBARQ'), ('FRAG', 'UCP'), ('FRAG', 'VP'), ('FRAG', 'WHADVP'), ('FRAG', 'WHNP'), ('FRAG', 'WHPP'), ('INTJ',), ('INTJ', 'S'), ('LST',), ('NAC',), ('NP',), ('NP', 'ADJP'), ('NP', 'ADVP'), ('NP', 'FRAG'), ('NP', 'INTJ'), ('NP', 'NP'), ('NP', 'NP', 'NP'), ('NP', 'NP', 'QP'), ('NP', 'PP'), ('NP', 'PRN'), ('NP', 'QP'), ('NP', 'S'), ('NP', 'S', 'VP'), ('NP', 'SBAR'), ('NP', 'SBAR', 'S', 'VP'), ('NX',), ('NX', 'NX'), ('NX', 'QP'), ('NX', 'S'), ('NX', 'S', 'VP'), ('PP',), ('PP', 'NP'), ('PP', 'PP'), ('PRN',), ('PRN', 'FRAG', 'WHADJP'), ('PRN', 'NP'), ('PRN', 'PP'), ('PRN', 'S'), ('PRN', 'S', 'VP'), ('PRN', 'SBAR'), ('PRN', 'SINV'), ('PRT',), ('QP',), ('RRC',), ('RRC', 'VP'), ('S',), ('S', 'ADJP'), ('S', 'ADVP'), ('S', 'NP'), ('S', 'PP'), ('S', 'S'), ('S', 'UCP'), ('S', 'VP'), ('S', 'VP', 'ADVP'), ('S', 'VP', 'VP'), ('SBAR',), ('SBAR', 'FRAG'), ('SBAR', 'S'), ('SBAR', 'S', 'VP'), ('SBAR', 'SBAR', 'S'), ('SBAR', 'SBARQ'), ('SBAR', 'SINV'), ('SBAR', 'WHADVP'), ('SBAR', 'WHNP'), ('SBARQ',), ('SBARQ', 'WHADVP'), ('SINV',), ('SQ',), ('SQ', 'VP'), ('UCP',), ('UCP', 'ADJP'), ('UCP', 'PP'), ('VP',), ('VP', 'ADVP'), ('VP', 'FRAG', 'ADJP'), ('VP', 'NP'), ('VP', 'PP'), ('VP', 'S', 'VP'), ('VP', 'SBAR'), ('VP', 'VP'), ('WHADJP',), ('WHADVP',), ('WHNP',), ('WHNP', 'QP'), ('WHNP', 'WHNP'), ('WHPP',), ('X',), ('X', 'ADVP'), ('X', 'NP'), ('X', 'PP'), ('X', 'SBARQ'), ('X', 'VP')] 
            tag_vocabs = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``'] 
            label_set= set()
            for labels in label_vocabs:
                if len(labels) > 0:
                    for label in labels:
                        label_set.add(label)

            for label in tag_vocabs:
                label_set.add(label)

            directions = ['left', 'right']

            full_map = {}
            for label in label_set:
                for direction in directions:
                    if direction == 'left':
                        key = '(' + label + '_' + direction
                    else:
                        key = label+ '_' + direction + ')'
                    full_map[key.lower()] = "[unused" + str(len(full_map)) + "]"

            line = replace_token(line, full_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_conspa_comb_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]

            label_vocabs =[(), ('ADJP',), ('ADJP', 'ADJP'), ('ADJP', 'ADVP'), ('ADJP', 'NP'), ('ADJP', 'QP'), ('ADVP',), ('ADVP', 'ADJP'), ('ADVP', 'ADVP'), ('ADVP', 'PRT'), ('CONJP',), ('FRAG',), ('FRAG', 'ADJP'), ('FRAG', 'ADVP'), ('FRAG', 'INTJ'), ('FRAG', 'NP'), ('FRAG', 'PP'), ('FRAG', 'S'), ('FRAG', 'S', 'ADJP'), ('FRAG', 'S', 'VP'), ('FRAG', 'SBAR'), ('FRAG', 'SBARQ'), ('FRAG', 'UCP'), ('FRAG', 'VP'), ('FRAG', 'WHADVP'), ('FRAG', 'WHNP'), ('FRAG', 'WHPP'), ('INTJ',), ('INTJ', 'S'), ('LST',), ('NAC',), ('NP',), ('NP', 'ADJP'), ('NP', 'ADVP'), ('NP', 'FRAG'), ('NP', 'INTJ'), ('NP', 'NP'), ('NP', 'NP', 'NP'), ('NP', 'NP', 'QP'), ('NP', 'PP'), ('NP', 'PRN'), ('NP', 'QP'), ('NP', 'S'), ('NP', 'S', 'VP'), ('NP', 'SBAR'), ('NP', 'SBAR', 'S', 'VP'), ('NX',), ('NX', 'NX'), ('NX', 'QP'), ('NX', 'S'), ('NX', 'S', 'VP'), ('PP',), ('PP', 'NP'), ('PP', 'PP'), ('PRN',), ('PRN', 'FRAG', 'WHADJP'), ('PRN', 'NP'), ('PRN', 'PP'), ('PRN', 'S'), ('PRN', 'S', 'VP'), ('PRN', 'SBAR'), ('PRN', 'SINV'), ('PRT',), ('QP',), ('RRC',), ('RRC', 'VP'), ('S',), ('S', 'ADJP'), ('S', 'ADVP'), ('S', 'NP'), ('S', 'PP'), ('S', 'S'), ('S', 'UCP'), ('S', 'VP'), ('S', 'VP', 'ADVP'), ('S', 'VP', 'VP'), ('SBAR',), ('SBAR', 'FRAG'), ('SBAR', 'S'), ('SBAR', 'S', 'VP'), ('SBAR', 'SBAR', 'S'), ('SBAR', 'SBARQ'), ('SBAR', 'SINV'), ('SBAR', 'WHADVP'), ('SBAR', 'WHNP'), ('SBARQ',), ('SBARQ', 'WHADVP'), ('SINV',), ('SQ',), ('SQ', 'VP'), ('UCP',), ('UCP', 'ADJP'), ('UCP', 'PP'), ('VP',), ('VP', 'ADVP'), ('VP', 'FRAG', 'ADJP'), ('VP', 'NP'), ('VP', 'PP'), ('VP', 'S', 'VP'), ('VP', 'SBAR'), ('VP', 'VP'), ('WHADJP',), ('WHADVP',), ('WHNP',), ('WHNP', 'QP'), ('WHNP', 'WHNP'), ('WHPP',), ('X',), ('X', 'ADVP'), ('X', 'NP'), ('X', 'PP'), ('X', 'SBARQ'), ('X', 'VP')] 
            tag_vocabs = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``'] 
            label_set= set()
            for labels in label_vocabs:
                if len(labels) > 0:
                    label_set.add("_".join([sublabel for sublabel in self.label]))

            for label in tag_vocabs:
                label_set.add(label)

            directions = ['left', 'right']

            full_map = {}
            for label in label_set:
                for direction in directions:
                    if direction == 'left':
                        key = '(' + label + '_' + direction
                    else:
                        key = label+ '_' + direction + ')'
                    full_map[key.lower()] = "[unused" + str(len(full_map)) + "]"

            line = replace_token(line, full_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_gec_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            gec_map = {"(insert_left": "[unused0]",
                      "insert_right)": "[unused1]",
                      "(delete_left": "[unused2]",
                      "delete_right)": "[unused3]",
                      "(orig_left": "[unused4]",
                      "orig_right)": "[unused5]",
                      "(substitute_left": "",
                      "substitute_right)": "[unused7]"}

            line = replace_token(line, gec_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_gec_concat_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["gec_outputs"]
            line = line[0]
            line = " [SEP] ".join(line)
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def encode_span_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            line = line.split("(ROOT_left")[1]
            line = line.split("ROOT_right)")[0]
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
            '''
            temp_text = []
            new_sentence = []
            for token in line.split():
                if token.lower() in span_map:
                    if len(temp_text) > 0:
                        bpe_text = self.encode(' '.join(temp_text))
                        temp_text = []
                        new_sentence.append(" ".join(bpe_text))
                    new_sentence.append(span_map[token.lower()])
                else:
                    temp_text.append(token)
            '''
            line = replace_token(line, span_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_ns_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            line = line.split("(ROOT_left")[1]
            line = line.split("ROOT_right)")[0]
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
            '''
            temp_text = []
            new_sentence = []
            for token in line.split():
                if token.lower() in ns_map:
                    if len(temp_text) > 0:
                        bpe_text = self.encode(' '.join(temp_text))
                        temp_text = []
                        new_sentence.append(" ".join(bpe_text))
                    new_sentence.append(ns_map[token.lower()])
                else:
                    temp_text.append(token)
            '''
            line = replace_token(line, ns_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]

    def encode_rela_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            line = line.split("(ROOT_left")[1]
            line = line.split("ROOT_right)")[0]

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
            '''
            temp_text = []
            new_sentence = []
            for token in line.split():
                if token.lower() in rela_map:
                    if len(temp_text) > 0:
                        bpe_text = self.encode(' '.join(temp_text))
                        temp_text = []
                        new_sentence.append(" ".join(bpe_text))
                    new_sentence.append(rela_map[token.lower()])
                else:
                    temp_text.append(token)
            '''
            line = replace_token(line, rela_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            if len(line) < 3:
                print(line)
            enc_lines.append(line)
            
        return ["PASS", enc_lines]


    def encode_full_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = json.loads(line)["edu_strings"]
            line = line[0][0]
            line = line.split("(ROOT_left")[1]
            line = line.split("ROOT_right)")[0]

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
            '''
            temp_text = []
            new_sentence = []
            for token in line.split():
                if token.lower() in full_map:
                    if len(temp_text) > 0:
                        bpe_text = self.encode(' '.join(temp_text))
                        temp_text = []
                        new_sentence.append(" ".join(bpe_text))
                    new_sentence.append(full_map[token.lower()])
                else:
                    temp_text.append(token)
            '''
            line = replace_token(line, full_map, bpe)
            assert line is not None
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            enc_lines.append(line)
        return ["PASS", enc_lines]



    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
