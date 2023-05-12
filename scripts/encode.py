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
        type=str, default="edu", choices=['edu', 'span', 'ns', 'rela', 'full', 'autoedu', 'autospan', 'autons', 'autorela'],
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
        if args.binary_method in ['edu', 'autoedu']:
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        elif args.binary_method in ['span', 'autospan']:
            encoded_lines = pool.imap(encoder.encode_span_lines, zip(*inputs), 100)
        elif args.binary_method in ['ns', 'autons']:
            encoded_lines = pool.imap(encoder.encode_ns_lines, zip(*inputs), 100)
        elif args.binary_method in ['rela', 'autorela']:
            encoded_lines = pool.imap(encoder.encode_rela_lines, zip(*inputs), 100)
        elif args.binary_method == 'full':
            encoded_lines = pool.imap(encoder.encode_full_lines, zip(*inputs), 100)
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
            line = json.loads(line)["text"]
            line = line[0]
            line = [part + " [SEP]" for part in line ]
            line = " ".join(line)
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
            line = json.loads(line)["text"]
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
            line = json.loads(line)["text"]
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
            line = json.loads(line)["text"]
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
            line = json.loads(line)["text"]
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
