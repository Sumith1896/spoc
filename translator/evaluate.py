#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse the log file from translate.py and output evaluation scores.
"""
from __future__ import print_function

import re
import argparse
from collections import Counter


SKIPTHISLINE = "SKIPTHISLINE"
DUMMY = "DUMMY"
GOLD_NOT_FOUND = 999999


class Output(object):
    IGNORE_BRACES = False

    def __init__(self):
        self.index = None
        self.sentence = None
        self.preds = []
        self.pred_scores = []
        self.gold = None
        self.gold_score = None
        self.gold_unked = None

    @property
    def rank(self):
        """
        Compute the rank (1-indexed) of the gold among the predictions.
        If the gold is not found, return GOLD_NOT_FOUND.
        """
        for i, pred in enumerate(self.preds):
            if pred == self.gold:
                return i + 1
        return GOLD_NOT_FOUND

    @classmethod
    def dump_header(self, fout):
        stuff = [ 
            "index",
            "text",
            "gold_score",
            "pred_score",
            "gold",
            "pred",
        ]
        print("\t".join(str(x) for x in stuff), file=fout)


    def dump(self, fout, args):
        """Print a TSV line summarizing the example."""
        stuff = [ 
            self.index,
            self.sentence,
            self.gold_score,
            self.pred_scores[0],
            self.gold,
            self.preds[0],
        ]
        if args.gold_rank:
            stuff.append(self.rank)
        if args.dump_all_preds:
            stuff += self.preds[1:]
            stuff += self.pred_scores
        print("\t".join(str(x) for x in stuff), file=fout)
        if args.dump_all_preds_verbose:
            for i, pred in enumerate(self.preds):
                print("\t".join(["#", str(i), pred]), file=fout)
            print(file=fout)

    @classmethod
    def format_code(cls, code):
        # Resplit and rejoin
        code = ' '.join(code.split())
        # Remove braces if specified
        if cls.IGNORE_BRACES:
            code = re.sub('^}|{$', '', code).strip()
        return code

    @classmethod
    def parse_file(cls, pred_file, tgt_file):
        """"Yield Output objects based on the given files."""
        output = None
        with open(pred_file) as fin, open(tgt_file) as ftgt:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = line.rstrip('\n')
                # SENT 1: [...] (begins a new output)
                m = re.match(r'^SENT (\d+): (.*)$', line)
                if m:
                    if output is not None:
                        yield output
                    output = Output()
                    i, sentence = m.groups()
                    output.index = int(i)
                    output.sentence = " ".join(eval(sentence))
                    continue
                # GOLD 1: ...
                m = re.match(r'^GOLD (\d+): (.*)$', line)
                if m:
                    i, gold = m.groups()
                    assert int(i) == output.index
                    output.gold_unk = gold
                    output.gold = cls.format_code(ftgt.readline().strip())
                    continue
                # GOLD SCORE: ...
                m = re.match(r'^GOLD SCORE: (.*)$', line)
                if m:
                    output.gold_score = float(m.groups()[0])
                    continue
                # BEST HYP:
                if line == "BEST HYP:":
                    while True:
                        pred_line = fin.readline().strip()
                        if not pred_line:
                            break
                        m = re.match(r'\[([^]]+)\] (.*)$', pred_line)
                        score, pred = m.groups()
                        output.preds.append(cls.format_code(" ".join(eval(pred))))
                        output.pred_scores.append(float(score))
        # The last output
        if output is not None:
            yield output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_file', default='/dev/null',
            help='dump prediction TSV to this file')
    parser.add_argument('-b', '--ignore-braces', action='store_true',
            help='ignore the initial and ending braces in code')
    parser.add_argument('-a', '--dump-all-preds', action='store_true',
            help='When dumping predictions, dump all ranks')
    parser.add_argument('-A', '--dump-all-preds-verbose', action='store_true',
            help='When dumping predictions, dump all ranks in their own lines')
    parser.add_argument('-r', '--gold-rank', action='store_true',
            help='Print gold rank')
    parser.add_argument("pred_file",
            help="the prediction file from translate.py")
    parser.add_argument("tgt_file",
            help="gold target file")
    args = parser.parse_args()

    if args.ignore_braces:
        Output.IGNORE_BRACES = True

    ranks = Counter()

    with open(args.out_file, 'w') as fout:
        Output.dump_header(fout)
        for output in Output.parse_file(args.pred_file, args.tgt_file):
            if args.out_file != '/dev/null':
                output.dump(fout, args)
            if output.sentence != SKIPTHISLINE and output.sentence != DUMMY:
                ranks[output.rank] += 1

    n = sum(ranks.values())
    print("Number of examples: {}".format(n))
    accum = 0
    mrr = 0.
    for i, count in sorted(ranks.items()):
        accum += count
        if i != GOLD_NOT_FOUND:
            mrr += count * 1. / (i + 1)
        print("RANK {:>2} = {:5} = {:6.2f} % | accum: {:5} = {:6.2f} %".format(
            i if i != GOLD_NOT_FOUND else "no",
            count, count * 100. / n, accum, accum * 100. / n))
    print("MRR: {:.6f}".format(mrr / n))

if __name__ == '__main__':
    main()

