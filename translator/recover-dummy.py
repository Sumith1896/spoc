#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add DUMMY lines to the summary file
"""

import sys, os, shutil, re, argparse, json, random
from collections import defaultdict, Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('orig_test_tsv')
    parser.add_argument('summary_nodummy_tsv')
    args = parser.parse_args()

    index = 1
    with open(args.orig_test_tsv) as orig_f, open(args.summary_nodummy_tsv) as summ_f:
        orig_head = orig_f.readline().rstrip('\n')
        summ_head = summ_f.readline().rstrip('\n')
        print(summ_head)
        for orig_line in orig_f:
            orig_line = orig_line.rstrip('\n').split('\t')
            if not orig_line[0]:
                # Print a dummy line
                print('\t'.join([
                    str(index),
                    'DUMMY',
                    '0.0',
                    '0.0',
                    orig_line[1],
                    'DUMMY',
                ]))
            else:
                # Replace index
                summ_line = summ_f.readline().rstrip('\n').split('\t', 1)
                print('{}\t{}'.format(index, summ_line[1]))
            index += 1
        # Sanity check: the summary file should have been fully consumed
        blank = summ_f.readline()
        assert not blank


if __name__ == '__main__':
    main()

