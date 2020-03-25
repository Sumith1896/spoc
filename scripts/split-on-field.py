#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split the TSV file into train/eval/test such that:
- The amount of data follows the specified ratio
- Each group of examples with the same specified field goes together.
- If `test_field_values_file` is specified, read field values from the file,
    and put examples with those field values in the test set.
"""

import sys, os, shutil, re, argparse, json, random
from collections import defaultdict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-c', '--sort-by-count', action='store_true',
            help='Instead of shuffling, sort the groups by example counts.'
                 ' Groups with few examples will go to the test set')
    parser.add_argument('-t', '--train-ratio', type=float, default=.9,
            help='Fraction of training data within train + dev data'
                 ' (i.e., test data already excluded)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--test-ratio', type=float, default=.1,
            help='Fraction of test data within all data')
    group.add_argument('-f', '--test-field-values-file',
            help='Read field values from this file,'
                 ' and put examples with those field values in the test set.')
    parser.add_argument('infile', help='TSV file')
    parser.add_argument('field', help='Field for grouping examples')
    args = parser.parse_args()

    # field value -> (raw line string, key-value dict)
    data = defaultdict(list)
    n = 0

    with open(args.infile) as fin:
        header_line = fin.readline()
        header = header_line.rstrip('\n').split('\t')
        for i, line in enumerate(fin):
            n += 1
            kv = dict(zip(header, line.rstrip('\n').split('\t')))
            data[kv[args.field]].append((line, kv))

    print('Read {} lines in {} groups'.format(n, len(data)))

    # Split data
    keys = list(data.keys())
    if args.sort_by_count:
        keys.sort(key=lambda x: -len(data[x]))
    else:
        random.seed(args.seed)
        random.shuffle(keys)
    print('Num examples in each group:', [len(data[x]) for x in keys])

    train_data = []
    eval_data = []
    test_data = []

    if args.test_field_values_file:
        # Put test examples in test_data first
        with open(args.test_field_values_file) as fin:
            test_keys = [x.strip() for x in fin]
        for key in test_keys:
            if key not in data:
                print('WARNING: test key {} not in raw data'.format(key))
            else:
                test_data.extend(data[key])
    else:
        # Put a certain number of examples in test_data
        # Prioritize small groups
        key_iter = reversed(keys)
        test_keys = []
        while len(test_data) < n * args.test_raio:
            key = next(key_iter)
            test_data.extend(data[key])
            test_keys.append(key)

    # Divide the rest
    keys = [x for x in keys if x not in set(test_keys)]
    print('Remaining:', [len(data[x]) for x in keys])
    key_iter = iter(keys)
    while len(train_data) < (n - len(test_data)) * args.train_ratio:
        train_data.extend(data[next(key_iter)])
    for key in key_iter:
        eval_data.extend(data[key])

    print('Examples: {} train / {} eval / {} test'.format(
        len(train_data), len(eval_data), len(test_data)))
    if not train_data or not eval_data or not test_data:
        print('WARNING: some split has 0 examples!!!')

    # Print statistics
    if args.verbose:
        for other_field in header:
            print('Unique {}: {} train / {} eval / {} test'.format(
                other_field,
                len(set(x[other_field] for _, x in train_data)),
                len(set(x[other_field] for _, x in eval_data)),
                len(set(x[other_field] for _, x in test_data)),
            ))

    # Dump to files
    prefix = re.sub(r'\.tsv$', '', args.infile)
    for suffix, data in (
        ('-train.tsv', train_data),
        ('-eval.tsv', eval_data),
        ('-test.tsv', test_data),
    ):
        with open(prefix + suffix, 'w') as fout:
            fout.write(header_line)
            for line, _ in data:
                fout.write(line)


if __name__ == '__main__':
    main()

