#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the (text, code) pairs from mturk-to-pairs.py into various formats.

The default is to convert into templates. For instance:
    set x to 7      x = 7;
will become
    set $1 to $2    x = $2 ;
"""

import sys, os, shutil, re, argparse, json


################################################
# Basic tokenization

TEXT_TOKENIZER = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def tokenize_text(text):
    return TEXT_TOKENIZER.findall(text)


################################################
# Clang interface

def setup_clang(clang_path):
    from clang.cindex import Config
    Config.set_library_path(clang_path)

def fix_char_string_tok(tokens):
    res_tokens = []
    if tokens and tokens[0] == "}":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "{":
        tokens = tokens[:-1]
    for token in tokens:
        if token[0] == "\"" and token[-1] == "\"":
            res_tokens.append("\"")
            res_tokens.append(token[1:-1])
            res_tokens.append("\"")
        elif token[0] == "\'" and token[-1] == "\'":
            res_tokens.append("\'")
            res_tokens.append(token[1:-1])
            res_tokens.append("\'")
        else:
            res_tokens.append(token)
    return res_tokens

def tokenize_code(code):
    from clang.cindex import Index
    index = Index.create()
    tu = index.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', code)])
    tokens = [token.spelling for token in tu.get_tokens(extent=tu.cursor.extent)]
    tokens = fix_char_string_tok(tokens)
    return tokens

################################################
# Extract templates

VARNAMES = re.compile(r'[A-Za-z]\w*', re.UNICODE)
NUMBERS = re.compile(r'\d+', re.UNICODE)
RESERVED = {
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel',
    'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor', 'bool',
    'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'char8_t',
    'class', 'co_await', 'co_return', 'co_yield', 'compl', 'concept', 'const',
    'const_cast', 'consteval', 'constexpr', 'continue', 'decltype', 'default',
    'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
    'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if',
    'import', 'inline', 'int', 'long', 'module', 'mutable', 'namespace', 'new',
    'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
    'private', 'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast',
    'requires', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
    'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this',
    'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
    'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t',
    'while', 'xor', 'xor_eq',
}


def can_placehold(token):
    return (
        (VARNAMES.match(token) or NUMBERS.match(token))
        and token not in RESERVED
    )


def match(text_tokens, code_tokens):
    text_tokens = text_tokens[:]
    code_tokens = code_tokens[:]
    placeholder = 1
    for i, x in enumerate(text_tokens):
        if x in code_tokens and can_placehold(x):
            # Replace all occurrences of x in code_tokens and text_tokens
            for j, y in enumerate(code_tokens):
                if y == x:
                    code_tokens[j] = '${}'.format(placeholder)
            for j, y in enumerate(text_tokens):
                if y == x:
                    text_tokens[j] = '${}'.format(placeholder)
            placeholder += 1
    return text_tokens, code_tokens


################################################
# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', '--no-blank', action='store_true',
            help='Replace blank utterance with something else')
    parser.add_argument('-c', '--clang',
            help='Use clang from this location to tokenize code')
    parser.add_argument('-H', '--has-header', action='store_true',
            help='Input is a TSV file with a header')
    parser.add_argument('-t', '--tokenize-only', action='store_true',
            help='Do not replace matching tokens with placeholders.')
    parser.add_argument('infile', help='TSV files with text and code columns')
    args = parser.parse_args()

    if args.clang:
        setup_clang(args.clang)
    
    with open(args.infile) as fin:
        if args.has_header:
            header = fin.readline().rstrip('\n').split('\t')

        for line in fin:
            if header:
                data = dict(zip(header, line.rstrip('\n').split('\t')))
                text, code = data['text'], data['code']
                if not text:
                    if args.no_blank:
                        text = 'DUMMY'
                    else:
                        continue
            else:
                text, code = line.rstrip('\n').split('\t')

            text_tokens = tokenize_text(text)
            if args.clang:
                code_tokens = tokenize_code(code)
            else:
                code_tokens = tokenize_text(code)
            if not args.tokenize_only:
                text_tokens, code_tokens = match(text_tokens, code_tokens)

            print('{}\t{}'.format(' '.join(text_tokens), ' '.join(code_tokens)))


if __name__ == '__main__':
    main()
