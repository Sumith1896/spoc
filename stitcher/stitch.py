#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, os, sys, math, time
import argparse
import heapq
import subprocess
from enum import Enum
import itertools
import traceback


# Global arguments
ARGS = None


LINE_OFFSET = 5
SCORE_THRES = 1e6
ERR_BLACK_AMOUNT = -1e6


# indices in the respective tsv_files
class _inp():
    text = 0
    code = 1
    hitid = 2
    workerid = 3
    probid = 4
    subid = 5
    line = 6
    indent = 7

class _pred():
    text = 1
    gold_score = 2
    pred_score = 3
    gold = 4
    pred_best = 5

# errors and passing flags
class err():
    no_err = 0
    compile_err = 1
    runtime_err = 2
    mismatch_err = 3

class pass_test():
    none = 0
    public = 1
    both = 2


################################################
# Helper methods


def fix_strings(inp):
    res = ""
    temp_string = ""
    inside = False
    for i in range(len(inp)):
        if not inside:
            res += inp[i]
            if inp[i] == "\"":
                inside = True
            continue
        if inside:
            if inp[i] == "\"":
                inside = False
                if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                    res += temp_string[1:-1]
                else:
                    res += temp_string
                res += "\""
                temp_string = ""
            else:
                temp_string += inp[i]
    inp = res
    res = ""
    temp_string = ""
    inside = False
    for i in range(len(inp)):
        if not inside:
            res += inp[i]
            if inp[i] == "\'":
                inside = True
            continue
        if inside:
            if inp[i] == "\'":
                inside = False
                if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                    res += temp_string[1:-1]
                else:
                    res += temp_string
                res += "\'"
                temp_string = ""
            else:
                temp_string += inp[i]
    return res


def compile_code(code, probid, subid, compile_only=False):
    """
    Write the code to [probid]-[subid].cpp and compile it.
    Return None if the compilation succeeds.
    Otherwise, return the compiler message as a string.
    """
    unique_id = probid + "-" + subid
    with open(unique_id + ".cpp", "w") as src_file:
        src_file.write(code)

    if not compile_only:
        command = "timeout {} g++ {}.cpp -o {}".format(ARGS.gcc_timeout+1, unique_id, unique_id)
    else:
        command = "timeout {} g++ {}.cpp -c".format(ARGS.gcc_timeout+1, unique_id)
    
    try:
        process = subprocess.run(command, shell=True, timeout=ARGS.gcc_timeout, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return "g++ timeout!"
    
    if process.returncode == 0:
        return None
    else:
        return process.stderr.decode('utf8', 'backslashreplace')


def run_tests(code, probid, subid, test_name):
    """
    Run the code on test cases.
    Assume that the code is already compiled to [probid]-[subid].

    Return the error code (no_err, runtime_err, or mismatch_err) and extra info.

    Note: Does not clean up the files. Need to run cleanup afterwards.
    """
    unique_id = probid + "-" + subid
    objfile = unique_id
    inpfile = objfile + "_inp.txt"
    outfile = objfile + "_out.txt"
    prgfile = objfile + "_prg.txt"

    input_file = open(inpfile, "w")
    output_file = open(outfile, "w")

    testcases = "{}/{}/{}_{}.txt".format(
            ARGS.prog_dir, probid, probid, test_name)
    with open(testcases) as f:
        contents = f.readlines()

    error_code = err.no_err
    error_info = None
    num_test = 0
    counter = 0
    for line in contents:
        if line == "###ENDINPUT###\n":
            counter = 1
            continue
        if line == "###ENDOUTPUT###\n":
            input_file.close()
            output_file.close()
            command = "timeout {} ./{} < {} > {}".format(
                    ARGS.timeout, objfile, inpfile, prgfile)
            process = subprocess.run(command, shell=True, stderr=subprocess.PIPE)
            if process.returncode != 0:
                error_code = err.runtime_err
                if process.returncode == 124:
                    error_info = 'Timeout {}'
                else:
                    error_info = process.stderr.decode('utf8', 'backslashreplace')
                break
            if not compare_files(outfile, prgfile):
                error_code = err.mismatch_err
                error_info = 'Mismatch {}'.format(num_test)
                break
            num_test += 1
            counter = 0
            input_file = open(inpfile, "w")
            output_file = open(outfile, "w")
            continue
        if counter == 0:
            input_file.write(line)
        if counter == 1:
            output_file.write(line)

    input_file.close()
    output_file.close()
    return error_code, error_info


BIG_FILE_THRESHOLD = 1000000

def compare_files(outfile, prgfile):
    # Don't read big files
    if (
        os.path.getsize(prgfile) > BIG_FILE_THRESHOLD
        and os.path.getsize(outfile) < BIG_FILE_THRESHOLD
    ):
        return False
    with open(outfile, 'br') as fin:
        outdata = fin.read()
    with open(prgfile, 'br') as fin:
        return fin.read() == outdata


# cleanup generated files after stitching completes
def cleanup(objfile):
    if os.path.exists(objfile):
        os.remove(objfile)
    if os.path.exists(objfile + '_inp.txt'):
        os.remove(objfile + '_inp.txt')
    if os.path.exists(objfile + '_out.txt'):
        os.remove(objfile + '_out.txt')
    if os.path.exists(objfile + '_prg.txt'):
        os.remove(objfile + '_prg.txt')


################################################################
# Oracle stuff


def generate_report(inp_stmt, pred_stmt):
    rep_file = open("report.txt", "w")
    in_beam_exact_match = True
    
    for inp, pred in zip(inp_stmt, pred_stmt):
        # print text, best and gold if not dummy
        # if dummy, all scores are dummy
        if pred[_pred.text] == 'DUMMY':
            rep_file.write(inp[_inp.code] + "\t" + inp[_inp.code] + "\t" + inp[_inp.code] + "\t")
            rep_file.write("DUMMY\tDUMMY\tDUMMY\n")
            continue
        else:
            rep_file.write(pred[_pred.text] + "\t" + pred[_pred.pred_best] + "\t" + pred[_pred.gold] + "\t")
        # find gold ranks and check in_beam_exact_match to false if not in beam
        gold = pred[_pred.gold].strip()
        gold_found = False
        for i in range(_pred.pred_best, _pred.pred_best + ARGS.num_preds):
            if gold == pred[i].strip():
                gold_found, gold_rank = True, i - _pred.pred_best + 1
                break
        if not gold_found:
            gold_rank = "NA"
            in_beam_exact_match = False
        # report pred, gold probs and gold rank
        rep_file.write(str(pred[_pred.pred_score]) + " (" + str(math.exp(float(pred[3]))) + ")\t")
        rep_file.write(str(pred[_pred.gold_score]) + " (" + str(math.exp(float(pred[2]))) + ")\t")
        rep_file.write(str(gold_rank) + "\n")

    rep_file.close()
    return in_beam_exact_match


def oracle_code_check(code, probid, subid):
    unique_id = probid + "-" + subid
    # generate c++
    compile_errors = compile_code(code, probid, subid)
    if compile_errors is not None:
        cleanup(unique_id)
        return False
    # run testcases
    test_errors, _ = run_tests(code, probid, subid, 'testcases')
    cleanup(unique_id)
    return test_errors == err.no_err


def remove_braces_gold(gold):
    if len(gold) < 3:
        return gold
    gold = gold.strip()
    if gold[-1] == "{":
        gold = gold[:-1]
    if gold[0] == "}":
        gold = gold[1:]
    return gold


def true_oracle(inp_stmt, pred_stmt, probid, subid):
    code_header = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
    curr_i = 0
    # check for predictions in the i-th line with everything else gold
    for inp_i, pred_i in zip(inp_stmt, pred_stmt):
        if pred_i[_pred.text] == 'DUMMY':
            curr_i += 1
            continue
        gold_found = False
        # iterate over the i-th line predictions
        for i in range(_pred.pred_best, _pred.pred_best + ARGS.num_preds):
            curr_j = 0
            curr_ind, prev_line = 0, " "
            code = code_header
            # generate code with everything else gold except the i-th line
            for inp_j, pred_j in zip(inp_stmt, pred_stmt):            
                # find the number of tabs to insert
                indent = '\t' * int(inp_j[_inp.indent])
                tmp_ind = int(inp_j[_inp.indent])
                # if tabs are decreasing then add } if not closing already
                if tmp_ind < curr_ind:
                    if inp_j[_inp.code] != "}":
                        indent += "} "
                    if curr_ind - tmp_ind > 1:
                        indent += (curr_ind - tmp_ind - 1) * "} "
                # if tabs are increasing then add { if not open already
                elif tmp_ind > curr_ind:
                    if not prev_line or prev_line[-1] != "{":
                        indent += "{ "
                    if tmp_ind - curr_ind > 1:
                        indent += (tmp_ind - curr_ind - 1) * "{ "
                curr_ind = tmp_ind
                # pick the line of code
                if pred_j[_pred.text] == 'DUMMY' or curr_j != curr_i:
                    code += indent + remove_braces_gold(inp_j[_inp.code]) + "\n"
                    prev_line = remove_braces_gold(inp_j[_inp.code])
                else:
                    code += indent + fix_strings(pred_i[i]) + "\n"
                    prev_line = pred_j[i]
                curr_j += 1
            passing = oracle_code_check(code, probid, subid)
            if passing:
                gold_found = True
                break
        if not gold_found:
            return False
        curr_i += 1
    return True


def filter_error_message(message, unique_id):
    return '\n'.join(x.replace(unique_id + '.cpp', '')
        for x in message.split('\n')
        if x.startswith(unique_id)
    )


def detailed_oracle(inp_stmt, pred_stmt, probid, subid):
    unique_id = probid + "-" + subid
    code_header = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
    log_file = open('detailed-oracle.txt', 'w')
    curr_i, prob_list_i = 0, 0
    # check for predictions in the i-th line with everything else gold
    for curr_i, (inp_i, pred_i) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred_i[_pred.text] == 'DUMMY':
            continue
        # iterate over the i-th line predictions
        print('Processing line {} (prob_list_i = {}) ...'.format(curr_i, prob_list_i))
        for rank in range(ARGS.num_preds):
            print('Candidate {}'.format(rank))
            curr_j = 0
            curr_ind, prev_line = 0, " "
            code = code_header
            # generate code with everything else gold except the i-th line
            for inp_j, pred_j in zip(inp_stmt, pred_stmt):            
                # find the number of tabs to insert
                indent = '\t' * int(inp_j[_inp.indent])
                tmp_ind = int(inp_j[_inp.indent])
                # if tabs are decreasing then add } if not closing already
                if tmp_ind < curr_ind:
                    if inp_j[_inp.code] != "}":
                        indent += "} "
                    if curr_ind - tmp_ind > 1:
                        indent += (curr_ind - tmp_ind - 1) * "} "
                # if tabs are increasing then add { if not open already
                elif tmp_ind > curr_ind:
                    if not prev_line or prev_line[-1] != "{":
                        indent += "{ "
                    if tmp_ind - curr_ind > 1:
                        indent += (tmp_ind - curr_ind - 1) * "{ "
                curr_ind = tmp_ind
                # pick the line of code
                if pred_j[_pred.text] == 'DUMMY' or curr_j != curr_i:
                    code += indent + remove_braces_gold(inp_j[_inp.code]) + "\n"
                    prev_line = remove_braces_gold(inp_j[_inp.code])
                else:
                    code += indent + fix_strings(pred_i[_pred.pred_best + rank]) + "\n"
                    prev_line = pred_j[_pred.pred_best + rank]
                curr_j += 1
            passed, error, error_message = compile_and_run_tests(code, probid, subid, None)
            if passed == pass_test.none and error == err.compile_err:
                error_message = filter_error_message(error_message, unique_id)
            print('\t'.join(str(x) for x in [
                curr_i,
                prob_list_i,
                rank + 1,
                passed,
                error,
                repr(error_message),
            ]), file=log_file)
            log_file.flush()
        prob_list_i += 1
    log_file.close()


################################################
# Stitchers


def compile_and_run_tests(code, probid, subid, iter_count):
    """
    Compile the code, run on public and hidden test cases, then clean up.
    Return (pass_test code, err code, extra_info).
    """
    unique_id = probid + "-" + subid
    if ARGS.verbose:
        with open('verbose-{:05d}'.format(iter_count), 'w') as fout:
            fout.write(code)
            fout.write('###############################\n')
    # generate c++
    compile_errors = compile_code(code, probid, subid)
    if compile_errors is not None:
        if ARGS.verbose:
            print('{}: Compilation fails!'.format(iter_count))
            with open('verbose-{:05d}'.format(iter_count), 'a') as fout:
                fout.write('\n\n@@@ {} {}\n'.format(pass_test.none, err.compile_err))
                fout.write('\n################ COMPILE ERROR ################\n')
                fout.write(compile_errors)
        cleanup(unique_id)
        return pass_test.none, err.compile_err, compile_errors
    # run public test cases
    test_errors, test_error_info = run_tests(code, probid, subid, 'testcases_public')
    if test_errors != err.no_err:
        if ARGS.verbose:
            if test_errors == err.runtime_err:
                print('{}: Public test runtime error'.format(iter_count))
            else:
                print('{}: Public test mismatch'.format(iter_count))
            with open('verbose-{:05d}'.format(iter_count), 'a') as fout:
                fout.write('\n\n@@@ {} {}\n'.format(pass_test.none, test_errors))
                fout.write('Error: {}\n'.format(test_error_info))
                with open(unique_id + '_inp.txt', 'br') as fin:
                    fout.write('Input: {}\n'.format(repr(fin.read())))
                with open(unique_id + '_out.txt', 'br') as fin:
                    fout.write('Gold: {}\n'.format(repr(fin.read())))
                with open(unique_id + '_prg.txt', 'br') as fin:
                    fout.write('Pred: {}\n'.format(repr(fin.read())[:2000]))
        cleanup(unique_id)
        return pass_test.none, test_errors, test_error_info
    # run hidden test cases
    test_errors, test_error_info = run_tests(code, probid, subid, 'testcases_hidden')
    if test_errors != err.no_err:
        if ARGS.verbose:
            if test_errors == err.runtime_err:
                print('{}: Hidden test runtime error'.format(iter_count))
            else:
                print('{}: Hidden test mismatch'.format(iter_count))
            with open('verbose-{:05d}'.format(iter_count), 'a') as fout:
                fout.write('\n\n@@@ {} {}\n'.format(pass_test.public, test_errors))
                fout.write('Error: {}\n'.format(test_error_info))
                with open(unique_id + '_inp.txt', 'br') as fin:
                    fout.write('Input: {}\n'.format(repr(fin.read())))
                with open(unique_id + '_out.txt', 'br') as fin:
                    fout.write('Gold: {}\n'.format(repr(fin.read())))
                with open(unique_id + '_prg.txt', 'br') as fin:
                    fout.write('Pred: {}\n'.format(repr(fin.read())[:2000]))
        cleanup(unique_id)
        return pass_test.public, test_errors, test_error_info
    # success!
    if ARGS.verbose:
        print('{}: Succeeded!'.format(iter_count))
        with open('verbose-{:05d}'.format(iter_count), 'a') as fout:
            fout.write('\n\n@@@ {} {}\n'.format(pass_test.both, err.no_err))
    cleanup(unique_id)
    return pass_test.both, err.no_err, None


def stitch_top1(inp_stmt, pred_stmt, probid, subid):
    code_header = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
    code = code_header
    curr_ind = 0
    prev_line = " "
    for inp, pred in zip(inp_stmt, pred_stmt):
        indent = '\t' * int(inp[_inp.indent])
        tmp_ind = int(inp[_inp.indent])
        if tmp_ind < curr_ind:
            if inp[_inp.code] != "}":
                indent += "} "
            if curr_ind - tmp_ind > 1:
                indent += (curr_ind - tmp_ind - 1) * "} "
        elif tmp_ind > curr_ind:
            if not prev_line or prev_line[-1] != "{":
                indent += "{ "
            if tmp_ind - curr_ind > 1:
                indent += (tmp_ind - curr_ind - 1) * "{ "
        curr_ind = tmp_ind
        if pred[_pred.text] == 'DUMMY':
            code += indent + inp[_inp.code] + "\n"
            prev_line = inp[_inp.code]
        else:
            code += indent + fix_strings(pred[_pred.pred_best]) + " // " + inp[_inp.text] + "\n"
            prev_line = pred[5]

    passed, error, _ = compile_and_run_tests(code, probid, subid, 1)
    if passed == pass_test.none:
        return False, False
    elif passed == pass_test.public:
        return True, False
    else:
        return True, True


################################################
# Gibbs sampling


def stitch_gibbs(inp_stmt, pred_stmt, probid, subid, thres=0.8, smooth=0.5):
    import numpy as np
    sampled_idx = []
    prob_list = []
    # store the probabilities of the lines to be sampled in prob_list and
    # generate initial random indices for the lines to be sampled in sampled_idx
    for inp, pred in zip(inp_stmt, pred_stmt):
        curr_prob_list = []
        if pred[_pred.text] != 'DUMMY' and math.exp(float(pred[_pred.pred_score])) < thres:
            for i in range(_pred.pred_best + ARGS.num_preds, _pred.pred_best + 2 * ARGS.num_preds):
                curr_prob_list.append(float(pred[i]))
            curr_prob_list = [math.exp(x) for x in curr_prob_list]
            total_sum = sum(curr_prob_list)
            curr_prob_list = [x * 1.0 / total_sum for x in curr_prob_list]
            prob_list.append(curr_prob_list)
            coin_toss = np.random.uniform(0, 1, 1)[0]
            if coin_toss < smooth:
                sampled_idx.append(np.random.choice(ARGS.num_preds, 1)[0])
            else:
                sampled_idx.append(np.random.choice(ARGS.num_preds, 1, p=curr_prob_list)[0])
    iter_count = 0
    compile_count = 0
    while iter_count < ARGS.compile_budget:
        iter_count += 1
        code = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
        curr_ind = 0
        prev_line = " "
        curr_idx = 0
        for inp, pred in zip(inp_stmt, pred_stmt):
            indent = '\t' * int(inp[_inp.indent])
            tmp_ind = int(inp[_inp.indent])
            if tmp_ind < curr_ind:
                if inp[_inp.code] != "}":
                    indent += "} "
                if curr_ind - tmp_ind > 1:
                    indent += (curr_ind - tmp_ind - 1) * "} "
            elif tmp_ind > curr_ind:
                if not prev_line or prev_line[-1] != "{":
                    indent += "{ "
                if tmp_ind - curr_ind > 1:
                    indent += (tmp_ind - curr_ind - 1) * "{ "
            curr_ind = tmp_ind
            if pred[_pred.text] == 'DUMMY':
                code += indent + inp[_inp.code] + "\n"
                prev_line = inp[_inp.code]
            else:
                if math.exp(float(pred[_pred.pred_score])) < thres:
                    code += indent + fix_strings(pred[_pred.pred_best + sampled_idx[curr_idx]]) + " // " + inp[_inp.text] + "\n"
                    prev_line = pred[_pred.pred_best + sampled_idx[curr_idx]]
                    curr_idx += 1
                else:
                    code += indent + fix_strings(pred[_pred.pred_best]) + " // " + inp[_inp.text] + "\n"
                    prev_line = pred[_pred.pred_best]
        # run the program 
        passed, error, _ = compile_and_run_tests(code, probid, subid, iter_count)
        stat_file = open("gibbs_stats.txt", "a")
        if error != err.compile_err:
            compile_count += 1
        stat_file.write("Stats after iteration # " + str(iter_count) + "\n")
        stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))
        stat_file.write("Number of programs compiled:  " + str(compile_count) + "\n")
        stat_file.write(str(passed) + " " + str(error) + "\n")
        if passed == pass_test.none:
            idx_to_change = np.random.choice(len(sampled_idx), 1)[0]
            coin_toss = np.random.uniform(0, 1, 1)[0]
            if coin_toss < smooth:
                sampled_idx[idx_to_change] = np.random.choice(ARGS.num_preds, 1)[0]
            else:
                sampled_idx[idx_to_change] = np.random.choice(ARGS.num_preds, 1, p=prob_list[idx_to_change])[0]
            stat_file.write("continuing sampling...\n\n")
            stat_file.close()
        elif passed == pass_test.public:
            stat_file.write("passed public but failed hidden!\n\n")
            stat_file.close()
            return True, False
        else:
            stat_file.write("passed public and hidden!\n\n")
            stat_file.close()
            return True, True
    return False, False


################################################
# Best first search


class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.used = set()

    def add(self, log_prob, idx):
        idx = tuple(idx)
        if idx not in self.used:
            heapq.heappush(self.heap, (log_prob, idx))
            self.used.add(idx)
            if len(self.heap) > ARGS.max_heap:
                print('Heap limit exceeded. Committing suicide ...')
                exit(0)

    def pop(self):
        log_prob, idx = heapq.heappop(self.heap)
        #self.used.remove(idx)
        return log_prob, list(idx)

    def __len__(self):
        return len(self.heap)

    def empty(self):
        return len(self.heap) == 0


def stitch_best_first(inp_stmt, pred_stmt, probid, subid):
    prob_list = []
    for inp, pred in zip(inp_stmt, pred_stmt):
        curr_prob_list = []
        if pred[_pred.text] != 'DUMMY':
            for i in range(_pred.pred_best + ARGS.num_preds, _pred.pred_best + 2 * ARGS.num_preds):
                curr_prob_list.append(float(pred[i]))
            prob_list.append(curr_prob_list)

    iter_count, compile_count = 0, 0
    # create a heap and add the first element
    # since we want a max_heap, we add a the negative of log prob (by default it's a min heap)
    heap = PrioritySet()
    log_prob = 0
    idx = []
    for prob_list_ in prob_list:
        log_prob += prob_list_[0]
        idx.append(0)
    heap.add(-log_prob, idx)
    # iterate until not empty
    while not heap.empty() and iter_count < ARGS.compile_budget:
        iter_count += 1
        stat_file = open("best_first.txt", "a")
        stat_file.write("Stats after iteration # " + str(iter_count) + "\n")
        stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))
        log_prob, curr_idx = heap.pop()
        stat_file.write(str(log_prob) + "\n")
        stat_file.write(str(curr_idx) + "\n")
        if log_prob >= SCORE_THRES:
            stat_file.write('Log_prob threshold reached. Committing suicide ...')
            return False, False
        for i in range(len(curr_idx)):
            if curr_idx[i] < ARGS.num_preds - 1:
                new_idx = curr_idx.copy()
                new_idx[i] += 1
                # add neighbours to the heap
                log_prob = 0
                for j in range(len(new_idx)):
                    log_prob += prob_list[j][new_idx[j]]
                heap.add(-log_prob, new_idx)
        # find the code
        code = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
        prev_line = " "
        idx_count = 0
        curr_ind = 0
        for inp, pred in zip(inp_stmt, pred_stmt):
            indent = '\t' * int(inp[_inp.indent])
            tmp_ind = int(inp[_inp.indent])
            if tmp_ind < curr_ind:
                if inp[_inp.code] != "}":
                    indent += "} "
                if curr_ind - tmp_ind > 1:
                    indent += (curr_ind - tmp_ind - 1) * "} "
            elif tmp_ind > curr_ind:
                if not prev_line or prev_line[-1] != "{":
                    indent += "{ "
                if tmp_ind - curr_ind > 1:
                    indent += (tmp_ind - curr_ind - 1) * "{ "
            curr_ind = tmp_ind
            if pred[_pred.text] == 'DUMMY':
                code += indent + inp[_inp.code] + "\n"
                prev_line = inp[_inp.code]
            else:
                code += indent + fix_strings(pred[_pred.pred_best + curr_idx[idx_count]]) + " // " + inp[_inp.text] + "\n"
                prev_line = pred[_pred.pred_best + curr_idx[idx_count]]
                idx_count += 1
        # run the program 
        passed, error, _ = compile_and_run_tests(code, probid, subid, iter_count)
        if error != err.compile_err:
            compile_count += 1
        stat_file.write("Number of programs compiled:  " + str(compile_count) + "\n")
        stat_file.write(str(passed) + " " + str(error) + "\n")
        # if public didn't pass then proceed
        if passed == pass_test.none:
            stat_file.write("continuing best first search...\n\n")
            stat_file.close()
            continue
        elif passed == pass_test.public:
            stat_file.write("passed public but failed hidden!\n\n")
            stat_file.close()
            return True, False
        else:
            stat_file.write("passed public and hidden!\n\n")
            stat_file.close()
            return True, True
    return False, False


################################################
# Prefix pruning

def stitch_prefix_pruning(inp_stmt, pred_stmt, probid, subid):
    bad_prefixes = {}

    def check_if_bad(curr_idx, bad_prefixes):
        for l in range(len(curr_idx)):
            if l + 1 in bad_prefixes:
                if tuple(curr_idx[:l + 1]) in bad_prefixes[l + 1]:
                    return True, tuple(curr_idx[:l + 1])
        return False, None

    prob_list = []
    prob_list_idx_to_stmt_idx = []
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        curr_prob_list = []
        if pred[_pred.text] != 'DUMMY':
            for i in range(_pred.pred_best + ARGS.num_preds, _pred.pred_best + 2 * ARGS.num_preds):
                curr_prob_list.append(float(pred[i]))
            prob_list.append(curr_prob_list)
            prob_list_idx_to_stmt_idx.append(stmt_idx)
    stmt_idx_to_prob_list_idx = {x: i for (i, x) in enumerate(prob_list_idx_to_stmt_idx)}

    iter_count, compile_count = 0, 0
    prefix_ccount, heap_ccount, skip_ccount = 0, 0, 0
    # create a heap and add the first element
    # since we want a max_heap, we add a the negative of log prob (by default it's a min heap)
    heap = PrioritySet()
    log_prob = 0
    idx = []
    for prob_list_ in prob_list:
        log_prob += prob_list_[0]
        idx.append(0)
    heap.add(-log_prob, idx)
    # iterate until not empty
    while not heap.empty() and prefix_ccount + heap_ccount < ARGS.compile_budget:
        iter_count += 1
        stat_file = open("best_first_prefix_pruning.txt", "a")
        stat_file.write("Stats after iteration # " + str(iter_count) + "\n")
        stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))
        log_prob, curr_idx = heap.pop()
        stat_file.write(str(log_prob) + "\n")
        stat_file.write(str(curr_idx) + "\n")

        code = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
        prev_line = " "
        idx_count = 0
        curr_ind = 0
        for inp, pred in zip(inp_stmt, pred_stmt):
            indent = '\t' * int(inp[_inp.indent])
            tmp_ind = int(inp[_inp.indent])
            if tmp_ind < curr_ind:
                if inp[_inp.code] != "}":
                    indent += "} "
                if curr_ind - tmp_ind > 1:
                    indent += (curr_ind - tmp_ind - 1) * "} "
            elif tmp_ind > curr_ind:
                if not prev_line or prev_line[-1] != "{":
                    indent += "{ "
                if tmp_ind - curr_ind > 1:
                    indent += (tmp_ind - curr_ind - 1) * "{ "
            curr_ind = tmp_ind
            if pred[_pred.text] == 'DUMMY':
                code += indent + inp[_inp.code] + "\n"
                prev_line = inp[_inp.code]
            else:
                code += indent + fix_strings(pred[_pred.pred_best + curr_idx[idx_count]]) + " // " + inp[_inp.text] + "\n"
                prev_line = pred[_pred.pred_best + curr_idx[idx_count]]
                idx_count += 1

        from err_utils import NaiveErrDetector    
        err_detector = NaiveErrDetector(None)

        passed, error, raw_err_msg = compile_and_run_tests(code, probid, subid, iter_count)
        heap_ccount += 1
        add_to_heap_upto = len(curr_idx)
        if error != err.compile_err:
            stat_file.write("Compiled. Proceeding..." + "\n")
            compile_count += 1
        else:
            stat_file.write("Did not compile. Finding prefix..." + "\n")
            stat_file.write("Length of program: " + str(len(curr_idx)) + "\n")
            
            try:
                err_line_stmt_idx, err_msg = err_detector.detect(None, raw_err_msg)
            except Exception as e:
                # Commit suicide
                print('PANIC! {}'.format(e))
                print('PANIC! {}'.format(e), file=stat_file)
                stat_file.write(traceback.format_exc())
                exit(1)
            # resolve the error line to prob_list_idx
            if err_line_stmt_idx is not None:
                err_line = stmt_idx_to_prob_list_idx.get(err_line_stmt_idx)
            else:
                err_line = None
            # after resolving, check if it's a predicted line
            if err_line is None:
                print('Error line UNKNOWN: {}'.format(err_msg), file=stat_file)
            else:
                stat_file.write("Length of prob_list: " + str(len(prob_list)) + "\n")
                stat_file.write("Found err_line: " + str(err_line) + "\n")
                
                def find_code_upto(mid_idx, inp_stmt, pred_stmt):
                    code = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
                    prev_line = " "
                    idx_count = 0
                    curr_ind = 0
                    for inp, pred in zip(inp_stmt, pred_stmt):
                        if idx_count > mid_idx:
                            next_ind = int(inp[_inp.indent])
                            break
                        indent = '\t' * int(inp[_inp.indent])
                        tmp_ind = int(inp[_inp.indent])
                        if tmp_ind < curr_ind:
                            if inp[_inp.code] != "}":
                                indent += "} "
                            if curr_ind - tmp_ind > 1:
                                indent += (curr_ind - tmp_ind - 1) * "} "
                        elif tmp_ind > curr_ind:
                            if not prev_line or prev_line[-1] != "{":
                                indent += "{ "
                            if tmp_ind - curr_ind > 1:
                                indent += (tmp_ind - curr_ind - 1) * "{ "
                        curr_ind = tmp_ind
                        if pred[_pred.text] == 'DUMMY':
                            code += indent + inp[_inp.code] + "\n"
                            prev_line = inp[_inp.code]
                        else:
                            code += indent + fix_strings(pred[_pred.pred_best + curr_idx[idx_count]]) + " // " + inp[_inp.text] + "\n"
                            prev_line = pred[_pred.pred_best + curr_idx[idx_count]]
                            idx_count += 1

                    if not prev_line or prev_line[-1] == "{":
                        code += "} "
                    if not prev_line or ((prev_line.startswith("while") or prev_line.startswith("for") or prev_line.startswith("if") or prev_line.startswith("else")) and prev_line[-1] == ")") or prev_line.startswith("else"):
                        code += "{ }"
                    if curr_ind == 0 and next_ind == 1:
                        code += "{ }"
                    code += curr_ind * "} "
                    return code

                prev_pass_fail = None
                curr_prefix_ccount = 0
                if err_line != len(prob_list) - 1:
                    stat_file.write("err_line is in the middle, not at the end.." + "\n")
                    code = find_code_upto(err_line, inp_stmt, pred_stmt)
                    if compile_code(code, probid, subid, True) == None:
                        stat_file.write("Compiled! Going right..." + "\n")
                        prev_pass_fail = True
                    else:
                        stat_file.write("Did not compile! Going left..." + "\n")
                        prev_pass_fail = False
                    curr_prefix_ccount += 1
                else:
                    stat_file.write("err_line is at the end, not compiling for err_line..." + "\n")
                    prev_pass_fail = False

                prefix_inc = None
                if prev_pass_fail:
                    prefix_inc = 1
                else:
                    prefix_inc = -1

                curr_prefix_idx = err_line + prefix_inc
                found_a_prefix = False
                while curr_prefix_ccount < 3 and -1 <= curr_prefix_idx and \
                    curr_prefix_idx < len(prob_list):
                    stat_file.write("Finding code upto " + str(curr_prefix_idx) + "\n")
                    if curr_prefix_idx == -1:
                        # Empty code always compiles successfully
                        compile_res = True
                    else:
                        code = find_code_upto(curr_prefix_idx, inp_stmt, pred_stmt)
                        compile_res = compile_code(code, probid, subid, True)
                        curr_prefix_ccount += 1
                    if prev_pass_fail and compile_res != None:
                        stat_file.write("Finding a failing program and found!" + "\n")
                        found_a_prefix = True
                        break
                    if (not prev_pass_fail) and compile_res == None:
                        stat_file.write("Finding a passing program and found!" + "\n")
                        found_a_prefix = True
                        break
                    curr_prefix_idx += prefix_inc
                    stat_file.write("Didn't find a required program, progressing search..." + "\n")
              
                if not found_a_prefix:
                    stat_file.write("Didn't find a required program at all :(" + "\n")
                else:
                    if prev_pass_fail:
                        stat_file.write("BAD PREFIX: " + str(curr_idx[:curr_prefix_idx + 1]) + "\n")
                        add_to_heap_upto = curr_prefix_idx + 1
                        if (curr_prefix_idx + 1) not in bad_prefixes:
                            bad_prefixes[(curr_prefix_idx + 1)] = set()
                        bad_prefixes[(curr_prefix_idx + 1)].add(tuple(curr_idx[:curr_prefix_idx + 1]))
                    else:
                        stat_file.write("BAD PREFIX: " + str(curr_idx[:curr_prefix_idx + 2]) + "\n")
                        add_to_heap_upto = curr_prefix_idx + 2
                        if (curr_prefix_idx + 2) not in bad_prefixes:
                            bad_prefixes[(curr_prefix_idx + 2)] = set()
                        bad_prefixes[(curr_prefix_idx + 2)].add(tuple(curr_idx[:curr_prefix_idx + 2]))
                    stat_file.write("added to set..." + "\n")
                prefix_ccount += curr_prefix_ccount


        for i in range(add_to_heap_upto):
            if curr_idx[i] < ARGS.num_preds - 1:
                new_idx = curr_idx.copy()
                new_idx[i] += 1
                check_flag, check_value = check_if_bad(new_idx, bad_prefixes)
                if check_flag:
                    continue
                # add neighbours to the heap
                log_prob = 0
                for j in range(len(new_idx)):
                    log_prob += prob_list[j][new_idx[j]]
                heap.add(-log_prob, new_idx)

        stat_file.write("Number of programs compiled:  " + str(compile_count) + "\n")
        stat_file.write(str(passed) + " " + str(error) + "\n")
        # if public didn't pass then proceed
        if passed == pass_test.none:
            stat_file.write("continuing best first search...\n\n")
            stat_file.close()
            continue
        elif passed == pass_test.public:
            stat_file.write("passed public but failed hidden!\n\n")
            stat_file.write("PREFIX PRUNING STATS:\nheap compiler calls: " + str(heap_ccount) + \
                "\nprefix compiler calls: " + str(prefix_ccount) + \
                "\nskip compiler calls: " + str(skip_ccount) + \
                "\ntotal iterations: " + str(iter_count))
            stat_file.close()
            return True, False
        else:
            stat_file.write("passed public and hidden!\n\n")
            stat_file.write("PREFIX PRUNING STATS:\nheap compiler calls: " + str(heap_ccount) + \
                "\nprefix compiler calls: " + str(prefix_ccount) + \
                "\nskip compiler calls: " + str(skip_ccount) + \
                "\ntotal iterations: " + str(iter_count))
            stat_file.close()
            return True, True
    
    stat_file = open("best_first_prefix_pruning.txt", "a")
    stat_file.write("PREFIX PRUNING STATS:\nheap compiler calls: " + str(heap_ccount) + \
        "\nprefix compiler calls: " + str(prefix_ccount) + \
        "\nskip compiler calls: " + str(skip_ccount) + \
        "\ntotal iterations: " + str(iter_count))
    stat_file.close()
    return False, False


################################################
# Error detection


def stitch_error_detect(inp_stmt, pred_stmt, probid, subid):
    # There are 2 different indexing systems (both 0-based)
    # * stmt_idx: index of inp_stmt and pred_stmt (i.e., with DUMMY lines)
    #     Note: stmt_idx = real line number minus LINE_OFFSET
    # * prob_list_idx: index of prob_list (i.e., excluding DUMMY lines)
    prob_list = []
    prob_list_idx_to_stmt_idx = []
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        curr_prob_list = []
        if pred[_pred.text] != 'DUMMY':
            for i in range(_pred.pred_best + ARGS.num_preds, _pred.pred_best + 2 * ARGS.num_preds):
                curr_prob_list.append(float(pred[i]))
            prob_list.append(curr_prob_list)
            prob_list_idx_to_stmt_idx.append(stmt_idx)
    stmt_idx_to_prob_list_idx = {x: i for (i, x) in enumerate(prob_list_idx_to_stmt_idx)}

    iter_count, compile_count = 0, 0
    # create a heap and add the first element
    # since we want a max_heap, we add a the negative of log prob (by default it's a min heap)
    if ARGS.err_rebuild_heap:
        heap = FragilePrioritySet(prob_list)
    else:
        heap = PrioritySet()
        log_prob = 0
        idx = []
        for prob_list_ in prob_list:
            log_prob += prob_list_[0]
            idx.append(0)
        heap.add(-log_prob, idx)

    # error detector
    from err_utils import get_err_detector
    err_detector = get_err_detector(ARGS)
    # blacklist[prob_list_idx] = set of candidate_idxs that are blacklisted
    blacklist = [set() for _ in range(len(prob_list))]

    # iterate until not empty
    with open("error_detect.txt", "w") as stat_file:
        while not heap.empty() and iter_count < ARGS.compile_budget:
            stat_file.flush()
            iter_count += 1
            stat_file.write("Stats after iteration # " + str(iter_count) + "\n")
            stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))
            # log_prob: float
            # curr_idx: list[int] of length len(prob_list)
            log_prob, curr_idx = heap.pop()
            stat_file.write(str(log_prob) + "\n")
            stat_file.write(str(curr_idx) + "\n")
            if log_prob >= SCORE_THRES:
                stat_file.write('Log_prob threshold reached. Committing suicide ...')
                return False, False

            # detect if there is a blacklisted candidate
            found_blacklist = None
            for prob_list_idx, candidate_idx in enumerate(curr_idx):
                if candidate_idx in blacklist[prob_list_idx]:
                    found_blacklist = (prob_list_idx, candidate_idx)
                    break

            # decide whether to proceed with code generation            
            skip_synthesis = (found_blacklist is not None and ARGS.err_handling == 'black')
            if skip_synthesis:
                stat_file.write("Skip since {}.{} is in blacklist\n".format(*found_blacklist))
                iter_count -= 1
            else:
                # find the code
                code = "#include <bits/stdc++.h>\n\nusing namespace std;\n\n"
                code_lines = []         # For the error detection model
                prev_line = " "
                idx_count = 0
                curr_ind = 0
                for inp, pred in zip(inp_stmt, pred_stmt):
                    indent = '\t' * int(inp[_inp.indent])
                    tmp_ind = int(inp[_inp.indent])
                    if tmp_ind < curr_ind:
                        if inp[_inp.code] != "}":
                            indent += "} "
                        if curr_ind - tmp_ind > 1:
                            indent += (curr_ind - tmp_ind - 1) * "} "
                    elif tmp_ind > curr_ind:
                        if not prev_line or prev_line[-1] != "{":
                            indent += "{ "
                        if tmp_ind - curr_ind > 1:
                            indent += (tmp_ind - curr_ind - 1) * "{ "
                    curr_ind = tmp_ind
                    if pred[_pred.text] == 'DUMMY':
                        curr_code = inp[_inp.code]
                        code += indent + curr_code + "\n"
                    else:
                        curr_code = fix_strings(pred[_pred.pred_best + curr_idx[idx_count]])
                        code += indent + curr_code + " // " + inp[_inp.text] + "\n"
                        idx_count += 1
                    prev_line = curr_code
                    code_lines.append((inp[_inp.text], curr_code, curr_ind))

                # run the program 
                passed, error, raw_err_msg = compile_and_run_tests(code, probid, subid, iter_count)
                if error != err.compile_err:
                    compile_count += 1
                else:
                    # detect error message and blacklist the candidate
                    try:
                        err_line_stmt_idx, err_msg = err_detector.detect(code_lines, raw_err_msg)
                    except Exception as e:
                        # Commit suicide
                        print('PANIC! {}'.format(e))
                        print('PANIC! {}'.format(e), file=stat_file)
                        stat_file.write(traceback.format_exc())
                        exit(1)
                    # resolve the error line to prob_list_idx
                    if err_line_stmt_idx is not None:
                        err_line = stmt_idx_to_prob_list_idx.get(err_line_stmt_idx)
                    else:
                        err_line = None
                    # after resolving, check if it's a predicted line
                    if err_line is None:
                        print('Error line UNKNOWN: {}'.format(err_msg), file=stat_file)
                    else:
                        print('Error line {} -> {}: {}'.format(
                            err_line_stmt_idx, err_line, err_msg,
                        ), file=stat_file)
                        print(code_lines[err_line_stmt_idx], file=stat_file)
                        print('Blacklisting {}.{}'.format(err_line, curr_idx[err_line]), file=stat_file)
                        blacklist[err_line].add(curr_idx[err_line])
                        if ARGS.err_handling == 'gray':
                            prob_list[err_line][curr_idx[err_line]] -= ARGS.err_gray_amount
                        else:
                            prob_list[err_line][curr_idx[err_line]] = ERR_BLACK_AMOUNT
                        if ARGS.err_rebuild_heap:
                            heap.rebuild()

                stat_file.write("Number of programs compiled:  " + str(compile_count) + "\n")
                stat_file.write(str(passed) + " " + str(error) + "\n")
                # if public didn't pass then proceed
                if passed == pass_test.none:
                    stat_file.write("continuing best first search...\n\n")
                elif passed == pass_test.public:
                    stat_file.write("passed public but failed hidden!\n\n")
                    return True, False
                else:
                    stat_file.write("passed public and hidden!\n\n")
                    return True, True

            # add neighbours to the heap, but handle blacklisted candidates with care
            if ARGS.err_rebuild_heap:
                pass
            elif ARGS.err_handling == 'black':
                gen_neighbors_skip_blacks(heap, curr_idx, prob_list, blacklist)
            elif ARGS.err_handling == 'gray':
                gen_neighbors_down_grays(heap, curr_idx, prob_list, blacklist)
            else:
                raise ValueError('Unknown err_handling: {}'.format(ARGS.err_handling))

        return False, False


def gen_neighbors_skip_blacks(heap, curr_idx, prob_list, blacklist):
    for i in range(len(curr_idx)):
        new_idx = curr_idx.copy()
        new_idx[i] += 1
        log_prob = 0
        for j in range(len(new_idx)):
            # skip all blacklisted entries
            while new_idx[j] in blacklist[j]:
                new_idx[j] += 1
            if new_idx[j] >= ARGS.num_preds:
                break       # else below is not called
            log_prob += prob_list[j][new_idx[j]]
        else:
            heap.add(-log_prob, new_idx)


def gen_neighbors_down_grays(heap, curr_idx, prob_list, blacklist):
    for i in range(len(curr_idx)):
        new_idx = curr_idx.copy()
        new_idx[i] += 1
        # combos[j] = [new_idx[j]] if new_idx[j] is not blacklisted.
        # otherwise, it's [new_idx[j], new_idx[j] + 1, ...]
        # where the list continues until a non-blacklisted candidate is found
        combos = [[] for _ in range(len(new_idx))]
        for j in range(len(new_idx)):
            k = new_idx[j]
            while k < ARGS.num_preds:
                combos[j].append(k)
                if k not in blacklist[j]:
                    break
                k += 1
        for new_idx in itertools.product(*combos):
            log_prob = sum(prob_list[j][v] for (j, v) in enumerate(new_idx))
            heap.add(-log_prob, list(new_idx))


class FragilePrioritySet(object):
    """
    Rebuild the heap every time prob_list changes.
    """

    def __init__(self, prob_list):
        # prob_list[i][j] = logprob
        # prob_list is a shared object that can be updated outside the class
        self.prob_list = prob_list
        self.L = len(self.prob_list)
        # (neg_logprob, (j1, ..., jL), (r1, ..., rL))
        # true indices (j1, ..., jL) are used to track used combinations
        # ranks (r1, ..., rL) are used for finding neighbors
        self.heap = []
        self.rebuild()
        # Store combinations (j1, ..., jL) returnd during any reincarnation
        self.used = set()
        # Store combinations (j1, ..., jL) added to heap in this reincarnation
        self.added = set()

    def rebuild(self):
        # rankings[i][r] = j
        self.rankings = []
        for i, logprobs in enumerate(self.prob_list):
            ranked = sorted((-logprob, j) for (j, logprob) in enumerate(logprobs))
            self.rankings.append([j for (_, j) in ranked])
        self.heap.clear()
        self.added = set()
        self._add_r_tuple(tuple([0] * self.L))

    def _add_r_tuple(self, r_tuple):
        assert isinstance(r_tuple, tuple)
        j_tuple = tuple(self.rankings[i][r] for (i, r) in enumerate(r_tuple))
        if j_tuple in self.added:
            return
        assert len(j_tuple) == self.L
        logprob = sum(self.prob_list[i][j] for (i, j) in enumerate(j_tuple))
        heapq.heappush(self.heap, (-logprob, j_tuple, r_tuple))
        self.added.add(j_tuple)

    def add(self, log_prob, idx):
        raise NotImplementedError('Should not be called.')

    def pop(self):
        # Keep popping the heap until an unused item is found
        while True:
            neg_logprob, j_tuple, r_tuple = heapq.heappop(self.heap)
            # Immediately add all neighbors
            for i in range(self.L):
                if r_tuple[i] < ARGS.num_preds - 1:
                    neighbor = list(r_tuple)
                    neighbor[i] += 1
                    self._add_r_tuple(tuple(neighbor))
            # If not used, return it
            if j_tuple not in self.used:
                self.used.add(j_tuple)
                return (neg_logprob, list(j_tuple))

    def __len__(self):
        return len(self.heap)

    def empty(self):
        return len(self.heap) == 0


################################################


def stitch():
    probno, folder = ARGS.probno, ARGS.folder
    count = 0
    inp_stmt, pred_stmt = [], []

    # the following look extracts the input/pred lines for the probno specified
    # and passes it further for stitching
    with open(folder + '.tsv','r') as tsvin, open(folder + '.summary','r') as predin:
        tsvin.readline(), predin.readline()
        probid, subid = None, None
        
        while True:
            inp = tsvin.readline()
            if not inp:
                # Special handling for last line
                assert count == probno, \
                    'num problems = {} but probno = {}'.format(count, probno)
                break
            inp = inp.split('\t')
            pred = predin.readline().split("\t")
            if int(inp[_inp.line].strip()) == 0:
                if count == probno:
                    break
                count += 1
                probid, subid = inp[_inp.probid].strip(), inp[_inp.subid].strip()
                hitid = inp[_inp.hitid].strip()
            if count == probno:
                inp_stmt.append(inp)
                pred_stmt.append(pred)

    # generate a unique id for this program
    unique_id = "{:04d}-{}-{}".format(probno, probid, subid)
    print("Unique ID: " + unique_id)
    # make dir for this program to store .cpp & stats
    # os.system("rm -rf " + unique_id)
    os.system("mkdir " + os.path.join(ARGS.out_dir, unique_id))
    os.chdir(os.path.join(ARGS.out_dir, unique_id))
    global START_TIME
    START_TIME = time.time()
    # oracle
    if ARGS.oracle:
        # generate a report for the interface to parse
        in_beam_exact_match = generate_report(inp_stmt, pred_stmt)
        # if in_beam_exact_match, log it
        if in_beam_exact_match:
            tmp_f = open("in_beam_exact_match.txt", "w")
            tmp_f.close()
        # check if in beam for true oracle
        in_beam_true_oracle = true_oracle(inp_stmt, pred_stmt, probid, subid)
        if in_beam_true_oracle:
            tmp_f = open("in_beam_true_oracle.txt", "w")
            tmp_f.close()
    # detailed oracle
    if ARGS.detailed_oracle:
        detailed_oracle(inp_stmt, pred_stmt, probid, subid)
    # stitcher -- top 1
    if ARGS.top1:
        public, hidden = stitch_top1(inp_stmt, pred_stmt, probid, subid)
        if public:
            tmp_f = open("passed_top1_public.txt", "w")
            tmp_f.close()
        if hidden:
            tmp_f = open("passed_top1_hidden.txt", "w")
            tmp_f.close()
    # stitcher -- gibbs
    if ARGS.gibbs:
        public, hidden = stitch_gibbs(inp_stmt, pred_stmt, probid, subid)
        if public:
            tmp_f = open("passed_gibbs_public.txt", "w")
            tmp_f.close()
        if hidden:
            tmp_f = open("passed_gibbs_hidden.txt", "w")
            tmp_f.close()
    # stitcher -- best_first
    if ARGS.best_first:
        public, hidden = stitch_best_first(inp_stmt, pred_stmt, probid, subid)
        if public:
            tmp_f = open("passed_best_first_public.txt", "w")
            tmp_f.close()
        if hidden:
            tmp_f = open("passed_best_first_hidden.txt", "w")
            tmp_f.close()
    # stitcher -- best_first with prefix pruning
    if ARGS.prefix_pruning:
        public, hidden = stitch_prefix_pruning(inp_stmt, pred_stmt, probid, subid)
        if public:
            tmp_f = open("passed_prefix_pruning_public.txt", "w")
            tmp_f.close()
        if hidden:
            tmp_f = open("passed_prefix_pruning_hidden.txt", "w")
            tmp_f.close()
    # stitcher -- best_first with error detector
    if ARGS.error_detect:
        public, hidden = stitch_error_detect(inp_stmt, pred_stmt, probid, subid)
        if public:
            tmp_f = open("passed_error_detect_public.txt", "w")
            tmp_f.close()
        if hidden:
            tmp_f = open("passed_error_detect_hidden.txt", "w")
            tmp_f.close()
    os.chdir("../")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--prog-dir', default='./data/testcases',
            help='Path the codeforces-data repository, which contains test cases')
    parser.add_argument('--out-dir', default='./',
            help='Path to the output directory')
    parser.add_argument('--max-heap', type=int, default=999999,
            help='Suicide when heap is bigger than this')
    parser.add_argument('-t', '--timeout', type=int, default=2,
            help='Timeout for execution (in seconds)')
    parser.add_argument('-T', '--gcc-timeout', type=int, default=60,
            help='Timeout for compilation (in seconds)')
    parser.add_argument('-c', '--compile-budget', type=int, default=999999,
            help='Number of maximum g++ calls')
    parser.add_argument('-p', '--num-preds', type=int, default=100,
            help='Number of predictions per line')
    parser.add_argument('folder')
    parser.add_argument('probno', type=int)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--oracle', action='store_true',
            help='Run the oracles')
    group.add_argument('-O', '--detailed-oracle', action='store_true',
            help='Print detailed info about each candidate')
    group.add_argument('-x', '--top1', action='store_true',
            help='Run the top 1 stitcher')
    group.add_argument('-g', '--gibbs', action='store_true',
            help='Run the gibbs stitcher')
    group.add_argument('-b', '--best-first', action='store_true',
            help='Run the best first stitcher')
    group.add_argument('-B', '--prefix-pruning', action='store_true',
            help='Run the best first stitcher with prefix pruning')
    group.add_argument('-e', '--error-detect', action='store_true',
            help='Run the best first stitcher with error detection')

    group = parser.add_argument_group('error')
    group.add_argument('-r', '--err-rebuild-heap', action='store_true',
            help='Rebuild the heap after an error is detected')
    group.add_argument('--err-handling', choices=['black', 'gray'], default='black',
            help='Whether to blacklist or downweight error lines')
    group.add_argument('--err-gray-amount', type=float, default=10,
            help='(for graylisting) Amount to penalize error lines')
    group.add_argument('--err-detector',
            choices=['naive', 'template', 'binary', 'advanced'], default='naive',
            help='Name of the error detector to use')
    group.add_argument('--err-template-file',
            help='(template) TSV file with template stats')
    group.add_argument('--err-template-threshold', type=float, default=90.,
            help='(template) Minimum percentage for the template to trigger')
    group.add_argument('--err-server',
            help='(binary/advanced) Server + Port of the PyTorch detector')
    group.add_argument('--err-advanced-threshold', type=float, default=0.,
            help='(advanced) Minimum probability for the advanced detector to trigger')

    global ARGS
    ARGS = parser.parse_args()

    if os.environ.get('PROG_DIR'):
        ARGS.prog_dir = str(os.environ['PROG_DIR'])    

    if not os.path.isabs(ARGS.prog_dir):
        ARGS.prog_dir = os.path.abspath(ARGS.prog_dir)

    stitch()


if __name__ == "__main__":
    main()
