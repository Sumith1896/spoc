# Error detection methods
import json
import math
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LINE_OFFSET = 5
TEXT_TOKENIZER = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def tokenize_err_msg(text):
    return TEXT_TOKENIZER.findall(text)


def parse_error(raw_err_msg, tokenize=True):
    """
    Return the first error line number and error message.

    Args:
        raw_err_msg (str): Raw string from g++
    Returns:
        stmt_index: real line number - LINE_OFFSET.
        That is, the line number where the first non preamble line is line 0,
        and where DUMMY lines are still included.
    """
    lines = raw_err_msg.split('\n')
    for line in lines:
        m = re.match('[^:]*:(\d+):[:0-9 ]+error: (.*)', line)
        if not m:
            continue
        lineno, message = m.groups()
        if tokenize:
            message = ' '.join(tokenize_err_msg(message))
        return int(lineno) - LINE_OFFSET, message.strip()
    return None, None


def post_request(server, post_fields):
    request = Request(server, urlencode(post_fields).encode())
    response = urlopen(request).read().decode()
    response = json.loads(response)
    return response


################################################


class ErrDetector(object):
    
    def __init__(self, args):
        pass

    def detect(self, code_lines, raw_err_msg):
        """
        Detect where the error actually happens based on the code and g++ message.

        Args:
            code_lines: tuple of (pseudocode str, code str, indent int)
            raw_err_msg: (str) error message from g++

        Returns:
            tuple (err_line, err_msg)
            - err_line: (int) The stmt_idx of the predicted error line
                (i.e., real line number minus LINE_OFFSET)
                To abstain prediction, let err_line = None
            - err_msg: (str) The tokenized message.
        """
        raise NotImplementedError


class NaiveErrDetector(ErrDetector):
    """
    Just return the first error line number from the g++ message.
    """

    def detect(self, code_lines, raw_err_msg):
        return parse_error(raw_err_msg)


class TemplateErrDetector(ErrDetector):
    """
    Match the error message against a template corpus.
    If a high-confident match is found, return the error line.
    Otherwise, abstain.
    """

    MODES = ['all', 'vars', 'none']

    def __init__(self, args):
        self.templates = {k: {} for k in self.MODES}
        with open(args.err_template_file) as fin:
            for line in fin:
                # mode total_count template line_offset count_percent
                line = line.rstrip('\n').split('\t')
                if float(line[4]) >= args.err_template_threshold:
                    self.templates[line[0]][line[2]] = int(line[3])
        for mode, templates in self.templates.items():
            print('Read {} {} templates'.format(len(templates), mode))

    def anonymize(self, msg, mode):
        if mode == 'none':
            return msg
        if mode == 'vars':
            return re.sub('‘[A-Za-z0-9_ ]*’', '@@@', msg)
        return re.sub('‘[^’]*’', '@@@', msg)

    def detect(self, code_lines, raw_err_msg):
        lineno, msg = parse_error(raw_err_msg, tokenize=False)
        if msg is None:
            return None, None
        tokenized_msg = ' '.join(tokenize_err_msg(msg)).strip()
        for mode in self.MODES:
            anon_msg = self.anonymize(msg, mode)
            if anon_msg in self.templates[mode]:
                return lineno - self.templates[mode][anon_msg], tokenized_msg
        return None, tokenized_msg


class BinaryErrDetector(ErrDetector):
    """
    Ask the PyTorch server if the error line from g++ is correct.
    If not, abstain.
    """

    def __init__(self, args):
        self.server = args.err_server
        self.info = {'probno': args.probno}

    def detect(self, code_lines, raw_err_msg):
        lineno, msg = parse_error(raw_err_msg, tokenize=True)
        if msg is None:
            return None, None
        # Call the server
        q = {
            'info': self.info,
            'code_lines': code_lines,
            'err_line': {
                'lineno': lineno,
                'msg': msg,
            }
        }
        response = post_request(self.server, {'q': json.dumps(q)})
        if response['pred'][0]:
            return lineno, msg
        return None, msg


class AdvancedErrDetector(ErrDetector):
    """
    Ask the PyTorch server what the actual error line is.
    """

    def __init__(self, args):
        self.server = args.err_server
        self.info = {'probno': args.probno}
        self.threshold = args.err_advanced_threshold

    def detect(self, code_lines, raw_err_msg):
        lineno, msg = parse_error(raw_err_msg, tokenize=True)
        if msg is None:
            return None, None
        # Call the server
        q = {
            'info': self.info,
            'code_lines': code_lines,
            'err_line': {
                'lineno': lineno,
                'msg': msg,
            }
        }
        response = post_request(self.server, {'q': json.dumps(q)})
        argmax = response['pred'][0]
        probs = self.softmax(response['logit'][0])
        if probs[argmax] >= self.threshold:
            return argmax, msg
        return None, msg

    def softmax(self, numbers):
        numbers = [math.exp(x - max(numbers)) for x in numbers]
        return [x / sum(numbers) for x in numbers]


################################################


def get_err_detector(args):
    if args.err_detector == 'naive':
        return NaiveErrDetector(args)
    if args.err_detector == 'template':
        return TemplateErrDetector(args)
    if args.err_detector == 'binary':
        return BinaryErrDetector(args)
    if args.err_detector == 'advanced':
        return AdvancedErrDetector(args)
    raise ValueError('Unknown detector: {}'.format(args.err_detector))
