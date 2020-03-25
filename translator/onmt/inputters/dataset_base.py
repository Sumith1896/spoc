# coding: utf-8
import gzip
import json
from itertools import chain, starmap
from collections import Counter, defaultdict

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


################ MODIFIED ################

class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, examples, fields, sort_key, src_vocabs=None):
        super(Dataset, self).__init__(examples, fields)
        self.sort_key = sort_key
        self.src_vocabs = src_vocabs or []

    @classmethod
    def from_raw(cls, fields, readers, data, dirs, sort_key,
                    filter_pred=None):
        assert filter_pred is None, 'filter_pred != None fucks up the data'
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        return cls(examples, fields, sort_key, src_vocabs)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
    
    def save_jsonl(self, path):
        assert not self.src_vocabs or len(self.src_vocabs) == len(self.examples), \
                'Some examples got dropped!'
        # Attributes to save: 'src_vocabs', 'examples'
        with gzip.open(path, 'wt') as fout:
            for i, example in enumerate(self.examples):
                serialized = self._serialize_example(example)
                if self.src_vocabs:
                    serialized['src_vocab'] = self.src_vocabs[i].itos
                json.dump(serialized, fout)
                fout.write('\n')
    
    def _serialize_example(self, example):
        serialized = {}
        for k, v in example.__dict__.items():
            if isinstance(v, torch.Tensor):
                v = v.tolist()
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                v = v[0]
            serialized[k] = v
        return serialized

    @classmethod
    def load_jsonl(cls, path, fields, sort_key):
        examples = []
        src_vocabs = []
        with gzip.open(path, 'rt') as fin:
            for line in fin:
                serialized = json.loads(line)
                example, src_vocab = cls._deserialize_example(serialized, fields)
                examples.append(example)
                if src_vocab is not None:
                    src_vocabs.append(src_vocab)
        return cls(examples, fields, sort_key, src_vocabs)

    @classmethod
    def _deserialize_example(cls, serialized, fields):
        """ Return (example, src_vocab) """
        ex_dict = {}
        src_vocab = None
        for k, v in serialized.items():
            if k == 'src_vocab':
                src_vocab = MiniVocab(v)
            elif k in ('src_map', 'alignment'):
                ex_dict[k] = torch.tensor(v)
            else:
                ex_dict[k] = v
        ex_fields = {k: [(k, v)] for k, v in fields.items() if
                     k in ex_dict}
        example = Example.fromdict(ex_dict, ex_fields)
        return example, src_vocab


class MiniVocab(object):
    __slots__ = ['itos', 'stoi']

    def __init__(self, itos):
        self.itos = itos
        self.stoi = defaultdict(int)
        for i, x in enumerate(itos):
            self.stoi[x] = i

    def __len__(self):
        return len(self.itos)

##########################################
