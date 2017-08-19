import re
import unicodedata
from util import numbers


# Input alphabet (63 symbols), plus french (26 symbols):
_pad         = '_'
_eos         = '~'
_uppercase   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_lowercase   = 'abcdefghijklmnopqrstuvwxyz'
_punctuation = '!\'(),-.:;?'
_space       = ' '
_fr_accent   = ''.join([a for a in unicodedata.normalize('NFKD', 'çèéêë') if a not in 'ce'])
_fr_uppercase = 'ÀÂÇÈÉÊËÎÏÔÙÛÜ'
_fr_lowercase = 'àâçèéêëîïôùûü'

_valid_input_chars = _uppercase + _lowercase + _punctuation + _space + _fr_accent + _fr_uppercase + _fr_lowercase
_trans_table = str.maketrans({chr(i): ' ' for i in range(256) if chr(i) not in _valid_input_chars})

_normal_symbols = _pad + _eos + _valid_input_chars
_num_normal_symbols = len(_normal_symbols)
_char_to_id = {c: i for i, c in enumerate(_normal_symbols)}
_id_to_char = {i: c for i, c in enumerate(_normal_symbols)}
_num_symbols = _num_normal_symbols
_whitespace_re = re.compile(r'\s+')


def num_symbols():
  '''Returns number of symbols in the alphabet.'''
  return _num_symbols


def to_sequence(text, force_lowercase=True, expand_abbreviations=True):
  '''Converts a string of text to a sequence of IDs for the symbols in the text'''
  text = text.strip()
  text = text.replace('"', '')
  text = unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore').decode('utf-8')
  sequence = []
  sequence += _text_to_sequence(text, force_lowercase, expand_abbreviations)
  sequence.append(_char_to_id[_eos])
  return sequence


def to_string(sequence, remove_eos=False):
  '''Returns the string for a sequence of characters.'''
  s = ''
  for sym in sequence:
    if sym < _num_normal_symbols:
      s += _id_to_char[sym]
  s = s.replace('}{', ' ')
  if remove_eos and s[-1] == _eos:
    s = s[:-1]
  return s


def _text_to_sequence(text, force_lowercase, expand_abbreviations):
  text = text.translate(_trans_table)
  if force_lowercase:
    text = text.lower()
  text = re.sub(_whitespace_re, ' ', text)
  text = text.replace('’', "'")
  text = text.replace('–', '--')
  text = text.replace('œ', 'oe')
  text = text.replace('“', '')
  text = text.replace('”', '')
  return [_char_to_id[c] for c in text]

