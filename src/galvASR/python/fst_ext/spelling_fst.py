from functools import partial
import math
from tempfile import NamedTemporaryFile

from galvASR.python.fst_ext import EPSILON_INT

import pywrapfst as openfst


def create_spelling_fst(word_table, alphabet_table, repeat_char,
                        self_transition_prob):
  # TODO: Should really prefer to enforce this via a ValueError.
  assert isinstance(word_table, openfst.SymbolTable)
  assert isinstance(alphabet_table, openfst.SymbolTable)
  assert 0.0 < self_transition_prob < 1.0

  WORD_EPSILON_STR = word_table.Find(EPSILON_INT)
  ALPHABET_EPSILON_STR = alphabet_table.Find(EPSILON_INT)
  next_transition_prob = 1. - self_transition_prob
  self_transition_cost = -math.log(self_transition_prob)
  next_transition_cost = -math.log(next_transition_prob)

  compiler = openfst.Compiler(
      fst_type="const",
      arc_type="log",
      isymbols=alphabet_table,
      osymbols=word_table,
      keep_isymbols=False,
      keep_osymbols=False)
  build_fst = partial(print, file=compiler)

  start_state_index = 0
  final_state_index = 1

  state_index = 2
  edge_format_str = "{start} {end} {char} {word} {weight:.5f}"
  for word in openfst.SymbolTableIterator(word_table):
    if word not in word_table:
      raise ValueError("Word {0} not in vocabulary {1}".format(
          word, word_table))
    for i, char in enumerate(word):
      if char not in alphabet_table:
        raise ValueError("Character {0} not in alphabet {1}".format(
            char, alphabet_table.name()))
      # Edge case: single-letter word?
      if i == 0:
        build_fst(
            edge_format_str.format(
                start=start_state_index,
                end=state_index if len(word) != 1 else final_state_index,
                char=char,
                word=WORD_EPSILON_STR,
                weight=next_transition_cost))
        build_fst(
            edge_format_str.format(
                start=start_state_index,
                end=start_state_index,
                char=ALPHABET_EPSILON_STR,
                word=WORD_EPSILON_STR,
                weight=self_transition_cost))
      else:
        # It is possible that this letter could be a repeat. Warning:
        # This doesn't check if a letter happens 3 times in a row, but
        # I don't know of any words in English that ever do that.
        last_char = word[i - 1]
        char = repeat_char if char == last_char else char

        if i == len(word) - 1:
          build_fst(
              edge_format_str.format(
                  start=state_index,
                  end=final_state_index,
                  char=char,
                  word=word,
                  weight=next_transition_cost))
          build_fst(
              edge_format_str.format(
                  start=state_index,
                  end=state_index,
                  char=ALPHABET_EPSILON_STR,
                  word=WORD_EPSILON_STR,
                  weight=self_transition_cost))
        else:
          build_fst(
              edge_format_str.format(
                  start=state_index,
                  end=state_index + 1,
                  char=char,
                  word=word,
                  weight=next_transition_cost))
          build_fst(
              edge_format_str.format(
                  start=state_index,
                  end=state_index,
                  char=ALPHABET_EPSILON_STR,
                  word=WORD_EPSILON_STR,
                  weight=self_transition_cost))
      state_index += 1
  # Add final state
  build_fst(str(final_state_index))

  S = compiler.compile().determinize()
  S.minimize()
  S = S.arcsort("olabel")
  return S


def create_alphabet_symbol_table(word_table):
  assert isinstance(word_table, openfst.SymbolTable)

  letters = set()

  for word in openfst.SymbolTableIterator(word_table):
    # We need words to be unicode strings, not bytes, for this to work
    # for all languages.
    assert isinstance(word, str)
    letters.add([letter for letter in word])

  with NamedTemporaryFile('w+t') as table_txt:
    table_txt.writelines(letter + '\n' for letter in sorted(letters))
    alphabet_table = openfst.SymbolTable.read(table_txt.name)
  return alphabet_table
