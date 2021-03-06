# This file is part of UMTagger <http://github.com/ufal/sigmorphon2019/>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import re

import fastText
import numpy as np

from unimorph import UniMorph

class UMDataset:
    IDS = 0
    FORMS = 1
    LEMMAS = 2
    UPOS = 3
    XPOS = 4
    FEATS = 5
    HEAD = 6
    DEPREL = 7
    DEPS = 8
    MISC = 9
    FACTORS = 10
    FACTORS_MAP = {"IDS": IDS, "FORMS": FORMS, "LEMMAS": LEMMAS, "UPOS": UPOS, "XPOS": XPOS,
                   "FEATS": FEATS, "HEAD": HEAD, "DEPREL": DEPREL, "DEPS": DEPS, "MISC": MISC}

    FEAT_COMPONENTS = 10
    EMBEDDINGS = 11
    ELMO = 12

    PAD = 0
    UNK = 1

    re_extras = re.compile(r"^#|^\d+-")

    class _Factor:
        def __init__(self, characters, train=None):
            self.words_map = train.words_map if train else {'<pad>': UMDataset.PAD, '<unk>': UMDataset.UNK}
            self.words = train.words if train else ['<pad>', '<unk>']
            self.word_ids = []
            self.strings = []
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {'<pad>': UMDataset.PAD, '<unk>': UMDataset.UNK}
                self.alphabet = train.alphabet if train else ['<pad>', '<unk>']
                self.charseqs_map = {'<pad>': UMDataset.PAD, '<unk>': UMDataset.UNK}
                self.charseqs = [[UMDataset.PAD], [UMDataset.UNK]]
                self.charseq_ids = []

    class FactorBatch:
        def __init__(self, word_ids, charseq_ids=None, charseqs=None, charseq_lens=None):
            self.word_ids = word_ids
            self.charseq_ids = charseq_ids
            self.charseqs = charseqs
            self.charseq_lens = charseq_lens

    def __init__(self, filename, embeddings=None, elmo=None, train=None, shuffle_batches=True, max_sentence_len=None, max_sentences=None, lr_allow_copy=None):
        # Create factors
        self._factors = []
        for f in range(self.FACTORS):
            self._factors.append(self._Factor(f == self.FORMS, train._factors[f] if train else None))

        self._extras = []
        self._lr_allow_copy = train._lr_allow_copy if train else lr_allow_copy
        lemma_dict_with_copy, lemma_dict_no_copy = {}, {}

        self.unimorph = train.unimorph if train else UniMorph()

        # Prepare embeddings
        if train is not None:
            self._embeddings = train._embeddings
            self._embeddings_casing = train._embeddings_casing
        elif embeddings is not None:
            self._embeddings = fastText.load_model(embeddings)
            with open("{}.casing".format(embeddings), "r") as embeddings_casing_file:
                self._embeddings_casing = embeddings_casing_file.read().strip()
        else:
            self._embeddings = None
            self._embeddings_casing = None

        # Load contextualized embeddings
        self._elmo = []
        if elmo:
            for elmo_path in elmo.split(","):
                with np.load(elmo_path) as elmo_file:
                    for i, (_, value) in enumerate(elmo_file.items()):
                        if max_sentence_len: value = value[:max_sentence_len]
                        if i >= len(self._elmo): self._elmo.append(value)
                        else: self._elmo[i] = np.concatenate([self._elmo[i], value], axis=1)
                    assert i + 1 == len(self._elmo)
        self._elmo_size = self._elmo[0].shape[1] if self._elmo else 0

        # Load the sentences
        with open(filename, "r", encoding="utf-8") as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")

                if line:
                    if self.re_extras.match(line):
                        if in_sentence:
                            while len(self._extras) < len(self._factors[0].word_ids): self._extras.append([])
                            while len(self._extras[-1]) <= len(self._factors[0].word_ids[-1]):
                                self._extras[-1].append("")
                        else:
                            while len(self._extras) <= len(self._factors[0].word_ids): self._extras.append([])
                            if not len(self._extras[-1]): self._extras[-1].append("")
                        self._extras[-1][-1] += ("\n" if self._extras[-1][-1] else "") + line
                        continue

                    if max_sentence_len and in_sentence and len(self._factors[0].strings[-1]) >= max_sentence_len:
                        continue

                    columns = line.split("\t")
                    for f in range(self.FACTORS):
                        factor = self._factors[f]
                        if not in_sentence:
                            if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.strings.append([])
                            if factor.characters: factor.charseq_ids.append([])

                        word = columns[f]
                        factor.strings[-1].append(word)

                        # Preprocess word
                        if f == self.FEATS:
                            word = ";".join(sorted(word.split(";")))
                            if not train: self.unimorph.add_features(word)
                        if f == self.LEMMAS and self._lr_allow_copy is not None:
                            word = self._gen_lemma_rule(columns[self.FORMS], columns[self.LEMMAS], self._lr_allow_copy)

                        # Character-level information
                        if factor.characters:
                            if word not in factor.charseqs_map:
                                factor.charseqs_map[word] = len(factor.charseqs)
                                factor.charseqs.append([])
                                for c in word:
                                    if c not in factor.alphabet_map:
                                        if train:
                                            c = '<unk>'
                                        else:
                                            factor.alphabet_map[c] = len(factor.alphabet)
                                            factor.alphabet.append(c)
                                    factor.charseqs[-1].append(factor.alphabet_map[c])
                            factor.charseq_ids[-1].append(factor.charseqs_map[word])

                        # Word-level information
                        if f == self.LEMMAS and self._lr_allow_copy is None:
                            factor.word_ids[-1].append(0)
                            lemma_dict_with_copy[self._gen_lemma_rule(columns[self.FORMS], word, True)] = 1
                            lemma_dict_no_copy[self._gen_lemma_rule(columns[self.FORMS], word, False)] = 1
                        else:
                            if word not in factor.words_map:
                                if train:
                                    word = '<unk>'
                                else:
                                    factor.words_map[word] = len(factor.words)
                                    factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and len(self._factors[self.FORMS].word_ids) >= max_sentences:
                        break

        # Finalize lemmas if needed
        if self._lr_allow_copy is None:
            self._lr_allow_copy = True if len(lemma_dict_with_copy) < len(lemma_dict_no_copy) else False
            lemmas = self._factors[self.LEMMAS]
            for i in range(len(lemmas.word_ids)):
                for j in range(len(lemmas.word_ids[i])):
                    word = self._gen_lemma_rule(self._factors[self.FORMS].strings[i][j], lemmas.strings[i][j], self._lr_allow_copy)
                    if word not in lemmas.words_map:
                        lemmas.words_map[word] = len(lemmas.words)
                        lemmas.words.append(word)
                    lemmas.word_ids[i][j] = lemmas.words_map[word]

        # Generate FEAT components for training
        self._components = None
        if not train:
            self._components = []
            feats = self._factors[self.FEATS]
            for i in range(len(feats.strings)):
                self._components.append(np.zeros([len(feats.strings[i]), self.unimorph.groups], dtype=np.uint8))
                for j in range(len(feats.strings[i])):
                    self._components[i][j, :] = self.unimorph.indices(feats.strings[i][j])

        # Compute sentence lengths
        sentences = len(self._factors[self.FORMS].word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._factors[self.FORMS].word_ids)):
            self._sentence_lens[i] = len(self._factors[self.FORMS].word_ids[i])

        # Shuffling initialization
        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))

        # Asserts
        if self._elmo:
            assert sentences == len(self._elmo)
            for i in range(sentences):
                assert self._sentence_lens[i] == len(self._elmo[i])

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._factors

    @property
    def embeddings_size(self):
        return self._embeddings.get_dimension() if self._embeddings else 0

    @property
    def elmo_size(self):
        return self._elmo_size

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))
            return True
        return False

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        factors = []
        for factor in self._factors:
            factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
            for i in range(batch_size):
                factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = factor.word_ids[batch_perm[i]]

        # Feat components
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len, self.unimorph.groups], np.int32)))
        if self._components is not None:
            for i in range(batch_size):
                factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = self._components[batch_perm[i]]

        # Embeddings
        forms = self._factors[self.FORMS]
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len, self.embeddings_size], np.float32)))
        if self._embeddings:
            lowercase = self._embeddings_casing == "uncased"
            for i in range(batch_size):
                for j, string in enumerate(forms.strings[batch_perm[i]]):
                    if lowercase: string = string.lower()
                    factors[-1].word_ids[i, j] = self._embeddings.get_word_vector(string)

        # Contextualized embeddings
        if self._elmo:
            factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len, self.elmo_size], np.float32)))
            for i in range(batch_size):
                factors[-1].word_ids[i, :len(self._elmo[batch_perm[i]])] = self._elmo[batch_perm[i]]

        # Character-level data
        for f, factor in enumerate(self._factors):
            if not factor.characters: continue

            factors[f].charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    factors[f].charseq_ids[i, j] = charseqs_map[charseq_id]

            factors[f].charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
            factors[f].charseqs = np.zeros([len(charseqs), np.max(factors[f].charseq_lens)], np.int32)
            for i in range(len(charseqs)):
                factors[f].charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        return self._sentence_lens[batch_perm], factors

    def write_sentence(self, output, index, overrides):
        for i in range(self._sentence_lens[index] + 1):
            # Start by writing extras
            if index < len(self._extras) and i < len(self._extras[index]) and self._extras[index][i]:
                print(self._extras[index][i], file=output)
            if i == self._sentence_lens[index]: break

            fields = []
            for f in range(self.FACTORS):
                factor = self._factors[f]
                field = factor.strings[index][i]

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    field = factor.words[overrides[f][i]]
                    if f == self.LEMMAS:
                        try:
                            field = self._apply_lemma_rule(fields[-1], field)
                        except:
                            import sys
                            print("Cannot decode lemma rule {}".format(field), file=sys.stderr)
                            pass

                fields.append(field)

            print("\t".join(fields), file=output)
        print(file=output)

    @staticmethod
    def _min_edit_script(source, target, allow_copy):
        a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
        for i in range(0, len(source) + 1):
            for j in range(0, len(target) + 1):
                if i == 0 and j == 0:
                    a[i][j] = (0, "")
                else:
                    if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                        a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                    if i and a[i-1][j][0] < a[i][j][0]:
                        a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                    if j and a[i][j-1][0] < a[i][j][0]:
                        a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
        return a[-1][-1][1]

    @staticmethod
    def _gen_lemma_rule(form, lemma, allow_copy):
        form = form.lower()

        previous_case = -1
        lemma_casing = ""
        for i, c in enumerate(lemma):
            case = "↑" if c.lower() != c else "↓"
            if case != previous_case:
                lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
            previous_case = case
        lemma = lemma.lower()

        best, best_form, best_lemma = 0, 0, 0
        for l in range(len(lemma)):
            for f in range(len(form)):
                cpl = 0
                while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
                if cpl > best:
                    best = cpl
                    best_form = f
                    best_lemma = l

        rule = lemma_casing + ";"
        if not best:
            rule += "a" + lemma
        else:
            rule += "d{}¦{}".format(
                UMDataset._min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
                UMDataset._min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
            )
        return rule

    @staticmethod
    def _apply_lemma_rule(form, lemma_rule):
        casing, rule = lemma_rule.split(";", 1)
        if rule.startswith("a"):
            lemma = rule[1:]
        else:
            form = form.lower()
            rules, rule_sources = rule[1:].split("¦"), []
            assert len(rules) == 2
            for rule in rules:
                source, i = 0, 0
                while i < len(rule):
                    if rule[i] == "→" or rule[i] == "-":
                        source += 1
                    else:
                        assert rule[i] == "+"
                        i += 1
                    i += 1
                rule_sources.append(source)

            try:
                lemma, form_offset = "", 0
                for i in range(2):
                    j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                    while j < len(rules[i]):
                        if rules[i][j] == "→":
                            lemma += form[offset]
                            offset += 1
                        elif rules[i][j] == "-":
                            offset += 1
                        else:
                            assert(rules[i][j] == "+")
                            lemma += rules[i][j + 1]
                            j += 1
                        j += 1
                    if i == 0:
                        lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
            except:
                lemma = form

        for rule in casing.split("¦"):
            if rule == "↓0": continue # The lemma is lowercased initially
            case, offset = rule[0], int(rule[1:])
            lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

        return lemma
