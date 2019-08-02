#!/usr/bin/env python3

# This file is part of UMTagger <http://github.com/ufal/sigmorphon2019/>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import collections
import sys

import numpy as np

class UniMorph:
    class Group:
        NONE = 0
        def __init__(self, ident):
            self._ident = ident
            self._values = ["<None>"]
            self._values_map = {"<None": self.NONE}

    def __init__(self):
        self._schema_index = {}
        with open("unimorph-schema.txt", "r") as schema_file:
            for line in schema_file:
                group, feature = line.rstrip("\n").split("\t")
                if feature not in self._schema_index:
                    self._schema_index[feature] = []
                self._schema_index[feature].append(group)

        self._groups = []
        self._groups_map = {}

    def add_features(self, features):
        groups = collections.defaultdict(lambda: set())

        for feature in features.replace("+", ";").replace("/", ";").replace("{", ";").replace("}", ";").split(";"):
            if feature and feature != "_":
                if feature not in self._schema_index:
                    print("Ignoring unknown UniMorpho feature {}".format(feature), file=sys.stderr)
                    continue
                for group in self._schema_index[feature]:
                    groups[group].add(feature)

        for group in groups:
            if group not in self._groups_map:
                self._groups_map[group] = len(self._groups)
                self._groups.append(self.Group(self._groups_map[group]))

            feature = ";".join(sorted(groups[group]))
            group = self._groups[self._groups_map[group]]
            if not feature in group._values_map:
                group._values_map[feature] = len(group._values)
                group._values.append(feature)

    @property
    def groups(self):
        return len(self._groups)

    @property
    def group_sizes(self):
        return [len(group._values) for group in self._groups]

    def indices(self, features):
        indices = np.zeros([len(self._groups)], dtype=np.uint8)
        groups = collections.defaultdict(lambda: set())

        for feature in features.replace("+", ";").replace("/", ";").replace("{", ";").replace("}", ";").split(";"):
            if feature and feature != "_":
                for group in self._schema_index[feature]:
                    groups[group].add(feature)

        for group in groups:
            feature = ";".join(sorted(groups[group]))
            group = self._groups_map[group]
            indices[group] = self._groups[group]._values_map[feature]

        return indices
