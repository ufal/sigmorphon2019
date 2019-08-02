#!/bin/bash

# This file is part of UMTagger <http://github.com/ufal/sigmorphon2019/>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Generates Flair and ELMo embeddings where available.

PYTHON="~/venv/tf-1.12-cpu/bin/python3"
SRC="../ner_research_git/utils"
OUTPUT_DIR="../../generated/bert_embeddings"

submit_bert() {
  corpus=$1
  name=$2
  
  lang=${corpus:0:2}

  f=../../data/${corpus}/${corpus,,}-um-$name.conllu
  if [ ! -f $f ]; then echo "$f does not exist"; return 1; fi

  case $lang in
    en) bert_language=english;;
    zh) bert_language=chinese;;
    *) bert_language=multilingual;;
  esac

  echo qsub -q cpu-troja.q@* -l mem_free=8G,act_mem_free=8G,h_vmem=16G -pe smp 4 -N bert_${corpus}-${name} $PYTHON conllu_bert_embeddings.py $f $OUTPUT_DIR/${corpus,,}-${name}.npz --language=$bert_language --threads=4
 
  return 0
}

for dataset in train dev covered-test; do
  while read code size rest; do
    submit_bert $code $dataset
  done <../../data/sizes.txt
done
