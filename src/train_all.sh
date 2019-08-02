#!/bin/bash

# This file is part of UMTagger <http://github.com/ufal/sigmorphon2019/>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

script="$1"; shift
cmdfile="$1"; shift
embeddings="$1"; shift
elmo="$1"; shift
global_args="$@"; shift

[ -f "$cmdfile" ] && { echo File $cmdfile already exists! >&2; exit ; }
>$cmdfile
while read code size rest; do
  args="$global_args"
  [ "$size" -lt 50000 ] && args="$args --we_dim=512 --cle_dim=256 --rnn_cell_dim=384"
  [ "$size" -ge 50000 ] && args="$args --we_dim=512 --cle_dim=256 --rnn_cell_dim=512"
  [ "$code" = ja_modern ] && args="$args --batch_size=16 --min_epoch_batches=1"
  [ "$code" = zh_cfl ] && args="$args --batch_size=16 --min_epoch_batches=1"
  [ -f "$embeddings$code.bin" ] && args="$args --embeddings=$embeddings$code.bin"
  [ -f "$elmo$code-train.npz" ] && args="$args --elmo=$elmo$code"
  echo withcuda ../../generated/venv-gpu/bin/python $script ../../data/$code/$code-um $args >>$cmdfile
done <../../data/sizes.txt

echo qsub -q "'gpu-ms.q@!dll2'" -l gpu=1,gpu_ram=8G,mem_free=24G,h_data=32G -j y -o $cmdfile.\\\$TASK_ID.log -t 1-$(cat $cmdfile | wc -l) -tc 1 arrayjob_runner $cmdfile
