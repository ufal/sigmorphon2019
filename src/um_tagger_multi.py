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
import json

import numpy as np
import tensorflow as tf

import evaluate_2019_task2
from um_dataset import UMDataset

class Network:
    METRICS = ["Lemmas", "LemmasLev", "Tags", "TagsF1"]

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads,
                                                                       allow_soft_placement=True))

    def construct(self, args, num_words, num_chars, factor_words, unimorph, predict_only):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            if args.embeddings_size: self.embeddings = tf.placeholder(tf.float32, [None, None, args.embeddings_size])
            if args.elmo_size: self.elmo = tf.placeholder(tf.float32, [None, None, args.elmo_size])
            self.factors = dict((factor, tf.placeholder(tf.int32, [None, None])) for factor in args.factors)
            if args.components_regularization: self.factors["FEAT_COMPONENTS"] = tf.placeholder(tf.int32, [None, None, unimorph.groups])
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Word embeddings
            inputs = []
            if args.we_dim:
                word_embeddings = tf.get_variable("word_embeddings", shape=[num_words, args.we_dim], dtype=tf.float32)
                inputs.append(tf.nn.embedding_lookup(word_embeddings, self.word_ids))

            # Character-level embeddings
            character_embeddings = tf.get_variable("character_embeddings", shape=[num_chars, args.cle_dim], dtype=tf.float32)
            characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.charseqs)
            characters_embedded = tf.layers.dropout(characters_embedded, rate=args.dropout, training=self.is_training)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                characters_embedded, sequence_length=self.charseq_lens, dtype=tf.float32)
            cle = tf.concat([state_fwd, state_bwd], axis=1)
            cle_inputs = tf.nn.embedding_lookup(cle, self.charseq_ids)
            # If CLE dim is half WE dim, we add them together, which gives
            # better results; otherwise we concatenate CLE and WE.
            if 2 * args.cle_dim == args.we_dim:
                inputs[-1] += cle_inputs
            else:
                inputs.append(cle_inputs)

            # Pretrained embeddings
            if args.embeddings:
                inputs.append(self.embeddings)

            # Contextualized embeddings
            if args.elmo_size:
                inputs.append(self.elmo)

            # All inputs done
            inputs = tf.concat(inputs, axis=2)

            # RNN layers
            hidden_layer = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="word-level-rnn-{}".format(i))
                previous = hidden_layer
                hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                if i: hidden_layer += previous

            # Tagger
            loss = 0
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            weights_sum = tf.reduce_sum(weights)
            self.predictions, self.prediction_probs = {}, {}
            for factor in args.factors:
                factor_layer = hidden_layer
                for _ in range(args.factor_layers):
                    factor_layer += tf.layers.dropout(tf.layers.dense(factor_layer, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                if factor == "LEMMAS": factor_layer = tf.concat([factor_layer, cle_inputs], axis=2)
                output_layer = tf.layers.dense(factor_layer, factor_words[factor])
                self.predictions[factor] = tf.argmax(output_layer, axis=2, output_type=tf.int32)
                self.prediction_probs[factor] = tf.nn.softmax(output_layer, axis=2)

                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.factors[factor], factor_words[factor]) * (1 - args.label_smoothing) + args.label_smoothing / factor_words[factor]
                    loss += tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.factors[factor], output_layer, weights=weights)

                if factor == "FEATS" and args.components_regularization:
                    components = self.factors["FEAT_COMPONENTS"]
                    for i, size in enumerate(unimorph.group_sizes):
                        output_layer = tf.layers.dense(factor_layer, size)
                        if args.label_smoothing:
                            gold_labels = tf.one_hot(components[:, :, i], size) * (1 - args.label_smoothing) + args.label_smoothing / size
                            loss += args.components_regularization / unimorph.groups * tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                        else:
                            loss += args.components_regularization / unimorph.groups * tf.losses.sparse_softmax_cross_entropy(components[:, :, i], output_layer, weights=weights)

            # Pretrain saver
            self.saver_inference = tf.train.Saver(max_to_keep=1)
            if predict_only: return

            # Training
            self.global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

            # Train saver
            self.saver_train = tf.train.Saver(max_to_keep=1)

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [
                    tf.contrib.summary.scalar("train/loss", loss),
                    tf.contrib.summary.scalar("train/lr", self.learning_rate)]
                for factor in args.factors:
                    self.summaries["train"].append(tf.contrib.summary.scalar(
                        "train/{}".format(factor),
                        tf.reduce_sum(tf.cast(tf.equal(self.factors[factor], self.predictions[factor]), tf.float32) * weights) /
                        weights_sum))

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.metrics = dict((metric, tf.placeholder(tf.float32, [])) for metric in self.METRICS)
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = []
                    for metric in self.METRICS:
                        self.summaries[dataset].append(tf.contrib.summary.scalar("{}/{}".format(dataset, metric),
                                                                                 self.metrics[metric]))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, learning_rate, args):
        batches, at_least_one_epoch = 0, False
        while batches < args.min_epoch_batches:
            while not train.epoch_finished():
                sentence_lens, b = train.next_batch(args.batch_size)
                if args.word_dropout:
                    mask = np.random.binomial(n=1, p=args.word_dropout, size=b[train.FORMS].word_ids.shape)
                    b[train.FORMS].word_ids = (1 - mask) * b[train.FORMS].word_ids + mask * train.UNK

                feeds = {self.is_training: True, self.learning_rate: learning_rate, self.sentence_lens: sentence_lens,
                         self.charseqs: b[train.FORMS].charseqs, self.charseq_lens: b[train.FORMS].charseq_lens,
                         self.word_ids: b[train.FORMS].word_ids, self.charseq_ids: b[train.FORMS].charseq_ids}
                if args.embeddings_size:
                    if args.word_dropout:
                        mask = np.random.binomial(n=1, p=args.word_dropout, size=b[train.EMBEDDINGS].word_ids.shape[0:2] + (1,))
                        b[train.EMBEDDINGS].word_ids = (1 - mask) * b[train.EMBEDDINGS].word_ids
                    feeds[self.embeddings] = b[train.EMBEDDINGS].word_ids
                if args.elmo_size:
                    feeds[self.elmo] = b[train.ELMO].word_ids
                for factor in args.factors:
                    feeds[self.factors[factor]] = b[train.FACTORS_MAP[factor]].word_ids
                if args.components_regularization:
                    feeds[self.factors["FEAT_COMPONENTS"]] = b[train.FEAT_COMPONENTS].word_ids
                self.session.run([self.training, self.summaries["train"]], feeds)
                batches += 1
                if at_least_one_epoch: break
            at_least_one_epoch = True

    def predict(self, treebank, args, probs_basename=None):
        import io
        conllu, sentences = io.StringIO(), 0
        probs = {factor: [] for factor in args.factors}

        while not treebank.data.epoch_finished():
            # Generate batch
            sentence_lens, b = treebank.data.next_batch(args.batch_size)

            # Prepare feeds
            feeds = {self.is_training: False, self.sentence_lens: sentence_lens,
                     self.charseqs: b[train.FORMS].charseqs, self.charseq_lens: b[train.FORMS].charseq_lens,
                     self.word_ids: b[train.FORMS].word_ids, self.charseq_ids: b[train.FORMS].charseq_ids}
            if args.embeddings_size:
                feeds[self.embeddings] = b[train.EMBEDDINGS].word_ids
            if args.elmo_size:
                feeds[self.elmo] = b[train.ELMO].word_ids


            # Prepare targets and run the network
            targets = [self.prediction_probs]
            prediction_probs, = self.session.run(targets, feeds)
            if treebank.maps:
                remapped_probs = {}
                for factor in args.factors:
                    remapped_probs[factor] = np.zeros(prediction_probs[factor].shape[0:2] + (len(treebank.maps[factor]),))
                    for i in range(remapped_probs[factor].shape[0]):
                        for j in range(remapped_probs[factor].shape[1]):
                            remapped_probs[factor][i, j] = prediction_probs[factor][i, j][treebank.maps[factor]]
                prediction_probs = remapped_probs

            if probs_basename:
                for i in range(len(sentence_lens)):
                    for factor in args.factors:
                        probs[factor].append(prediction_probs[factor][i][:sentence_lens[i]])

            # Generate output
            for i in range(len(sentence_lens)):
                overrides = [None] * treebank.data.FACTORS
                for factor in args.factors:
                    overrides[treebank.data.FACTORS_MAP[factor]] = np.argmax(prediction_probs[factor][i], axis=1)
                (treebank.data_original if treebank.data_original else treebank.data).write_sentence(conllu, sentences, overrides)
                sentences += 1

        # Generate probs file
        if probs_basename:
            for factor in args.factors:
                np.save("{}-{}.npy".format(probs_basename, factor.lower()), np.concatenate(probs[factor]))

        return conllu.getvalue()

    def evaluate(self, dataset_name, treebank, args):
        import io

        conllu = self.predict(getattr(treebank, dataset_name), args)
        results = evaluate_2019_task2.manipulate_data(evaluate_2019_task2.input_pairs(getattr(treebank, dataset_name).data_gold, conllu.split("\n")))
        metrics = collections.OrderedDict((name, value) for name, value in zip(self.METRICS, results))
        if treebank.summary_writer:
            step = self.session.run(self.global_step)
            for name, value in metrics.items():
                summary = tf.summary.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = dataset_name + "/" + name
                treebank.summary_writer.add_summary(summary, step)
            treebank.summary_writer.flush()
        else:
            self.session.run(self.summaries[dataset_name],
                             dict((self.metrics[metric], metrics[metric]) for metric in self.METRICS))
        return metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import json
    import os
    import sys
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", default=None, type=str, help="Input data basename")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--components_regularization", default=0., type=float, help="Weight of component regularization.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--elmo", default=None, type=str, help="External contextualized embeddings to use.")
    parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4,5:5e-5", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--factors", default="LEMMAS,FEATS", type=str, help="Factors to predict.")
    parser.add_argument("--factor_layers", default=1, type=int, help="Per-factor layers.")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--max_sentence_len", default=200, type=int, help="Max sentence length.")
    parser.add_argument("--min_epoch_batches", default=300, type=int, help="Minimum number of batches per epoch.")
    parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    parser.add_argument("--predict_probs", default=None, type=str, help="Predict probabilities to given file.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=3, type=int, help="RNN layers.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--treebanks", default=None, type=str, help="Treebanks to evaluate on.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    args = parser.parse_args()

    if args.predict:
        # Load saved options from the model
        with open("{}/options.json".format(args.predict), mode="r") as options_file:
            args = argparse.Namespace(**json.load(options_file))
        parser.parse_args(namespace=args)
    else:
        # Create logdir name
        if args.exp is None:
            args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        do_not_log = {"exp", "max_sentence_len", "min_epoch_batches", "predict", "predict_probs", "rnn_cell", "threads", "treebanks"}
        args.logdir = "logs/{}-{}".format(
            args.exp,
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("[^,]*/", "", value) if type(value) == str else value)
                      for key, value in sorted(vars(args).items()) if key not in do_not_log))
        )
        if not os.path.exists("logs"): os.mkdir("logs")
        if not os.path.exists(args.logdir): os.mkdir(args.logdir)

        # Dump passed options, command line and script name
        with open("{}/options.json".format(args.logdir), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True)
        with open("{}/cmd".format(args.logdir), mode="w") as cmd_file:
            print("\n".join(sys.argv[1:]), file=cmd_file)
        with open("{}/script".format(args.logdir), mode="w") as script_file:
            print(os.path.basename(__file__), file=script_file)

    # Postprocess args
    args.factors = args.factors.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]
    if args.treebanks: args.treebanks = [treebank.split(":") for treebank in args.treebanks.split(",")]

    # Load data
    train = UMDataset("{}-train.conllu".format(args.basename), max_sentence_len=args.max_sentence_len,
                      embeddings=args.embeddings, elmo=re.sub("(?=,|$)", "-train.npz", args.elmo) if args.elmo and not args.predict else None)
    args.embeddings_size = train.embeddings_size
    args.elmo_size = train.elmo_size
    if args.predict:
        predict = UMDataset("/dev/stdin", train=train, shuffle_batches=False,
                            elmo=re.sub("(?=,|$)", "-dev.npz", args.elmo) if args.elmo else None)
        args.elmo_size = predict.elmo_size
    else:
        class TreebankSet:
            def __init__(self, data, data_gold=None, maps=None, data_original=None):
                self.data, self.data_gold, self.maps, self.data_original = data, data_gold, maps, data_original
        class Treebank:
            def __init__(self, code, summary_writer, log, dev, test):
                self.code, self.summary_writer, self.log, self.dev, self.test = code, summary_writer, log, dev, test
        treebanks = []
        if args.treebanks:
            for treebank, elmo in args.treebanks:
                treebank_train = UMDataset("{}-train.conllu".format(treebank), max_sentence_len=args.max_sentence_len, lr_allow_copy=train._lr_allow_copy)
                treebank_maps = {}
                for factor in args.factors:
                    treebank_maps[factor] = np.zeros([len(treebank_train.factors[treebank_train.FACTORS_MAP[factor]].words)], dtype=np.int32)
                    for i, word in enumerate(treebank_train.factors[treebank_train.FACTORS_MAP[factor]].words):
                        treebank_maps[factor][i] = train.factors[train.FACTORS_MAP[factor]].words_map[word]
                treebanks.append(Treebank(
                    os.path.basename(treebank), None, None,
                    TreebankSet(UMDataset("{}-dev.conllu".format(treebank), train=train, shuffle_batches=False,
                                          elmo=elmo + "-dev.npz" if elmo else None),
                                open("{}-dev.conllu".format(treebank), "r", encoding="utf-8").readlines(),
                                treebank_maps,
                                UMDataset("{}-dev.conllu".format(treebank), train=treebank_train, shuffle_batches=False)),
                    TreebankSet(UMDataset("{}-covered-test.conllu".format(treebank), train=train, shuffle_batches=False,
                                          elmo=elmo + "-covered-test.npz" if elmo else None),
                                None, treebank_maps,
                                UMDataset("{}-covered-test.conllu".format(treebank), train=treebank_train, shuffle_batches=False))))
        else:
            treebanks.append(Treebank(
                "", None, None,
                TreebankSet(UMDataset("{}-dev.conllu".format(args.basename), train=train, shuffle_batches=False,
                                      elmo=re.sub("(?=,|$)", "-dev.npz", args.elmo) if args.elmo else None),
                            open("{}-dev.conllu".format(args.basename), "r", encoding="utf-8").readlines()),
                TreebankSet(UMDataset("{}-covered-test.conllu".format(args.basename), train=train, shuffle_batches=False,
                                      elmo=re.sub("(?=,|$)", "-covered-test.npz", args.elmo) if args.elmo else None))))
        args.treebanks = treebanks

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args=args,
                      num_words=len(train.factors[train.FORMS].words),
                      num_chars=len(train.factors[train.FORMS].alphabet),
                      factor_words=dict((factor, len(train.factors[train.FACTORS_MAP[factor]].words)) for factor in args.factors),
                      unimorph=train.unimorph,
                      predict_only=args.predict)

    if args.predict:
        network.saver_inference.restore(network.session, "{}/checkpoint-inference".format(args.predict))
        conllu = network.predict(TreebankSet(predict), args, probs_basename=args.predict_probs)
        print(conllu, end="")
    else:
        for treebank in treebanks:
            treebank.summary_writer = tf.summary.FileWriter("{}/{}".format(args.logdir, treebank.code)) if treebank.code else None
            treebank.log = open("{}/{}/log".format(args.logdir, treebank.code), "w")
            for factor in args.factors:
                print("{}: {}".format(factor, len(treebank.dev.data.factors[train.FACTORS_MAP[factor]].words)), file=treebank.log, flush=True)
        print("Tagging with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items()))), flush=True)

        for i, (epochs, learning_rate) in enumerate(args.epochs):
            for epoch in range(epochs):
                network.train_epoch(train, learning_rate, args)

                for treebank in treebanks:
                    metrics = network.evaluate("dev", treebank, args)
                    metrics_log = ", ".join(("{}: {:.2f}".format(metric, metrics[metric]) for metric in metrics))
                    for f in [sys.stderr, treebank.log]:
                        print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)

        network.saver_inference.save(network.session, "{}/checkpoint-inference".format(args.logdir), write_meta_graph=False)

        for treebank in treebanks:
            for dataset_name, dataset in [("dev", treebank.dev), ("test", treebank.test)]:
                predicted_basename = "{}/{}/{}-{}".format(args.logdir, treebank.code,
                                                          treebank.code if treebank.code else os.path.basename(args.basename), dataset_name)
                conllu = network.predict(dataset, args, probs_basename=predicted_basename)
                with open(predicted_basename + ".conllu", "w", encoding="utf-8") as predicted_file:
                    print(conllu, end="", file=predicted_file)
