# UFAL-Prague Sigmorphon 2019 Task 2 Participant System

This is the participant system of UFAL-Prague for Sigmorphon 2019 Shared Task 2.

The main sources are
- `src/um_tagger.py`: training on a single treebank
- `src/um_tagger_multi.py`: training on multiple treebanks

To train, you will need word embeddings:
- fastText `.bin` embeddings

  In addition to `.bin` file, there should also be a `.bin.casing` file, which
  contains either `uncased` or `cased` content indicating if a word should be
  lowercased before looking up the embeddings

- BERT embeddings

  The BERT embeddings must be in `.npz` format and should contain an array of
  nparrays, each corresponding to a sentence from the data, containing
  embeddings for its words. There should be one file for each train/dev/test
  set.

  Such embeddings can be generated from CoNLL-U files using
  `src/generate_embeddings/conllu_bert_embeddings.py` script.
