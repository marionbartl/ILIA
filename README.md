# Mitigating bias with gender-inclusive terminology
This repository holds the code for the paper: Mitigating Bias in Language Models: Disrupting stereotypes with gender-inclusive terminology

There are several steps that need to be executed for replicating our paper:

## 0. Install requirements

```commandline
pip install -r requirements.txt
```

## 1. Download corpus

Use the `--no_tokens` argument to specifiy how many tokens should be downloaded. For example, the original Word2Vec model
was trained on 1.6 billion tokens. Specific values recognized are `1M`, `200M`, and `1.6B`, otherwise numeric values can be used.

```commandline
python code/dataset_download.py --no_tokens 100
```

Instead of downloading the random corpus from scratch, you can also use the data we downloaded contained in the `data` directory.

## 2. Re-write corpus

For re-writing our random corpus, which was filtered for long (>1000 words) sentences, run the following command:
```commandline
python code/rewriting_v2.py --corpus data/OWT-32-f
```
For adapting the code to your own corpus, see here:
```commandline
usage: rewriting_v2.py [-h] --corpus CORPUS [--log LOG]

Provide the path to corpus files that need to be rewritten.

options:
  -h, --help       show this help message and exit
  --corpus CORPUS  path to training corpus directory
  --log LOG        path to log file. If not provided, will create and write to logs directory.
```

## 3. Build word embedding models

Next, we built the Word2Vec models. This code needs to be run twice, once for the unchanged and once for the rewritten
corpus. The models will be saved in the `models/` directory.

### Unchanged model

```commandline
python code/make_embedding_model.py --corpus data/OWT-32M-f --train --bin --name w2v-OWT32
```

### Rewritten model
```commandline
python code/make_embedding_model.py --corpus data/OWT-32M-f-neutral+ --train --bin --name w2v-OWT32-neutral
```

Further information on the code arguments can be found here:
```commandline
usage: make_embedding_model.py [-h] [--corpus CORPUS] [--train] [--bin] [--name NAME]

Create an embedding model from a corpus of text files.

options:
  -h, --help       show this help message and exit
  --corpus CORPUS  path to training corpus file
  --train          train a new word2vec model
  --bin            whether or nor the w2v model will additionally be saved in binary format
  --name NAME      provide model name
```

## 4. Run evaluation
We provide a wide selection of embedding model evaluation methods. Using the respective arguments in the commandline
will run the respective tests on both the _normal_ (unchanged) and _neutral_ (rewritten) models.

For example, the following code will perform several Word Embedding Association Test (WEAT), as well as a qualitative analysis.

```commandline
python code/eval.py --weat --word_analysis --lower
```

Further evaluation test options are laid out below:

```commandline
usage: eval.py [-h] [--weat] [--cluster] [--svm] [--bat] [--ect] [--sim] [--word_analysis] [--norm] [--lower]

Modular choices for model evaluation

options:
  -h, --help       show this help message and exit
  --weat           run WEAT tests
  --cluster        run clustering tests
  --svm            run classification tests
  --bat            run bias analogy tests
  --ect            run embedding coherence test
  --sim            run similarity tests
  --word_analysis  performing most-similar-word analysis
  --norm           normalize vectors
  --lower          whether or not the model is lowercased
```