import argparse
import os
import re

import gensim
from gensim.models import Word2Vec

from utils import Timer

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for corpus_dir in os.listdir(self.dirname):
            for fname in os.listdir(self.dirname + '/' + corpus_dir):
                for line in open(os.path.join(self.dirname, corpus_dir, fname)):
                    # for sentence in doc.sentences:
                    #     yield [word.text.lower() for word in sentence.words]
                    yield [w.lower() for w in line.split()]
                    # yield line.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an embedding model from a corpus of text files.')
    parser.add_argument('--corpus', help='path to training corpus file')
    parser.add_argument('--train', action='store_true', help='train a new word2vec model')
    parser.add_argument('--name', help='provide model name')
    parser.add_argument('--dim', help='provide dimensionality of the word vectors', required=False, default=300)

    args = parser.parse_args()
    if args.train:

        # get the corpus size out of the file name
        size = re.findall(r'((\d{1,3}|\d\W\d)[MB]+)', args.corpus)[0][0]

        high_model_path = 'models/w2v/'

        match = re.match(r'\D*', args.corpus)
        result = match.group() if match else ''
        result = result.split('/')[-1]
        corpus_name = result.strip('_').strip('-')

        if not args.name:
            model_name = f'w2v-{corpus_name}_{str(size)}-{args.dim}d'
            specs = '-'.join(args.corpus.split('-')[1:])
            if specs != '':
                model_name += '-' + specs
            # if 'neutral' in args.corpus:
            #     model_name += '-neutral'
            # if 'openwebtext' in args.corpus:
            #     model_name += '-openwebtext'
        else:
            model_name = args.name
        
        print(f'{model_name} will be saved to {high_model_path}')

        model_path = os.path.join(high_model_path, model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        timer = Timer()
        timer.start()

        dim = int(args.dim)

        sentences = MySentences(args.corpus)  # a memory-friendly iterable
        print('corpus loaded!')
        print(timer.stop())

        model = gensim.models.Word2Vec(sentences=sentences, vector_size=dim, seed=42)

        print('model trained!')
        print(timer.stop())

        model.save(os.path.join(model_path,'model.model'))
        
        word_vectors = model.wv

        # save regular and binary format
        word_vectors.save_word2vec_format(os.path.join(model_path, 'vectors.txt'), binary=False)
        word_vectors.save_word2vec_format(os.path.join(model_path, 'vectors.bin'), binary=True)

        print('model saved!')

    elif not args.train:
        # check that everything works smoothly
        model = gensim.models.Word2Vec.load(args.corpus)
        vector = model.wv['computer']  # get numpy vector of a word
        sims = model.wv.most_similar('computer', topn=10)
        print(sims)
        print(model.wv.distance('man', 'flower'))

    else:
        print('sorry, I don\'t know what to do')
