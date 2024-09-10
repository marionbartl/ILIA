import argparse
import logging
import operator
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

sys.path.append('.')
from external_libs.XWEAT import weat
from external_libs.DEBIE import eval as evil
from numpy import linalg

from sklearn.metrics.pairwise import cosine_similarity

# reproducibility ---------------- # from Neutral Rewriter
from random import seed
from numpy.random import seed as np_seed
import os

seed(42)
np_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# -----------------------------------


def load_from_gensim(filename):
    wv_from_bin = KeyedVectors.load_word2vec_format(filename, binary=True)  # C bin format
    vocabulary = wv_from_bin.key_to_index
    vecs = np.array([wv_from_bin[w] for w in vocabulary])

    embd_dict = {}
    for term, index in vocabulary.items():
        embd_dict[term] = wv_from_bin[index]

    assert len(embd_dict) == len(vocabulary)

    return embd_dict, vocabulary, vecs


def read_targets(file, lower=False):
    targets = {'T1': [], 'T2': [], 'A1': [], 'A2': []}
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            if line[0][:-1] in targets:
                targets[line[0][:-1]] = line[1:]
        if lower:
            targets = {k: [l.lower() for l in v] for k, v in targets.items()}
    return targets['T1'], targets['T2'], targets['A1'], targets['A2']


def compute_bias_by_projection(vecs, vocab, male_vec, female_vec):
    """ adapted from https://github.com/gonenhila/gender_bias_lipstick """
    # computing the dot products (similarity, effectively)
    # between the vector space and the vector for man/woman
    # --> get a list (numpy array) of dot products

    # males = vecs.dot(male_vec)
    # females = vecs.dot(female_vec)

    # calculate cosine similarity (dot product but normalized) to 'male vector'
    p1_m = vecs.dot(male_vec)
    p2 = linalg.norm(vecs, axis=1) * linalg.norm(male_vec)
    males = p1_m / p2

    # calculate cosine similarity (dot product but normalized) to 'female vector'
    p1_f = vecs.dot(female_vec)
    females = p1_f / p2

    distance_dict = {}
    for w, m, f in zip(vocab, males, females):
        # compute distance between male and female 'bias'
        # aka the last step in calculating the 'projection on the bias direction'
        distance_dict[w] = m - f

    return distance_dict


def normalize(word_vectors):
    """ from https://github.com/gonenhila/gender_bias_lipstick/source/remaining_bias_2016.ipynb"""
    # normalize vectors
    norms = np.apply_along_axis(linalg.norm, 1, word_vectors)
    word_vectors = word_vectors / norms[:, np.newaxis]
    return word_vectors


def most_biased_words(m_words, f_words, vocab_m, vecs_m, n=200, v=True):
    """get n most biased words from model """
    assert len(m_words) == len(f_words)
    if len(m_words) == 1:
        print('only 1 seed word each')
        m_vec = vecs_m[vocab_m[m_words[0]]]
        f_vec = vecs_m[vocab_m[f_words[0]]]
    else:
        m_vec = np.mean([vecs_m[vocab_m[tok]] for tok in m_words], axis=0)
        f_vec = np.mean([vecs_m[vocab_m[tok]] for tok in f_words], axis=0)
    assert len(m_vec) == len(f_vec) == 300

    dist_dict = compute_bias_by_projection(vecs=vecs_m,
                                           vocab=vocab_m,
                                           male_vec=m_vec,
                                           female_vec=f_vec)

    dist_df = pd.Series(dist_dict)

    sorted_g = sorted(dist_dict.items(), key=operator.itemgetter(1))
    most_fem = [item[0] for item in sorted_g[:n] if item not in f_words]  # aka female words
    sorted_g_rev = sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True)
    most_masc = [item[0] for item in sorted_g_rev[:n] if item not in m_words]  # aka male words
    top = 25
    if v:
        print(f'{top} words most "female-biased" (t2):', most_fem[:top])
        print(f'{top} words most "male-biased" (t1):', most_masc[:top])

    return most_masc, most_fem, dist_df


def choose_and_remove(items, voc):
    """from https://stackoverflow.com/questions/3791400/how-can-you-select-a-random-element-from-a-list-and-have-it
    -be-removed"""
    # pick an item index
    if items:
        while True:
            index = random.randrange(len(items))
            if items[index] in voc:
                return items.pop(index)
    # nothing left!
    return None


def get_freq_dist(corpus_path):
    freq_dict = defaultdict(int)

    file_count = 0

    for root, dirs, files in os.walk(corpus_path):
        print(root, dirs, '# files', len(files))

        if len(files) > 0:
            file_count += len(files)
            for i in tqdm(range(len(files))):
                with open(os.path.join(root, files[i]), 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        for word in line:
                            try:
                                freq_dict[word.lower()] += 1
                            except:
                                print(f'{word} caused trouble')

    # print(f'# tokens:\t{len(freq_dict)}')
    # calculate the rank of each word
    sorted_f_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    rank_dict = defaultdict(int)
    for idx, (word, count) in enumerate(sorted_f_dict):
        rank_dict[word] = idx

    freq_df = pd.DataFrame(pd.Series(freq_dict), columns=['freq'])
    # add ranks to the df. This works bc. the Series has the same keys as the df.
    freq_df['rank'] = pd.Series(rank_dict)
    # print(freq_df.head(5))

    return freq_df


def weat_test(w2v_dict, test_numbers, lower=True):
    logging.info("WEAT started")
    test = weat.XWEAT()

    permutation_number = 1000000
    similarity_type = "cosine"

    for test_number in test_numbers:
        print('WEAT', test_number)

        if test_number == 1:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_1()
        elif test_number == 2:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_2()
        elif test_number == 3:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_3()
        elif test_number == 4:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_4()
        elif test_number == 5:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_5()
        elif test_number == 6:
            targets_1, targets_2, _, _ = test.weat_6()
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        elif test_number == 7:
            targets_1, targets_2, _, _ = test.weat_7()
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        elif test_number == 8:
            targets_1, targets_2, _, _ = test.weat_8()
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        elif test_number == 9:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_9()
        elif test_number == 10:
            targets_1, targets_2, attributes_1, attributes_2 = test.weat_10()
        elif test_number == 'a':
            targets_1, targets_2 = male_profs, female_profs
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        elif test_number == 'b':
            targets_1, targets_2 = computer_science, childcare
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        elif test_number == 'c':
            targets_1, targets_2 = military_history, sex
            attributes_1, attributes_2 = attrs_1_aug, attrs_2_aug
        else:
            raise ValueError("Only WEAT 1 to 10 are supported")

        if lower:
            targets_1 = [t.lower() for t in targets_1]
            targets_2 = [t.lower() for t in targets_2]
            attributes_1 = [a.lower() for a in attributes_1]
            attributes_2 = [a.lower() for a in attributes_2]

        for name, d in w2v_dict.items():
            print('---'+name+'---')
            test.set_embd_dict(w2v_dict[name]['d'])

            logging.info("Embeddings loaded")
            logging.info("Running test")
            result = test.run_test_precomputed_sims(targets_1, targets_2, attributes_1, attributes_2,
                                                    permutation_number, similarity_type)
            logging.info(result)
            print()


def get_embeddings(**models):
    w2v_dict = {}

    for name, model in models.items():
        if model:
            if os.path.isdir(model):
                bin_model = os.path.join(model, 'vectors.bin')
            else:
                bin_model = model

            emb_dict, vocab, vectors = load_from_gensim(bin_model)
            w2v_dict[name] = {'d': emb_dict,
                              'vectors': vectors,
                              'vocab': vocab}
    
    if not w2v_dict:
        raise ValueError('No embeddings were loaded, because no path to model was provided')

    return w2v_dict  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modular choices for model evaluation')
    parser.add_argument('--model', help='model to be evaluated', required=False, type=str)
    parser.add_argument('--neutral_model', help='neutral model to be evaluated', required=False, type=str)
    parser.add_argument('--hd_model', help='hard debiased model to be evaluated', required=False, type=str)
    parser.add_argument('--weat', help='run WEAT tests', action='store_true', default=False)
    parser.add_argument('--cluster', help='run clustering tests', action='store_true', default=False)
    parser.add_argument('--svm', help='run classification tests', action='store_true', default=False)
    parser.add_argument('--bat', help='run bias analogy tests', action='store_true', default=False)
    parser.add_argument('--ect', help='run embedding coherence test', action='store_true',
                        default=False)
    parser.add_argument('--sim', help='run similarity tests', action='store_true', default=False)
    parser.add_argument('--word_analysis', help='performing most-similar-word analysis', action='store_true',
                        default=False)
    # parser.add_argument('--norm', help='normalize vectors', action='store_true', default=False)
    parser.add_argument('--lower', help='whether or not the model is lowercased', action='store_true', default=True)
    args = parser.parse_args()

    ###### LOAD DATA ######

    w2v_dict = get_embeddings(normal=args.model, neutral=args.neutral_model, hard_debiased=args.hd_model)
    print('Embeddings loaded!')

    ###### TARGETS & ATTRIBUTES ######
    computer_science = ["firmware", "gui", "programmer", "hardware", "notebook", "database", "router", "pc"]
    childcare = ["children", "babysitter", "daycare", "homemaker", "newborn", "baby", "toddler", "parenting"]
    military_history = ["stalin", "mussolini", "dictator", "fascist", "lieutenant", "overthrow", "autocrat", "enforce"]
    sex = ["sex", "sexual", "nude", "cam", "aroused", "consensual", "prostitute", "porn"]
    # professions from Manzini et al. 2019
    male_profs = ["manager", "executive", "doctor", "lawyer", "programmer", "scientist", "soldier",
                  "supervisor", "rancher", "janitor", "firefighter", "officer"]
    female_profs = ["secretary", "nurse", "clerk", "artist", "homemaker", "dancer", "singer", "librarian",
                    "maid", "hairdresser", "stylist", "receptionist", "counselor"]

    # get gender attribute words
    eval_file = 'external_libs/DEBIE/data/weat_8_no_pro.txt'
    targets_1, targets_2, attributes_1, attributes_2 = read_targets(eval_file, args.lower)

    print("targets 1:", targets_1)
    print("targets 2:", targets_2)
    print("attributes 1:", attributes_1)
    print("attributes 2:", attributes_2)

    attrs_1_aug = attributes_1 + ["boy", "man", "male"]
    attrs_2_aug = attributes_2 + ["girl", "woman", "female"]

    print("augmented A1:", attrs_1_aug)
    print("augmented A2:", attrs_2_aug)

    # make sure info logs are displayed in console
    logging.basicConfig(level=logging.INFO)

    ###### WEAT EVALUATION ######
    if args.weat:
        # arguments normally given before
        # test_numbers = [6, 7, 8]
        test_numbers = [9,10]
        # test_numbers = [6,7,8, 'a', 'b', 'c']
        # test_numbers = ['a', 'b']
        # test_numbers = [no for no in list(range(1, 11)) if no not in test_numbers]
        weat_test(w2v_dict, test_numbers, lower=args.lower)

    # # 1 # #
    # seed_words_m = attributes_1 + ['boy', 'man']
    # seed_words_f = attributes_2 + ['girl', 'woman']

    seed_words_m = attrs_1_aug
    seed_words_f = attrs_2_aug

    # # # # 2 # #
    # seed_words_m = ['he']
    # seed_words_f = ['she']

    # # # 3 # #
    # seed_words_m = ['man']
    # seed_words_f = ['woman']

    # # # 4 # #
    # seed_words_m = targets_1
    # seed_words_f = targets_2

    n = 500
    # get the 2500 words closest to elementwise mean of the targets
    targets_1_long, targets_2_long, bias_df = most_biased_words(
        seed_words_m,
        seed_words_f,
        w2v_dict["normal"]["vocab"],
        w2v_dict["normal"]["vectors"],
        n=n,
    )

    print(f"len of augm. T1: {len(targets_1_long)}")
    print(f"len of augm. T2: {len(targets_2_long)}")

    # get all the augmented words from Lauscher et al. (2020)
    aug_file_prefix = 'external_libs/DEBIE/data/weat_8_aug_postspec_'
    aug_t1 = []
    aug_t2 = []

    for i in range(2, 6):
        t1_aug, t2_aug, _, _ = read_targets(aug_file_prefix + str(i) + '.txt', args.lower)
        for t in t1_aug:
            if t not in aug_t1:
                aug_t1.append(t)
        for t in t2_aug:
            if t not in aug_t2:
                aug_t2.append(t)

    print(f'len of aug T1 (Lauscher): {len(aug_t1)} (e.g. {", ".join(random.sample(aug_t1, 3))})')
    print(f'len of aug T2 (Lauscher): {len(aug_t2)} (e.g. {", ".join(random.sample(aug_t2, 3))})')

    if args.word_analysis:
        # freq_df_n = get_freq_dist('data/OWT-32M-f-neutral+')
        # freq_df = get_freq_dist('data/OWT-32M-f')

        vocab_n = w2v_dict["neutral"]["vocab"]

        _, _, bias_df_n = most_biased_words(seed_words_m, seed_words_f,
                                            w2v_dict['neutral']['vocab'],
                                            w2v_dict['neutral']['vectors'], n=n, v=True)
        exit()

        # get the bias score that is dependent on the man/woman binary
        bias_df = pd.DataFrame(bias_df)
        bias_df = bias_df.reset_index()
        bias_df.columns = ['word', 'bias_before']  # rename columns
        bias_df = bias_df.set_index('word')  # use word column as index again
        print(bias_df.head())

        # create some empty columns for word frequency
        bias_df['bias_after'] = np.nan
        bias_df['freq_before'] = np.nan
        bias_df['freq_after'] = np.nan
        bias_df['rank_before'] = np.nan
        bias_df['rank_after'] = np.nan

        # populate the empty frequency+rank columns
        for idx in bias_df.index:
            bias_df.loc[idx, 'freq_before'] = freq_df.loc[idx, 'freq']
            bias_df.loc[idx, 'rank_before'] = freq_df.loc[idx, 'rank']
            if idx in vocab_n.keys():
                bias_df.loc[idx, 'freq_after'] = freq_df_n.loc[idx, 'freq']
                bias_df.loc[idx, 'rank_after'] = freq_df_n.loc[idx, 'rank']

        # populate the empty bias column
        for idx, val in bias_df_n.items():
            bias_df.loc[idx, 'bias_after'] = val

        # negative values: shift to female; positive values: shift to male
        bias_df['bias_dif'] = bias_df['bias_after'] - bias_df['bias_before']

        cols = list(bias_df.columns)
        new_order = cols[0:2] + [cols[6]] + cols[2:6]
        bias_df = bias_df[new_order]

        print(bias_df.head())

        bias_df = bias_df.convert_dtypes()
        top = 20

        print('Most Female Before')
        print(bias_df.sort_values('bias_before', ascending=True).head(top).to_csv())
        print('Most Male Before')
        print(bias_df.sort_values('bias_before', ascending=False).head(top).to_csv())

        print('Most Female After')
        print(bias_df.sort_values('bias_after', ascending=True).head(top).to_csv())
        print('Most Male After')
        print(bias_df.sort_values('bias_after', ascending=False).head(top).to_csv())

    if args.cluster:
        print('CLUSTERING')

        for name, d in w2v_dict.items():
            print(f'\n---{name}---')
            clustering_score = evil.eval_k_means(targets_1_long, targets_2_long,
                                                 vecs=w2v_dict[name]['vectors'],
                                                 vocab=w2v_dict[name]['vocab'])
            logging.info(f'clustering_score:{round(clustering_score,2)}')

    if args.bat:
        print('Bias Analogy Test'.upper())

        for name, d in w2v_dict.items():
            print('\n---'+name+'---')
            print('Science vs. Art')
            print(f'targets 1: {targets_1}')
            print(f'targets 2: {targets_2}')

            bat_score = evil.bias_analogy_test(vecs=w2v_dict[name]['vectors'],
                                               vocab=w2v_dict[name]['vocab'],
                                               target_1=targets_1, target_2=targets_2,
                                               attributes_1=attrs_1_aug, attributes_2=attrs_2_aug)
            logging.info(f'bias analogy test:{bat_score}')

            print('Professions')
            print(f'targets 1: {male_profs}')
            print(f'targets 2: {female_profs}')
            bat_score_profs = evil.bias_analogy_test(vecs=w2v_dict[name]['vectors'],
                                                     vocab=w2v_dict[name]['vocab'],
                                                     target_1=male_profs, target_2=female_profs,
                                                     attributes_1=attrs_1_aug, attributes_2=attrs_2_aug)
            logging.info(f'bias analogy test:{bat_score_profs}')

            print('CompSci vs. CareWork')
            print(f'targets 1: {computer_science}')
            print(f'targets 2: {childcare}')
            bat_score_b = evil.bias_analogy_test(vecs=w2v_dict[name]['vectors'],
                                                 vocab=w2v_dict[name]['vocab'],
                                                 target_1=computer_science, target_2=childcare,
                                                 attributes_1=attrs_1_aug, attributes_2=attrs_2_aug)
            logging.info(f'bias analogy test:{bat_score_b}')

    if args.svm:
        # prepare training and test data

        # training
        t1_long = targets_1_long.copy()
        t2_long = targets_2_long.copy()

        # test
        # t1 = attrs_1_aug
        # t2 = attrs_2_aug

        # take a subset of the augmented targets as the test set
        t1 = []
        t2 = []

        vocab_n = w2v_dict['neutral']['vocab']
        print(f'vocab size: {len(vocab_n)}')

        test_size = int(n*0.1)

        for i in range(test_size):
            elem = choose_and_remove(t1_long, vocab_n.keys())
            t1.append(elem)

            elem = choose_and_remove(t2_long, vocab_n.keys())
            t2.append(elem)

        # test_words = [word for word in subset_t1 + subset_t2]
        test_words = [word for word in t1 + t2]
        print(f'# test words: {len(t1)} + {len(t2)} = {len(test_words)}')

        # make sure there is no overlap between test and train data
        t1_long = [w for w in t1_long if w not in test_words and w in vocab_n.keys()]
        t2_long = [w for w in t2_long if w not in test_words and w in vocab_n.keys()]
        train_words = [word for word in t1_long + t2_long]
        print(f'# train words before: {len(t1_long)} + {len(t2_long)} = {len(train_words)}')
        # print('overlap of train & test words:', set(train_words).intersection(set(test_words)))
        assert not set(train_words).intersection(set(test_words))

        # get everything indexed
        vocab_train = {word: i for i, word in enumerate(train_words)}
        vocab_test = {word: i for i, word in enumerate(test_words)}

        # run SVM experiments for the embedding spaces before & after
        for name, d in w2v_dict.items():
            print('\n---'+name+'---')
            vecs_svm = w2v_dict[name]['vectors']
            vocab_svm = w2v_dict[name]['vocab']

            # retrieve the relevant embedding vectors
            train_vecs = np.array([vecs_svm[vocab_svm[word]] for word in train_words])
            test_vecs = np.array([vecs_svm[vocab_svm[word]] for word in test_words])

            # check that the right embeddings were retrieved
            assert np.array_equal(train_vecs[1], vecs_svm[vocab_svm[train_words[1]]])
            assert np.array_equal(test_vecs[1], vecs_svm[vocab_svm[test_words[1]]])

            svm_score = evil.eval_svm(train_first=t1_long, train_second=t2_long,
                                      test_first=t1, test_second=t2,
                                      vocab_train=vocab_train,
                                      vocab_test=vocab_test,
                                      vecs_train=train_vecs,
                                      vecs_test=test_vecs)

            logging.info(f'SVM classification accuracy:{round(svm_score,3)}')

    if args.ect:
        print('EMBEDDING COHERENCE TEST')

        for name, d in w2v_dict.items():
            print('\n---'+name+'---')
            print('Science vs. Arts')
            vecs_ect = w2v_dict[name]['vectors']
            vocab_ect = w2v_dict[name]['vocab']

            ect_score = evil.embedding_coherence_test(vecs_ect, vocab_ect, targets_1, targets_2,
                                                      attrs_1_aug + attrs_2_aug)
            logging.info(f'ECT: {ect_score}')

            print('Professions')
            ect_score = evil.embedding_coherence_test(vecs_ect, vocab_ect, male_profs, female_profs,
                                                      attrs_1_aug + attrs_2_aug)
            logging.info(f'ECT: {ect_score}')

            print('CompSci vs. Childcare')
            ect_score = evil.embedding_coherence_test(vecs_ect, vocab_ect, computer_science, childcare,
                                                      attrs_1_aug + attrs_2_aug)
            logging.info(f'ECT: {ect_score}')

    if args.sim:
        simlex = pd.read_csv('external_libs/SimLex-999/SimLex-999.txt', sep='\t')
        simlex = simlex[['word1', 'word2', 'SimLex999']]

        word_sim = pd.read_csv('external_libs/ws353simrel/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt',
                               sep='\t')

        for name, d in w2v_dict.items():
            print('\n---'+name+'---')
            simlex_score = evil.eval_simlex(simlex.values.tolist(),
                                            w2v_dict[name]['vocab'],
                                            w2v_dict[name]['vectors'])
            logging.info(f'SimLex999: pearson: {simlex_score[0]} spearman {simlex_score[1]}')

            simlex_score = evil.eval_simlex(word_sim.values.tolist(),
                                            w2v_dict[name]['vocab'],
                                            w2v_dict[name]['vectors'])
            logging.info(f'WordSim353: pearson: {simlex_score[0]} spearman {simlex_score[1]}')
