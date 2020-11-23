import numpy as np
import copy
import re
import logging
import sys
import argparse
import pickle
import time
import pandas as pd
from sklearn.model_selection import KFold
from treehmm4glycan import create_forest_inputs, get_iupcas
from TreeHMM4Glycan.Glycan import Glycan
from treehmm import initHMM, baumWelch, forward, fwd_seq_gen
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


# Global variables for hyper params selection and logging
parser = argparse.ArgumentParser('Glycan TreeHMM')
parser.add_argument('--use_edge', type=bool, default=True, help='whether use link information as part of the features')
parser.add_argument('--n_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--max_iter', type=int, default=1, help='maximum number of epochs for BW to train')
parser.add_argument('--n_states', type=int, default=5, help='number of hidden states')
parser.add_argument('--delta', type=float, default=1e-5, help='stop training when difference is less than delta')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(args.save + '-log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def get_data():
    """Get IUPAC, and MScore.
    Return, zipped data, protein names
    """
    iupac_data = pd.read_csv('./Data/IUPAC.csv')
    iupacs = iupac_data['IUPAC'].tolist()
    mscore_data = pd.read_csv('./Data/MScore_useful.csv')
    mscore = mscore_data.drop('ID', axis=1)
    col_names = mscore.columns.tolist()
    mscore = mscore.values.tolist()
    data = list(zip(iupacs, mscore))
    return data, col_names


def n_fold(data, col_names, protein, num_folds, seed=None):
    """10-fold cross validation with random sampling."""
    assert len(data) == 600
    iupac, mscore = list(zip(*data))
    protein_idx = col_names.index(protein)
    strong_binding = []
    mid_binding = []
    weak_binding = []
    for i in range(600):
        if mscore[i][protein_idx] >= 3.5:
            strong_binding.append(iupac[i])
        elif 1.5 <= mscore[i][protein_idx] <3.5:
            mid_binding.append(iupac[i])
        else:
            weak_binding.append(iupac[i])
    strong_binding_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    mid_binding_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    weak_binding_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    strong_binding_train = []
    strong_binding_test = []
    for strong_binding_train_idx, strong_binding_test_idx, in strong_binding_kf.split(strong_binding):
        strong_binding_train.append([strong_binding[i] for i in strong_binding_train_idx.tolist()])
        strong_binding_test.append([strong_binding[i] for i in strong_binding_test_idx.tolist()])

    mid_binding_train = []
    mid_binding_test = []
    for mid_binding_train_idx, mid_binding_test_idx, in mid_binding_kf.split(mid_binding):
        mid_binding_train.append([mid_binding[i] for i in mid_binding_train_idx.tolist()])
        mid_binding_test.append([mid_binding[i] for i in mid_binding_test_idx.tolist()])

    weak_binding_train = []
    weak_binding_test = []
    for weak_binding_train_idx, weak_binding_test_idx in weak_binding_kf.split(weak_binding):
        weak_binding_train.append([weak_binding[i] for i in weak_binding_train_idx.tolist()])
        weak_binding_test.append([weak_binding[i] for i in weak_binding_test_idx.tolist()])

    return strong_binding_train, strong_binding_test, mid_binding_train, mid_binding_test, weak_binding_train, weak_binding_test


def train_and_test(use_edge=False, n_folds=10, n_states=5, max_iter=50, delta=1e-5, random_seed=None):
    """Train GlyNet using n-fold cross-validation.
    """
    print('Training with hyper parameters:')
    print('use edge: {}, n_folds: {}, n_states: {}, max_iter: {}, delta: {}, random seed: {}'.
          format(use_edge, n_folds, n_states, max_iter, delta, random_seed))

    pickle_file = open(args.save + '-model.pkl', 'wb')

    cv_label = []
    cv_pred = []
    cv_iupac = []

    # Get total possible number of emissions
    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
    glycans = {}

    monos = []
    links = []
    for id in iupacs:
        iupac_text = iupacs[id]
        glycan = Glycan(iupac_text)
        if glycan.get_num_nosaccharides() > 1:
            glycans[id] = glycan
            mono = glycans[id].get_monosaccharide_emssions()
            link = glycans[id].get_linkage_emssions()
            monos += mono
            links += link

    mono_emissions = list(set(monos))
    link_emissions = list(set(links))
    if use_edge:
        emissions = [mono_emissions, link_emissions]
    else:
        emissions = [mono_emissions]
    states = [str(i) for i in range(1, n_states + 1)]

    # Prepare data
    data, col_names = get_data()
    target_protein = 'AAL (100 ug/ml)'
    strong_bind_train, strong_bind_test, mid_bind_train, mid_bind_test, weak_bind_train, weak_bind_test = \
        n_fold(data, col_names, target_protein, num_folds=n_folds, seed=random_seed)
    for fold_iter, (strong_train, strong_test, mid_train, mid_test, weak_train, weak_test) in enumerate(
            zip(strong_bind_train, strong_bind_test, mid_bind_train, mid_bind_test, weak_bind_train, weak_bind_test)):
        logging.info('*' * 50)
        logging.info('Training and testing in {} folds'.format(fold_iter))
        # prepare glycans dictionary
        glycans_strong_train = {}
        glycans_mid_train = {}
        glycans_weak_train = {}
        for i in range(len(strong_train)):
            iupac_text = strong_train[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_strong_train = Glycan(iupac)
            glycans_strong_train[i] = glycan_strong_train

        for i in range(len(mid_train)):
            iupac_text = mid_train[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_mid_train = Glycan(iupac)
            glycans_mid_train[i] = glycan_mid_train

        for i in range(len(weak_train)):
            iupac_text = weak_train[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_weak_train = Glycan(iupac)
            glycans_weak_train[i] = glycan_weak_train

        strong_adj_matrix, strong_mono_emission, strong_link_emission = create_forest_inputs(glycans_strong_train)
        mid_adj_matrix, mid_mono_emission, mid_link_emission = create_forest_inputs(glycans_mid_train)
        weak_adj_matrix, weak_mono_emission, weak_link_emission = create_forest_inputs(glycans_weak_train)
        strong_parse_matrix = csr_matrix(strong_adj_matrix)
        mid_parse_matrix = csr_matrix(mid_adj_matrix)
        weak_parse_matrix = csr_matrix(weak_adj_matrix)

        strong_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,
                                     random_init_emission_probabilities=True)
        mid_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,
                                  random_init_emission_probabilities=True)
        weak_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,
                                   random_init_emission_probabilities=True)

        if use_edge:
            strong_emission = [strong_mono_emission, strong_link_emission]
            mid_emission = [mid_mono_emission, mid_link_emission]
            weak_emission = [weak_mono_emission, weak_link_emission]
        else:
            strong_emission = [strong_mono_emission]
            mid_emission = [mid_mono_emission]
            weak_emission = [weak_mono_emission]

        strong_param = baumWelch.hmm_train_and_test(copy.deepcopy(strong_hmm), strong_parse_matrix, strong_emission,
                                                    maxIterations=max_iter, delta=delta)
        mid_param = baumWelch.hmm_train_and_test(copy.deepcopy(mid_hmm), mid_parse_matrix, mid_emission,
                                                 maxIterations=max_iter, delta=delta)
        weak_param = baumWelch.hmm_train_and_test(copy.deepcopy(weak_hmm), weak_parse_matrix, weak_emission,
                                                  maxIterations=max_iter, delta=delta)

        # Finished training, reinitialized models with trained params
        strong_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=strong_param['hmm']['state_transition_probabilities'],
                                             emission_probabilities=strong_param['hmm']['emission_probabilities'])
        mid_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=mid_param['hmm']['state_transition_probabilities'],
                                          emission_probabilities=mid_param['hmm']['emission_probabilities'])
        weak_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=weak_param['hmm']['state_transition_probabilities'],
                                           emission_probabilities=weak_param['hmm']['emission_probabilities'])

        pickle.dump({'strong_model_{}'.format(fold_iter): strong_hmm_trained}, pickle_file)
        pickle.dump({'mid_model_{}'.format(fold_iter): mid_hmm_trained}, pickle_file)
        pickle.dump({'weak_model_{}'.format(fold_iter): weak_hmm_trained}, pickle_file)

        testset = weak_test + mid_test + strong_test
        testlabel = [0] * len(weak_test) + [1] * len(mid_test) + [2] * len(strong_test)
        testpred = []
        likelihood_list = []
        for i in range(len(testset)):
            iupac_text = testset[i]
            cv_iupac.append(iupac_text)
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_test = Glycan(iupac)

            # get test data features
            glycan_adj_matrix = glycan_test.get_adj_matrix()
            glycan_sparse_matrix = csr_matrix(glycan_adj_matrix)
            glycan_mono_emission = glycan_test.get_monosaccharide_emssions()
            glycan_link_emission = glycan_test.get_linkage_emssions()

            if use_edge:
                glycan_emission = [glycan_mono_emission, glycan_link_emission]
            else:
                glycan_emission = [glycan_mono_emission]

            # calculate the likelihood
            fwd_tree_sequence = fwd_seq_gen.forward_sequence_generator(glycan_sparse_matrix)

            strong_fwd_probs = forward.forward(strong_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)
            mid_fwd_probs = forward.forward(mid_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)
            weak_fwd_probs = forward.forward(weak_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)

            likelihood_list.append([logsumexp(weak_fwd_probs.iloc[:, -1]), logsumexp(mid_fwd_probs.iloc[:, -1]), logsumexp(strong_fwd_probs.iloc[:, -1])])

        ll_array = np.array(likelihood_list)
        testpred = ll_array.argmax(axis=1).tolist()
        logging.info(classification_report(testlabel, testpred))
        cv_label += testlabel
        cv_pred += testpred
    pickle.dump({'y_label': cv_label, 'y_pred': cv_pred, 'y_iupac': cv_iupac}, pickle_file)
    pickle_file.close()
    logging.info('*' * 50)
    logging.info('Overall performance')
    logging.info(classification_report(cv_label, cv_pred))


if __name__ == '__main__':
    logging.info('args = %s', args)
    train_and_test(use_edge=args.use_edge, n_folds=args.n_folds, max_iter=args.max_iter, n_states=args.n_states,
                   delta=args.delta, random_seed=args.seed)
