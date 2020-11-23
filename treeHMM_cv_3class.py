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
parser.add_argument('--use_edge', type=bool, default=False, help='whether use link information as part of the features')
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

    nonbinding_train = []
    nonbinding_test = []
    for nonbinding_train_idx, nonbinding_test_idx in nonbinding_kf.split(nonbinding):
        nonbinding_train.append([nonbinding[i] for i in nonbinding_train_idx.tolist()])
        nonbinding_test.append([nonbinding[i] for i in nonbinding_test_idx.tolist()])

    return binding_train, binding_test, nonbinding_train, nonbinding_test


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
    binding_train, binding_test, nonbinding_train, nonbinding_test = n_fold(data, col_names, target_protein,
                                                                            num_folds=n_folds, seed=random_seed)
    for fold_iter, (bind_train, bind_test, nonbind_train, nonbind_test) in enumerate(zip(binding_train, binding_test,
                                                                                         nonbinding_train,
                                                                                         nonbinding_test)):
        logging.info('*' * 50)
        logging.info('Training and testing in {} folds'.format(fold_iter))
        # prepare glycans dictionary
        glycans_bind_train = {}
        glycans_nonbind_train = {}
        for i in range(len(bind_train)):
            iupac_text = bind_train[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_bind_train = Glycan(iupac)
            glycans_bind_train[i] = glycan_bind_train

        for i in range(len(nonbind_train)):
            iupac_text = nonbind_train[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_nonbind_train = Glycan(iupac)
            glycans_nonbind_train[i] = glycan_nonbind_train

        bind_adj_matrix, bind_mono_emission, bind_link_emission = create_forest_inputs(glycans_bind_train)
        nonbind_adj_matrix, nonbind_mono_emission, nonbind_link_emission = create_forest_inputs(glycans_nonbind_train)
        bind_parse_matrix = csr_matrix(bind_adj_matrix)
        nonbind_parse_matrix = csr_matrix(nonbind_adj_matrix)

        binding_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,
                                      random_init_emission_probabilities=True)
        nonbinding_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,
                                         random_init_emission_probabilities=True)

        if use_edge:
            bind_emission = [bind_mono_emission, bind_link_emission]
            nonbind_emission = [nonbind_mono_emission, nonbind_link_emission]
        else:
            bind_emission = [bind_mono_emission]
            nonbind_emission = [nonbind_mono_emission]

        bind_param = baumWelch.hmm_train_and_test(copy.deepcopy(binding_hmm), bind_parse_matrix, bind_emission,
                                                  maxIterations=max_iter, delta=delta)
        nonbind_param = baumWelch.hmm_train_and_test(copy.deepcopy(nonbinding_hmm), nonbind_parse_matrix, nonbind_emission,
                                                     maxIterations=max_iter, delta=delta)

        # Finished training, reinitialized models with trained params
        binding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=bind_param['hmm']['state_transition_probabilities'],
                                              emission_probabilities=bind_param['hmm']['emission_probabilities'])
        nonbinding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=nonbind_param['hmm']['state_transition_probabilities'],
                                                 emission_probabilities=nonbind_param['hmm']['emission_probabilities'])

        pickle.dump({'bind_mode_{}'.format(fold_iter): binding_hmm_trained}, pickle_file)
        pickle.dump({'nonbind_mode_{}'.format(fold_iter): nonbinding_hmm_trained}, pickle_file)

        testset = bind_test + nonbind_test
        testlabel = [1] * len(bind_test) + [0] * len(nonbind_test)
        testpred = []
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

            bind_fwd_probs = forward.forward(binding_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)
            nonbind_fwd_probs = forward.forward(nonbinding_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)

            if logsumexp(bind_fwd_probs.iloc[:, -1]) >= logsumexp(nonbind_fwd_probs.iloc[:, -1]):
                testpred.append(1)
            else:
                testpred.append(0)

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
