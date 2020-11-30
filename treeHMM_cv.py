import math
import re
import logging
import sys
import argparse
import pickle
import time
import os
import pandas as pd
from sklearn.model_selection import KFold
from treehmm4glycan import create_forest_inputs, get_iupcas, get_glycans
from TreeHMM4Glycan.Glycan import Glycan
from treehmm import initHMM, baumWelch, forward, fwd_seq_gen
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


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
    binding = []
    nonbinding = []
    for i in range(600):
        if mscore[i][protein_idx] >= 2:
            binding.append(iupac[i])
        else:
            nonbinding.append(iupac[i])
    binding_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    nonbinding_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    binding_train = []
    binding_test = []
    for binding_train_idx, binding_test_idx, in binding_kf.split(binding):
        binding_train.append([binding[i] for i in binding_train_idx.tolist()])
        binding_test.append([binding[i] for i in binding_test_idx.tolist()])

    nonbinding_train = []
    nonbinding_test = []
    for nonbinding_train_idx, nonbinding_test_idx in nonbinding_kf.split(nonbinding):
        nonbinding_train.append([nonbinding[i]
                                 for i in nonbinding_train_idx.tolist()])
        nonbinding_test.append([nonbinding[i]
                                for i in nonbinding_test_idx.tolist()])

    return binding_train, binding_test, nonbinding_train, nonbinding_test


def prepare_data(training_data):
    glycans_train = {}

    for i in range(len(training_data)):
        iupac_text = training_data[i]
        iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
        glycans_train[i] = Glycan(iupac)

    adj_matrix, mono_emissions, link_emissions = create_forest_inputs(
        glycans_train)
    parse_matrix_adj_matrix = csr_matrix(adj_matrix)
    return parse_matrix_adj_matrix, mono_emissions, link_emissions
    #nonbinding_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=True,random_init_emission_probabilities=True)


def train_and_test(use_edge=False, n_folds=10, n_states=5, max_iter_1=50, max_iter_2=None, delta=1e-5,
                   random_seed=None):
    """Train GlyNet using n-fold cross-validation.
    """
    print('Training with hyper parameters:')
    print('use edge: {}, n_folds: {}, n_states: {}, max_iter_pos: {}, max_iter_neg: {}, delta: {}, random seed: {}'.
          format(use_edge, n_folds, n_states, max_iter_1, max_iter_2, delta, random_seed))

    if max_iter_2 is None:
        max_iter_2 = max_iter_1

    pickle_file = open(args.save + '-model.pkl', 'wb')

    dict_to_save = {}
    cv_label = []
    cv_pred = []
    cv_pred_posterior = []
    cv_iupac = []

    # Get total possible number of emissions
    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
    _, mono_emissions, link_emissions = get_glycans(iupacs)

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
        logging.info('Training and testing in fold #{}'.format(fold_iter))
        # compute by chance prob for each class
        by_chance_bind_prob = len(bind_train) / \
            (len(bind_train) + len(nonbind_train))
        logging.info('By Chance Bind Prob: {:.3f}'.format(by_chance_bind_prob))
        log_by_chance_bind_prob = math.log(by_chance_bind_prob)
        log_by_chance_nobind_prob = math.log(1 - by_chance_bind_prob)
        # prepare glycans dictionary
        bind_parse_matrix, bind_mono_emission, bind_link_emission = prepare_data(
            bind_train)
        nonbind_parse_matrix, nonbind_mono_emission, nonbind_link_emission = prepare_data(
            nonbind_train)

        binding_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=False,
                                      random_init_emission_probabilities=False)
        nonbinding_hmm = initHMM.initHMM(states, emissions, random_init_state_transition_probabilities=False,
                                         random_init_emission_probabilities=False)

        if use_edge:
            bind_emission = [bind_mono_emission, bind_link_emission]
            nonbind_emission = [nonbind_mono_emission, nonbind_link_emission]
        else:
            bind_emission = [bind_mono_emission]
            nonbind_emission = [nonbind_mono_emission]

        # training new models
        logging.info('*' * 8 + ' Training tree for binding cases ' + '*' * 9)
        bind_param = baumWelch.hmm_train_and_test(binding_hmm, bind_parse_matrix, bind_emission,
                                                  maxIterations=max_iter_1, delta=delta)
        logging.info(
            '*' * 6 + ' Training tree for non-binding cases ' + '*' * 7)
        nonbind_param = baumWelch.hmm_train_and_test(nonbinding_hmm, nonbind_parse_matrix, nonbind_emission,
                                                     maxIterations=max_iter_2, delta=delta)

        # Finished training, reinitialized models with trained params
        binding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=bind_param['hmm']['state_transition_probabilities'],
                                              emission_probabilities=bind_param['hmm']['emission_probabilities'])
        nonbinding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=nonbind_param['hmm']['state_transition_probabilities'],
                                                 emission_probabilities=nonbind_param['hmm']['emission_probabilities'])

        dict_to_save['bind_mode_{}'.format(fold_iter)] = binding_hmm_trained
        dict_to_save['nonbind_mode_{}'.format(
            fold_iter)] = nonbinding_hmm_trained

        testset = bind_test + nonbind_test
        testlabel = [1] * len(bind_test) + [0] * len(nonbind_test)
        testpred = []
        testpred_posterior = []
        for i in range(len(testset)):
            iupac_text = testset[i]
            cv_iupac.append(iupac_text)
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_test = Glycan(iupac)

            # get test data features
            glycan_adj_matrix = glycan_test.get_adj_matrix()
            glycan_sparse_matrix = csr_matrix(glycan_adj_matrix)
            glycan_mono_emission = glycan_test.get_filtered_monosaccharide_emssions()
            glycan_link_emission = glycan_test.get_filtered_linkage_emssions()

            if use_edge:
                glycan_emission = [glycan_mono_emission, glycan_link_emission]
            else:
                glycan_emission = [glycan_mono_emission]

            # calculate the likelihood
            fwd_tree_sequence = fwd_seq_gen.forward_sequence_generator(
                glycan_sparse_matrix)

            bind_fwd_probs = forward.forward(
                binding_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)
            nonbind_fwd_probs = forward.forward(
                nonbinding_hmm_trained, glycan_sparse_matrix, glycan_emission, fwd_tree_sequence)

            # not use posterior
            if logsumexp(bind_fwd_probs.iloc[:, -1]) >= logsumexp(nonbind_fwd_probs.iloc[:, -1]):
                testpred.append(1)
            else:
                testpred.append(0)

            # use posterior
            if logsumexp(bind_fwd_probs.iloc[:, -1]) + log_by_chance_bind_prob >= logsumexp(nonbind_fwd_probs.iloc[:, -1]) + log_by_chance_nobind_prob:
                testpred_posterior.append(1)
            else:
                testpred_posterior.append(0)

        logging.info('Performence Metrics\n' +
                     get_metric_str(testlabel, testpred))
        logging.info('Performence Metrics  (Use Posterior)\n' +
                     get_metric_str(testlabel, testpred_posterior))

        cv_label += testlabel
        cv_pred += testpred
        cv_pred_posterior += testpred_posterior

    dict_to_save['y_label'] = cv_label
    dict_to_save['y_pred'] = cv_pred
    dict_to_save['y_iupac'] = cv_iupac
    pickle.dump(dict_to_save, pickle_file)
    pickle_file.close()
    logging.info('*' * 50)
    logging.info('Overall Performence Metrics\n' +
                 get_metric_str(cv_label, cv_pred))
    logging.info('Overall Performence Metrics  (Use Posterior)\n' +
                 get_metric_str(cv_label, cv_pred_posterior))
    logging.info('*' * 50)


def get_metric_str(ground_truth, prediction):
    metrics_str = ''
    metrics_str += 'F1 Score: {:.3f}\n'.format(
        f1_score(ground_truth, prediction))
    metrics_str += 'Accuracy Score: {:.3f}\n'.format(
        accuracy_score(ground_truth, prediction))
    metrics_str += 'Precision Score: {:.3f}\n'.format(
        precision_score(ground_truth, prediction))
    metrics_str += 'Recall Score: {:.3f}\n'.format(
        recall_score(ground_truth, prediction))
    metrics_str += classification_report(ground_truth, prediction)
    return metrics_str


if __name__ == '__main__':

    # Global variables for hyper params selection and logging
    parser = argparse.ArgumentParser('Glycan TreeHMM')
    parser.add_argument('--use_edge', default=False, action='store_true',
                        help='whether use link information as part of the features')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='number of folds for cross-validation')
    parser.add_argument('--max_iter_1', type=int, default=1,
                        help='maximum number of epochs for BW to train the positive class')
    parser.add_argument('--max_iter_2', type=int, default=None,
                        help='maximum number of epochs for BW to train the negative class, same as positive if None')
    parser.add_argument('--n_states', type=int, default=5,
                        help='number of hidden states')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='stop training when difference is less than delta')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save', type=str, default='EXP',
                        help='experiment name')
    args = parser.parse_args()

    # build results folder
    os.makedirs('./Results', exist_ok=True)
    args.save = './Results/eval-{}-{}'.format(
        args.save, time.strftime("%Y%m%d-%H%M%S"))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(args.save + '-log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)
    train_and_test(use_edge=args.use_edge, n_folds=args.n_folds, max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2,
                   n_states=args.n_states, delta=args.delta, random_seed=args.seed)
