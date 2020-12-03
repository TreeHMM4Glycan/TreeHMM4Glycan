import math
from math import inf
import re
import logging
import sys
import argparse
import pickle
import time
import os
from typing import List
import pandas as pd
from sklearn.model_selection import KFold
from treehmm4glycan import create_forest_inputs, get_iupcas, get_glycans
from TreeHMM4Glycan.Glycan import Glycan
from treehmm import initHMM, baumWelch, forward, fwd_seq_gen
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
import numpy as np


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


def forward_tree_ll(hmm, adjacent_matrix: np.ndarray, emission_observation, fwd_tree_sequence):
    """
    Args:
        hmm: It is a dictionary given as output by initHMM.py file
        emission_observation: emission_observation is a list of list consisting "k" lists for "k"
            features, each vector being a character series of discrete emission
            values at different nodes serially sorted by node number
        forward_tree_sequence: It is a list denoting the order of nodes in which
            the tree should be traversed in forward direction(from roots to
            leaves).

    Returns:
        forward_probs: A dataframe of size (N * D) denoting the forward
        probabilites at each node of the tree, where "N" is possible no. of
        states and "D" is the total number of nodes in the tree
    """
    forward_probabilities = forward.forward(hmm, csr_matrix(
        adjacent_matrix), emission_observation, fwd_tree_sequence)

    # let us travesl this tree via Floyd–Warshall
    # node_id, segment start log prob
    # node_stack = [[0, 0]]

    # forward_ll = 0

    # while len(node_stack) > 0:
    #     # print(node_stack)
    #     current_node_data = node_stack.pop()
    #     node_id, segment_start_ll = current_node_data

    #     # segment start at branch and end at other branch
    #     if not sum(adjacent_matrix[node_id]) == 1:
    #         node_log_prob = logsumexp(forward_probabilities.iloc[:, node_id])
    #         # log_segment_prob is the segment end node ll - segement start ll
    #         log_segment_prob = node_log_prob - segment_start_ll
    #         # add ll of this segment
    #         forward_ll += log_segment_prob
    #         segment_start_ll = node_log_prob

    #     for child_id, edge in enumerate(adjacent_matrix[node_id]):
    #         if edge:
    #             node_stack.append([child_id, segment_start_ll])
    forward_ll = forward_tree_ll_recursion(adjacent_matrix, 0, forward_probabilities, 0)

    return forward_ll


def forward_tree_ll_recursion(adjacent_matrix: np.ndarray, node_id, forward_probabilities, segment_start_ll) -> float:
    num_child = np.count_nonzero(adjacent_matrix[node_id])
    # print(adjacent_matrix)
    # print(adjacent_matrix.shape)
    # print(node_id, num_child)
    # print(adjacent_matrix[node_id])
    # print(np.where(adjacent_matrix[node_id] == 1)[0][0])

    # retun segment_ll at leaf node
    if num_child == 0:
        return logsumexp(forward_probabilities.iloc[:, node_id]) - segment_start_ll
    # if there is no branch
    elif num_child == 1:
        child_id = np.where(adjacent_matrix[node_id, :] == 1)[0][0]
        # print(adjacent_matrix[node_id, :])
        # print(child_id)
        return forward_tree_ll_recursion(adjacent_matrix, child_id, forward_probabilities, segment_start_ll)
    # if there is a branch .... it become a bit complicated
    if not sum(adjacent_matrix[node_id]) == 1:
        cpd = forward_probabilities.iloc[:, node_id]
        segment_start_ll = logsumexp(forward_probabilities.iloc[:, node_id])
        terms = []
        for current_node_state_ll in cpd:
            state_ll = current_node_state_ll
            for child_id in np.where(adjacent_matrix[node_id] == 1)[0]:
                state_ll += forward_tree_ll_recursion(adjacent_matrix, child_id, forward_probabilities,
                                                      segment_start_ll)
            terms.append(state_ll)
        return logsumexp(terms)


def create_glycan_from_str(iupac_text: str) -> Glycan:
    """
    Method return an glycan based on input iupac names
    Args:
        iupac_text (str): iupac snfg string

    Returns:
        Glycan: returned glycan
    """
    iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
    return Glycan(iupac)


def prepare_data(training_data: List[int]):
    """[summary]

    Args:
        training_data (Dict[int,str]): [description]
        single_end (bool): [description]

    Returns:
        [type]: [description]
    """
    glycans_train = {}
    for i in range(len(training_data)):
        glycans_train[i] = create_glycan_from_str(training_data[i])

    adj_matrix, mono_emissions, link_emissions = create_forest_inputs(
        glycans_train)
    parse_matrix_adj_matrix = csr_matrix(adj_matrix)
    return parse_matrix_adj_matrix, mono_emissions, link_emissions


def train_and_test(use_edge=False, n_folds=5, n_states=2, max_iter=3, num_epoch=1, delta=1e-5, random_seed=None,
                   save_file=None):
    """Train GlyNet using n-fold cross-validation.
    """
    logging.info('Training with hyper parameters:\nUse Edge: {}\nNum Folds: {}\nNum States: {}\nMax Iter: {}\nNum Epoch: {}\nDelta: {}\nRandom seed: {}'.
                 format(use_edge, n_folds, n_states, max_iter, num_epoch, delta, random_seed))

    dict_to_save = {}
    dict_to_save['Config'] = {
        'Use Edge': use_edge,
        'Num Folds': n_folds,
        'Num States': n_states,
        'Max Iter': max_iter,
        'Num Epoch': num_epoch,
        'Delta': delta,
        'Random Seed': random_seed
    }
    dict_to_save['Metrics'] = {'Normal': {
        'F1': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
    },
        'Posterior': {
            'F1': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
        }
    }

    cv_label = []
    cv_pred = []
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
    strong_bind_train, strong_bind_test, mid_bind_train, mid_bind_test, weak_bind_train, weak_bind_test = \
        n_fold(data, col_names, target_protein, num_folds=n_folds, seed=random_seed)

    # prepare data
    stro_emissions = []
    mid_emissions = []
    weak_emissions = []
    stro_hmms = []
    mid_hmms = []
    weak_hmms = []
    stro_parse_matrices = []
    mid_parse_matrices = []
    weak_parse_matrices = []

    for fold_iter, (strong_train, strong_test, mid_train, mid_test, weak_train, weak_test) in enumerate(
            zip(strong_bind_train, strong_bind_test, mid_bind_train, mid_bind_test, weak_bind_train, weak_bind_test)):

        stro_parse_matrix, stro_mono_emission, stro_link_emission = prepare_data(strong_train)
        mid_parse_matrix, mid_mono_emission, mid_link_emission = prepare_data(mid_train)
        weak_parse_matrix, weak_mono_emission, weak_link_emission = prepare_data(weak_train)

        stro_hmm = initHMM.initHMM(states, emissions)
        mid_hmm = initHMM.initHMM(states, emissions)
        weak_hmm = initHMM.initHMM(states, emissions)

        stro_hmms.append(stro_hmm)
        mid_hmms.append(mid_hmm)
        weak_hmms.append(weak_hmm)
        stro_parse_matrices.append(stro_parse_matrix)
        mid_parse_matrices.append(mid_parse_matrix)
        weak_parse_matrices.append(weak_parse_matrix)

        if use_edge:
            stro_emission = [stro_mono_emission, stro_link_emission]
            mid_emission = [mid_mono_emission, mid_link_emission]
            weak_emission = [weak_mono_emission, weak_link_emission]
        else:
            stro_emission = [stro_mono_emission]
            mid_emission = [mid_mono_emission]
            weak_emission = [weak_mono_emission]

        stro_emissions.append(stro_emission)
        mid_emissions.append(mid_emission)
        weak_emissions.append(weak_emission)

    for epoch in range(0, num_epoch):

        cv_label = []
        cv_pred = []
        cv_pred_posterior = []
        cv_iupac = []
        cv_f1 = []
        cv_accu = []
        cv_precision = []
        cv_recall = []
        cv_f1_posterior = []
        cv_accu_posterior = []
        cv_precision_posterior = []
        cv_recall_posterior = []

        for fold_iter, (strong_train, strong_test, mid_train, mid_test, weak_train, weak_test) in enumerate(
                zip(strong_bind_train, strong_bind_test, mid_bind_train, mid_bind_test, weak_bind_train,
                    weak_bind_test)):
            logging.info('*' * 50)
            logging.info('Training and testing in Epoch #{} fold #{}'.format(epoch, fold_iter))
            by_chance_stro_prob = len(strong_train) / (len(strong_train) + len(mid_train) + len(weak_train))
            by_chance_mid_prob = len(mid_train) / (len(strong_train) + len(mid_train) + len(weak_train))
            by_chance_weak_prob = len(weak_train) / (len(strong_train) + len(mid_train) + len(weak_train))
            logging.info('By Chance Strong Prob: {:.3f}'.format(by_chance_stro_prob))
            logging.info('By Chance Medium Prob: {:.3f}'.format(by_chance_mid_prob))
            logging.info('By Chance Weak Prob: {:.3f}'.format(by_chance_weak_prob))

            # training new models
            logging.info(
                '*' * 8 + ' Training tree for strong-bind cases ' + '*' * 9)
            stro_param = baumWelch.hmm_train_and_test(stro_hmms[fold_iter], stro_parse_matrices[fold_iter],
                                                      stro_emissions[fold_iter], maxIterations=max_iter, delta=delta)
            logging.info(
                '*' * 8 + ' Training tree for medium-bind cases ' + '*' * 9)
            mid_param = baumWelch.hmm_train_and_test(mid_hmms[fold_iter], mid_parse_matrices[fold_iter],
                                                     mid_emissions[fold_iter], maxIterations=max_iter, delta=delta)
            logging.info(
                '*' * 9 + ' Training tree for weak-bind cases ' + '*' * 10)
            weak_param = baumWelch.hmm_train_and_test(weak_hmms[fold_iter], weak_parse_matrices[fold_iter],
                                                      weak_emissions[fold_iter], maxIterations=max_iter, delta=delta)

            # Finished training, reinitialized models with trained params
            stro_hmms[fold_iter] = initHMM.initHMM(states, emissions, state_transition_probabilities=stro_param['hmm']['state_transition_probabilities'],
                                                   emission_probabilities=stro_param['hmm']['emission_probabilities'])
            mid_hmms[fold_iter] = initHMM.initHMM(states, emissions, state_transition_probabilities=mid_param['hmm']['state_transition_probabilities'],
                                                  emission_probabilities=mid_param['hmm']['emission_probabilities'])
            weak_hmms[fold_iter] = initHMM.initHMM(states, emissions, state_transition_probabilities=weak_param['hmm']['state_transition_probabilities'],
                                                   emission_probabilities=weak_param['hmm']['emission_probabilities'])

            dict_to_save['strong_mode_{}_{}'.format(
                epoch, fold_iter)] = stro_hmms[fold_iter]
            dict_to_save['mid_mode_{}_{}'.format(
                epoch, fold_iter)] = mid_hmms[fold_iter]
            dict_to_save['weak_mode_{}_{}'.format(
                epoch, fold_iter)] = mid_hmms[fold_iter]

            # compute training metrics
            train_preds, train_preds_posterior, stro_train_ll, mid_train_ll, weak_train_ll = batch_predict(
                strong_train, mid_train, weak_train, cv_iupac, use_edge, stro_hmms[fold_iter], mid_hmms[fold_iter],
                weak_hmms[fold_iter], by_chance_stro_prob, by_chance_mid_prob, by_chance_weak_prob)
            train_labels = [2] * len(strong_train) + [1] * len(mid_train) + [0] * len(weak_train)

            _, _, _, _, metrics_str = get_metric(train_labels, train_preds)
            logging.info('\nEpcoh #{} Fold #{} Training Performence Metrics\nStrong Training LL: {:.3f} \nMedium Bind Training LL: {:.3f} \nWeak Bind Training LL: {:.3f}'.
                         format(epoch, fold_iter, stro_train_ll, mid_train_ll, weak_train_ll) + metrics_str)

            _, _, _, _, metrics_str = get_metric(train_labels, train_preds_posterior)
            logging.info('\nEpcoh #{} Fold #{} Training  Performence Metrics  (Use Posterior)\n'.format(
                         epoch, fold_iter) + metrics_str)

            # compute testing metrics
            test_preds, test_preds_posterior, _, _, _ = batch_predict(strong_test, mid_test, weak_test, cv_iupac,
                                                                      use_edge, stro_hmms[fold_iter],
                                                                      mid_hmms[fold_iter], weak_hmms[fold_iter],
                                                                      by_chance_stro_prob, by_chance_mid_prob,
                                                                      by_chance_weak_prob)
            test_labels = [2] * len(strong_test) + [1] * len(mid_test) + [0] * len(weak_test)
            cv_label += test_labels
            cv_pred += test_preds
            cv_pred_posterior += test_preds_posterior

            f1, accu, precision, recall, metrics_str = get_metric(test_labels, test_preds)
            cv_f1.append(f1)
            cv_accu.append(accu)
            cv_precision.append(precision)
            cv_recall.append(recall)
            logging.info('Fold #{} Testing Performence Metrics\n'.format(fold_iter) + metrics_str)

            f1, accu, precision, recall, metrics_str = get_metric(test_labels, test_preds_posterior)
            cv_f1_posterior.append(f1)
            cv_accu_posterior.append(accu)
            cv_precision_posterior.append(precision)
            cv_recall_posterior.append(recall)
            logging.info('Fold #{} Testing Performence Metrics  (Use Posterior)\n'.format(fold_iter) + metrics_str)

        dict_to_save['Epoch_{}_y_label'.format(epoch)] = cv_label
        dict_to_save['Epoch_{}_y_pred'.format(epoch)] = cv_pred
        dict_to_save['Epoch_{}_y_pred_posterior'.format(
            epoch)] = cv_pred_posterior
        dict_to_save['Epoch_{}_y_iupac'.format(epoch)] = cv_iupac

        logging.info('*' * 50)

        fill_save_metrics(dict_to_save, 'Normal', cv_f1,
                          cv_accu, cv_precision, cv_recall)
        logging.info('\nEpoch #{} Overall Performence Metrics\n'.format(
            epoch) + get_overall_metric_str(dict_to_save, 'Normal', epoch))

        fill_save_metrics(dict_to_save, 'Posterior', cv_f1_posterior,
                          cv_accu_posterior, cv_precision_posterior, cv_recall_posterior)
        logging.info('\nEpoch #{} Overall Performence Metrics (Use Posterior)\n'.format(
            epoch) + get_overall_metric_str(dict_to_save, 'Posterior', epoch))
        logging.info('*' * 50)

    if save_file is not None:
        with open(save_file + '-model.pkl', 'wb') as pickle_file:
            pickle.dump(dict_to_save, pickle_file)


def fill_save_metrics(dict_to_save, key, cv_f1, cv_accu, cv_precision, cv_recall):
    dict_to_save['Metrics'][key]['F1'].append([np.mean(cv_f1), np.std(cv_f1)])
    dict_to_save['Metrics'][key]['Accuracy'].append(
        [np.mean(cv_accu), np.std(cv_accu)])
    dict_to_save['Metrics'][key]['Precision'].append(
        [np.mean(cv_precision), np.std(cv_precision)])
    dict_to_save['Metrics'][key]['Recall'].append(
        [np.mean(cv_recall), np.std(cv_recall)])


def get_overall_metric_str(dict_to_save, key, epoch_idx):
    metrics_str = ''
    metrics_str += 'F1 Score: {:.3f} std:{:.3f}\n'.format(
        *dict_to_save['Metrics'][key]['F1'][epoch_idx])
    metrics_str += 'Accuracy Score: {:.3f} std:{:.3f}\n'.format(
        *dict_to_save['Metrics'][key]['Accuracy'][epoch_idx])
    metrics_str += 'Precision Score: {:.3f} std:{:.3f}\n'.format(
        *dict_to_save['Metrics'][key]['Precision'][epoch_idx])
    metrics_str += 'Recall Score: {:.3f} std:{:.3f}\n'.format(
        *dict_to_save['Metrics'][key]['Recall'][epoch_idx])
    return metrics_str


def get_metric(ground_truth, prediction):
    f1 = f1_score(ground_truth, prediction, average='weighted')
    accu = accuracy_score(ground_truth, prediction)
    precision = precision_score(ground_truth, prediction, average='weighted')
    recall = recall_score(ground_truth, prediction, average='weighted')

    metrics_str = ''
    metrics_str += 'F1 Score: {:.3f}\n'.format(f1)
    metrics_str += 'Accuracy Score: {:.3f}\n'.format(accu)
    metrics_str += 'Precision Score: {:.3f}\n'.format(precision)
    metrics_str += 'Recall Score: {:.3f}\n'.format(recall)
    metrics_str += classification_report(ground_truth, prediction)
    return f1, accu, precision, recall, metrics_str


def batch_predict(stro_test_set: List[str], mid_test_set: List[str], weak_test_set: List[str], cv_iupac: List[str],
                  use_edge: bool, stro_hmm_trained, mid_hmm_trained, weak_hmm_trained, by_chance_stro_prob,
                  by_chance_mid_prob, by_chance_weak_prob):

    test_pred = []
    test_pred_posterior = []

    log_by_chance_stro_prob = math.log(by_chance_stro_prob)
    log_by_chance_mid_prob = math.log(by_chance_mid_prob)
    log_by_chance_weak_prob = math.log(by_chance_weak_prob)

    stro_ll = 0
    mid_ll = 0
    weak_ll = 0
    test_set = stro_test_set + mid_test_set + weak_test_set

    num_cases = len(test_set)

    for i in range(num_cases):
        iupac_text = test_set[i]
        cv_iupac.append(iupac_text)
        glycan_test = create_glycan_from_str(iupac_text)

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

        case_stro_ll = forward_tree_ll(
            stro_hmm_trained, glycan_adj_matrix, glycan_emission, fwd_tree_sequence)
        case_mid_ll = forward_tree_ll(
            mid_hmm_trained, glycan_adj_matrix, glycan_emission, fwd_tree_sequence)
        case_weak_ll = forward_tree_ll(
            weak_hmm_trained, glycan_adj_matrix, glycan_emission, fwd_tree_sequence)

        # this is one for training evalution
        if i < len(stro_test_set):
            stro_ll += case_stro_ll
        elif i < len(mid_test_set) and i > len(stro_test_set):
            mid_ll += case_mid_ll
        else:
            weak_ll += case_weak_ll

        # not use posterior
        ll_list = [case_weak_ll, case_mid_ll, case_stro_ll]
        test_pred.append(ll_list.index(max(ll_list)))

        # use posterior
        post_ll_list = [case_weak_ll + log_by_chance_weak_prob, case_mid_ll + log_by_chance_mid_prob,
                        case_stro_ll + log_by_chance_stro_prob]
        test_pred_posterior.append(post_ll_list.index(max(post_ll_list)))

    return test_pred, test_pred_posterior, stro_ll/len(stro_test_set), mid_ll/len(mid_test_set), weak_ll/len(weak_test_set)



if __name__ == '__main__':

    # Global variables for hyper params selection and logging
    parser = argparse.ArgumentParser('Glycan TreeHMM')
    parser.add_argument('--use_edge', default=False, action='store_true',
                        help='whether use link information as part of the features')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='number of folds for cross-validation')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='maximum number of epochs for BW to train')
    parser.add_argument('--num_epoch', type=int, default=1,
                        help='number of epoch')
    parser.add_argument('--n_states', type=int, default=5,
                        help='number of hidden states')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='stop training when difference is less than delta')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    args = parser.parse_args()

    os.makedirs('./Saved3c', exist_ok=True)
    args.save = './Saved3c/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(args.save + '-log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)
    train_and_test(use_edge=args.use_edge, n_folds=args.n_folds, max_iter=args.max_iter, num_epoch=args.num_epoch,
                   n_states=args.n_states, delta=args.delta, random_seed=args.seed, save_file=args.save)
