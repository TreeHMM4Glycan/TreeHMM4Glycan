import math
import re
import logging
import sys
import argparse
import pickle
import time
import os
from typing import Dict, List
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


def create_glycan_from_str(iupac_text: str, single_end: bool) -> Glycan:
    """
    Method return an glycan based on input iupac names
    Args:
        iupac_text (str): iupac snfg string
        single_end (bool): boolean indicate if use single end on the tree

    Returns:
        Glycan: returned glycan
    """    
    iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
    return Glycan(iupac, single_end)


def prepare_data(training_data:List[int], single_end:bool):
    """[summary]

    Args:
        training_data (Dict[int,str]): [description]
        single_end (bool): [description]

    Returns:
        [type]: [description]
    """    
    glycans_train = {}
    for i in range(len(training_data)):
        glycans_train[i] = create_glycan_from_str(training_data[i], single_end)

    adj_matrix, mono_emissions, link_emissions = create_forest_inputs(
        glycans_train)
    parse_matrix_adj_matrix = csr_matrix(adj_matrix)
    return parse_matrix_adj_matrix, mono_emissions, link_emissions


def train_and_test(use_edge=False, single_end=True, n_folds=10, n_states=5, max_iter=3, num_epoch=1, delta=1e-5,
                   random_seed=None, save_file = None):
    """Train GlyNet using n-fold cross-validation.
    """
    logging.info('Training with hyper parameters:\nUse edge: {}\nUse single Node: {}\nNum folds: {}\nNum States: {}\nMax Iter: {}\nNum Epioch: {}\nDelta: {}\nRandom seed: {}'.
                 format(use_edge, single_end, n_folds, n_states, max_iter, num_epoch, delta, random_seed))

    dict_to_save = {}
    cv_label = []
    cv_pred = []
    cv_pred_posterior = []
    cv_iupac = []

    # Get total possible number of emissions
    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
    _, mono_emissions, link_emissions = get_glycans(iupacs, single_end)

    if use_edge:
        emissions = [mono_emissions, link_emissions]
    else:
        emissions = [mono_emissions]
    states = [str(i) for i in range(1, n_states + 1)]

    # Prepare data
    # TODO it works, but we should pass glycan around not iupac and read it over and over 
    data, col_names = get_data()
    target_protein = 'AAL (100 ug/ml)'
    binding_train, binding_test, nonbinding_train, nonbinding_test = n_fold(data, col_names, target_protein,
                                                                            num_folds=n_folds, seed=random_seed)

    # prepare data
    bind_emissions = []
    nonbind_emissions = []
    binding_hmms = []
    nonbinding_hmms = []
    bind_parse_matrices = []
    nonbind_parse_matrices = []

    for fold_iter, (bind_train, bind_test, nonbind_train, nonbind_test) in enumerate(zip(binding_train, binding_test,
                                                                                         nonbinding_train,
                                                                                         nonbinding_test)):
        
        # prepare glycans dictionary
        bind_parse_matrix, bind_mono_emission, bind_link_emission = prepare_data(
            bind_train, single_end)
        nonbind_parse_matrix, nonbind_mono_emission, nonbind_link_emission = prepare_data(
            nonbind_train, single_end)

        binding_hmm = initHMM.initHMM(states, emissions)
        nonbinding_hmm = initHMM.initHMM(states, emissions)

        binding_hmms.append(binding_hmm)
        nonbinding_hmms.append(nonbinding_hmm)
        bind_parse_matrices.append(bind_parse_matrix)
        nonbind_parse_matrices.append(nonbind_parse_matrix)

        if use_edge:
            bind_emission = [bind_mono_emission, bind_link_emission]
            nonbind_emission = [nonbind_mono_emission, nonbind_link_emission]
        else:
            bind_emission = [bind_mono_emission]
            nonbind_emission = [nonbind_mono_emission]

        bind_emissions.append(bind_emission)
        nonbind_emissions.append(nonbind_emission)

    # training for each epoch
    for epoch in range(0, num_epoch):
        for fold_iter, (bind_train, bind_test, nonbind_train, nonbind_test) in enumerate(zip(binding_train, binding_test,
                                                                                             nonbinding_train,
                                                                                             nonbinding_test)):
            logging.info('*' * 50)
            logging.info(
                'Training and testing in Epcoh #{} fold #{}'.format(epoch, fold_iter))
            # compute by chance prob for each class
            by_chance_bind_prob = len(bind_train) / \
                (len(bind_train) + len(nonbind_train))
            logging.info('By Chance Bind Prob: {:.3f}'.format(
                by_chance_bind_prob))

            # training new models
            logging.info(
                '*' * 8 + ' Training tree for binding cases ' + '*' * 9)
            bind_param = baumWelch.hmm_train_and_test(binding_hmms[fold_iter], bind_parse_matrices[fold_iter], bind_emissions[fold_iter],
                                                      maxIterations=max_iter, delta=delta)
            logging.info(
                '*' * 6 + ' Training tree for non-binding cases ' + '*' * 7)
            nonbind_param = baumWelch.hmm_train_and_test(nonbinding_hmms[fold_iter], nonbind_parse_matrices[fold_iter], nonbind_emissions[fold_iter],
                                                         maxIterations=max_iter, delta=delta)

            # Finished training, reinitialized models with trained params
            binding_hmms[fold_iter] = initHMM.initHMM(states, emissions, state_transition_probabilities=bind_param['hmm']['state_transition_probabilities'],
                                                      emission_probabilities=bind_param['hmm']['emission_probabilities'])
            nonbinding_hmms[fold_iter] = initHMM.initHMM(states, emissions, state_transition_probabilities=nonbind_param['hmm']['state_transition_probabilities'],
                                                         emission_probabilities=nonbind_param['hmm']['emission_probabilities'])

            dict_to_save['bind_mode_{}_{}'.format(
                epoch, fold_iter)] = binding_hmms[fold_iter]
            dict_to_save['nonbind_mode_{}_{}'.format(
                epoch, fold_iter)] = nonbinding_hmms[fold_iter]

            # compute training metrics
            train_preds, train_preds_posterior, bind_train_ll, non_bind_train_ll = batch_predict(
                bind_train, nonbind_train, cv_iupac, single_end, use_edge, binding_hmms[fold_iter], nonbinding_hmms[fold_iter], by_chance_bind_prob)
            train_labels = [1] * len(bind_train) + [0] * len(nonbind_train)

            logging.info('Epcoh #{} Fold #{} Training Performence Metrics\nBind Training LL: {:.3f} \nNon Bind Training LL: {:.3f} \n'.format(epoch,
                                                                                                                                              fold_iter, bind_train_ll, non_bind_train_ll))
            logging.info('Epcoh #{} Fold #{} Training Performence Metrics\n'.format(epoch, fold_iter) +
                         get_metric_str(train_labels, train_preds))
            logging.info('Epcoh #{} Fold #{} Training  Performence Metrics  (Use Posterior)\n'.format(epoch, fold_iter) +
                         get_metric_str(train_labels, train_preds_posterior))

            # compute testing metrics
            test_preds, test_preds_posterior, _, _ = batch_predict(
                bind_test, nonbind_test, cv_iupac, single_end, use_edge, binding_hmms[fold_iter], nonbinding_hmms[fold_iter], by_chance_bind_prob)
            test_labels = [1] * len(bind_test) + [0] * len(nonbind_test)
            cv_label += test_labels
            cv_pred += test_preds
            cv_pred_posterior += test_preds_posterior

            logging.info('Fold #{} Testing Performence Metrics\n'.format(fold_iter) +
                         get_metric_str(test_labels, test_preds))
            logging.info('Fold #{} Testing Performence Metrics  (Use Posterior)\n'.format(fold_iter) +
                         get_metric_str(test_labels, test_preds_posterior))

        dict_to_save['Epoch_{}_y_label'.format(epoch)] = cv_label
        dict_to_save['Epoch_{}_y_pred'.format(epoch)] = cv_pred
        dict_to_save['Epoch_{}_y_iupac'.format(epoch)] = cv_iupac
        
        logging.info('*' * 50)
        logging.info('Overall Performence Metrics\n' +
                     get_metric_str(cv_label, cv_pred))
        logging.info('Overall Performence Metrics  (Use Posterior)\n' +
                     get_metric_str(cv_label, cv_pred_posterior))
        logging.info('*' * 50)
    
    if save_file is not None:
        with open(save_file+ '-model.pkl', 'wb') as pickle_file:
            pickle.dump(dict_to_save, pickle_file)

def batch_predict(bind_test_set: List[str], nonbind_test_set: List[str], cv_iupac: List[str], single_end: bool, use_edge: bool, binding_hmm_trained, nonbinding_hmm_trained, by_chance_bind_prob):

    test_pred = []
    test_pred_posterior = []

    log_by_chance_bind_prob = math.log(by_chance_bind_prob)
    log_by_chance_nobind_prob = math.log(1 - by_chance_bind_prob)

    bind_ll = 0
    non_bind_ll = 0
    test_set = bind_test_set + nonbind_test_set

    num_cases = len(test_set)

    for i in range(num_cases):
        iupac_text = test_set[i]
        cv_iupac.append(iupac_text)
        glycan_test = create_glycan_from_str(iupac_text, single_end)

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

        # update lls:
        case_bind_ll_per_end = []
        case_none_bind_ll_per_end= []
        for end_idx in glycan_test.get_end_nodes_indices():
            case_bind_ll_per_end.append(logsumexp(bind_fwd_probs.iloc[:, end_idx]))
            case_none_bind_ll_per_end.append(logsumexp(nonbind_fwd_probs.iloc[:, end_idx]))
        
        case_bind_ll = logsumexp(case_bind_ll_per_end)
        case_non_bind_ll = logsumexp(case_none_bind_ll_per_end)
        
        # this is one for training evalution
        if i < len(bind_test_set):
            bind_ll += case_bind_ll
        else:
            non_bind_ll += case_non_bind_ll

        # not use posterior
        if case_bind_ll >= case_non_bind_ll:
            test_pred.append(1)
        else:
            test_pred.append(0)

        # use posterior
        if case_bind_ll + log_by_chance_bind_prob >= case_non_bind_ll + log_by_chance_nobind_prob:
            test_pred_posterior.append(1)
        else:
            test_pred_posterior.append(0)

    return test_pred, test_pred_posterior, bind_ll/len(bind_test_set), non_bind_ll/len(nonbind_test_set)


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
    parser.add_argument('--use_single', default=False, action='store_true',
                        help='whether use single end for glycan')
    parser.add_argument('--n_folds', type=int, default = 5,
                        help='number of folds for cross-validation')
    parser.add_argument('--max_iter', type=int, default = 3,
                        help='maximum number of training iteration per epoch')
    parser.add_argument('--num_epoch', type=int, default = 3,
                        help='number of epoch')
    parser.add_argument('--n_states', type=int, default = 2,
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
    train_and_test(use_edge=args.use_edge, single_end = args.use_single, n_folds=args.n_folds, max_iter=args.max_iter, num_epoch=args.num_epoch,
                   n_states=args.n_states, delta=args.delta, random_seed=args.seed, save_file = args.save)
