import numpy as np
import copy
import re
import pandas as pd
from sklearn.model_selection import KFold
from treehmm4glycan import create_forest_inputs, get_iupcas
from TreeHMM4Glycan.Glycan import Glycan
from treehmm import initHMM, baumWelch, forward, fwd_seq_gen
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


def get_data():
    """Get IUPAC, and MScore.
    Returen, zipped data, protein names
    """
    iupac_data = pd.read_csv('./Data/IUPAC.csv')
    iupacs = iupac_data['IUPAC'].tolist()
    mscore_data = pd.read_csv('./Data/MScore_useful.csv')
    mscore = mscore_data.drop('ID', axis=1)
    col_names = mscore.columns.tolist()
    mscore = mscore.values.tolist()
    data = list(zip(iupacs, mscore))
    return data, col_names

def n_fold(data, col_names, protein, num_folds, seed = None):
    """10-fold cross validation with random sampling."""
    assert len(data) == 600
    IUPAC, mscore = list(zip(*data))
    protein_idx = col_names.index(protein)
    binding = []
    nonbinding = []
    for i in range(600):
        if mscore[i][protein_idx] >= 2:
            binding.append(IUPAC[i])
        else:
            nonbinding.append(IUPAC[i])
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
        nonbinding_train.append([nonbinding[i] for i in nonbinding_train_idx.tolist()])
        nonbinding_test.append([nonbinding[i] for i in nonbinding_test_idx.tolist()])

    return binding_train, binding_test, nonbinding_train, nonbinding_test


def train(n_folds=10, n_states=5, max_iter=50, delta=1e-5, random_seed=None):
    """Train GlyNet using n-fold cross-validation.
    """
    cv_label = []
    cv_pred = []

    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
    gylcans = {}

    monos = []
    counts = {}
    for id in iupacs:
        inpuac_text = iupacs[id]
        gylcan = Glycan(inpuac_text)
        if gylcan.get_num_nosaccharides() > 1:
            gylcans[id] = gylcan
            mono = gylcans[id].get_monosaccharide_emssions()
            monos += mono
            mono_count = gylcans[id].get_num_nosaccharides()
            # print(gylcans[id].get_monosaccharide_emssions())
            if mono_count not in counts:
                counts[mono_count] = 0
            counts[mono_count] += 1
    emissions = list(set(monos))
    emissions = [emissions]
    states = [str(i) for i in range(1, n_states + 1)]

    data, col_names = get_data()
    target_protein = 'AAL (100 ug/ml)'
    binding_train, binding_test, nonbinding_train, nonbinding_test = n_fold(data, col_names, target_protein,
                                                                            num_folds=n_folds, seed=random_seed)
    for bind_train, bind_test, nonbind_train, nonbind_test in zip(binding_train, binding_test, nonbinding_train, nonbinding_test):
        # prepare glycan dictionary
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

        bind_mono_emission = [bind_mono_emission]
        nonbind_mono_emission = [nonbind_mono_emission]

        bind_param = baumWelch.hmm_train_and_test(copy.deepcopy(binding_hmm), bind_parse_matrix, bind_mono_emission,
                                                  maxIterations=max_iter, delta=delta)
        nonbind_param = baumWelch.hmm_train_and_test(copy.deepcopy(nonbinding_hmm), nonbind_parse_matrix, nonbind_mono_emission,
                                                     maxIterations=max_iter, delta=delta)

        binding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=bind_param['hmm']['state_transition_probabilities'],
                                              emission_probabilities=bind_param['hmm']['emission_probabilities'])
        nonbinding_hmm_trained = initHMM.initHMM(states, emissions, state_transition_probabilities=nonbind_param['hmm']['state_transition_probabilities'],
                                                 emission_probabilities=nonbind_param['hmm']['emission_probabilities'])

        testset = bind_test + nonbind_test
        testlabel = [1] * len(bind_test) + [0] * len(nonbind_test)
        testpred = []
        for i in range(len(testset)):
            iupac_text = testset[i]
            iupac = re.split(r"\([^\)]*$", iupac_text, 1)[0]
            glycan_test = Glycan(iupac)

            # get test data features
            glycan_adj_matrix = glycan_test.get_adj_matrix()
            glycan_sparse_matrix = csr_matrix(glycan_adj_matrix)
            glycan_mono_emission = glycan_test.get_monosaccharide_emssions()
            glycan_mono_emission = [glycan_mono_emission]
            glycan_link_emission = glycan_test.get_linkage_emssions()

            # calculate the likelihood
            fwd_tree_sequence = fwd_seq_gen.forward_sequence_generator(glycan_sparse_matrix)

            bind_fwd_probs = forward.forward(binding_hmm_trained, glycan_sparse_matrix, glycan_mono_emission, fwd_tree_sequence)
            nonbind_fwd_probs = forward.forward(nonbinding_hmm_trained, glycan_sparse_matrix, glycan_mono_emission, fwd_tree_sequence)

            if logsumexp(bind_fwd_probs.iloc[:, -1]) >= logsumexp(nonbind_fwd_probs.iloc[:, -1]):
                testpred.append(1)
            else:
                testpred.append(0)

        print(classification_report(testlabel, testpred))
        cv_label += testlabel
        cv_pred += testpred
    return cv_label, cv_pred


if __name__ == '__main__':
    y_label, y_pred = train(n_folds=10, max_iter=100)
    print(classification_report(y_label, y_pred))


