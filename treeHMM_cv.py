import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import KFold
from treehmm4glycan import create_forest_inputs, get_iupcas
from TreeHMM4Glycan.Glycan import Glycan
from treehmm import initHMM, baumWelch


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


def train():
    """Train GlyNet using n-fold cross-validation.
    """
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
    states = ['1', '2', '3', '4', '5']

    data, col_names = get_data()
    target_protein = 'AAL (100 ug/ml)'
    binding_train, binding_test, nonbinding_train, nonbinding_test = n_fold(data, col_names, target_protein, num_folds=10)
    for bind_train, bind_test, nonbind_train, nonbind_test in zip(binding_train, binding_test, nonbinding_train, nonbinding_test):
        # prepare glycan dictionary
        glycans_bind_train = {}
        glycans_nonbind_train = {}
        for i in range(len(bind_train)):
            inpuac_text = bind_train[i]
            glycan_bind_train = Glycan(inpuac_text)
            glycans_bind_train[i] = glycan_bind_train

        for i in range(len(nonbind_train)):
            inpuac_text = nonbind_train[i]
            glycan_nonbind_train = Glycan(inpuac_text)
            glycans_nonbind_train[i] = glycan_nonbind_train

        bind_adj_matrix, bind_mono_emission, bind_link_emission = create_forest_inputs(glycans_bind_train)
        nonbind_adj_matrix, nonbind_mono_emission_obs, nonbind_link_emission = create_forest_inputs(glycans_nonbind_train)

        binding_hmm = initHMM.initHMM(states, emissions, bind_adj_matrix)
        nonbinding_hmm = initHMM.initHMM(states, emissions, nonbind_adj_matrix)

        bind_mono_emission = [bind_mono_emission]
        nonbind_mono_emission_obs = [bind_mono_emission]

        bind_param = baumWelch.hmm_train_and_test(copy.deepcopy(binding_hmm), bind_mono_emission)
        nonbind_param = baumWelch.hmm_train_and_test(copy.deepcopy(nonbinding_hmm), nonbind_mono_emission_obs)

        # needs to be changed here
        binding_hmm_trained = initHMM.initHMM(states, emissions, )
        nonbinding_hmm_trained = initHMM.initHMM(states, emissions, )

        testset = bind_test + nonbind_test
        for i in range(len(testset)):
            inpuac_text = testset[i]
            glycan_test = Glycan(inpuac_text)

            glycan_adj_matrix = glycan_test.get_adj_matrix()
            glycan_mono_emission = glycan_test.get_monosaccharide_emssions()
            glycan_mono_emission = glycan_test.get_linkage_emssions()



if __name__ == '__main__':
    train()


