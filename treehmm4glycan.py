from treehmm import initHMM, baumWelch
from TreeHMM4Glycan.Glycan import Glycan
import csv
import numpy as np
import pandas as pd
import copy

def get_iupcas(iupac_name_file):
    iupacs = {}
    with open(iupac_name_file) as file_in:
        csv_reader = csv.reader(file_in)
        for idx,row in enumerate(csv_reader):
            if idx == 0:
                continue
            id = row[0]
            iupac = row[1]
            count = int(row[2])
            if count > 0:
                iupacs[id] = iupac
    return iupacs

def example1(glycan:Glycan, number_state = 2):
    sample_tree = glycan.get_adj_matrix()
    emissions = glycan.get_emssions()

    states = ['P','N']
    #state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states,emissions,sample_tree,state_transition_probabilities = state_transition_probabilities)

    # Declaring the emission_observation list
    emission_observation = [["L","M","H","M","L","L"],["M","L","H","H","L","L"]]

    # Declaring the observed_states_training_nodes
    data = {'node' : [0,3,4], 'state' : ['P','N','P']}
    observed_states_training_nodes = pd.DataFrame(data = data,columns=["node","state"])

    # Declaring the observed_states_validation_nodes
    data1 = {'node' : [1,2], 'state' : ['N','P']}
    observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)
    #learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)

    print("newparam :", newparam)
    print("\n")
    #print("learntHMM : ", learntHMM)
    #print("\n")

def example2():
    sample_tree = np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(6,6)
    #sparse = csr_matrix(sample_tree)
    states = ['P','N']
    emissions = [['L','M','H'],['L','M','H']]
    state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states,emissions,sample_tree,state_transition_probabilities = state_transition_probabilities)

    # Declaring the emission_observation list
    emission_observation = [["L","M","H","M","L","L"],["M","L","H","H","L","L"]]

    # Declaring the observed_states_training_nodes
    data = {'node' : [0,3,4], 'state' : ['P','N','P']}
    observed_states_training_nodes = pd.DataFrame(data = data,columns=["node","state"])

    # Declaring the observed_states_validation_nodes
    data1 = {'node' : [1,2], 'state' : ['N','P']}
    observed_states_validation_nodes = pd.DataFrame(data = data1,columns=["node","state"])

    # For calculating the forward probabilities
    # ForwardProbs = forward.forward(hmm,emission_observation,forward_tree_sequence,observed_states_training_nodes)
    # print(ForwardProbs)

    # For calculating the backward probabilities
    # BackwardProbs = backward.backward(hmm,emission_observation,backward_tree_sequence,observed_states_training_nodes)
    # print(BackwardProbs)

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)
    #learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm), emission_observation, observed_states_training_nodes, observed_states_validation_nodes)

    print("newparam :", newparam)
    print("\n")
    #print("learntHMM : ", learntHMM)
    #print("\n")

if __name__ == "__main__":
    #pass
    example2()
    #iupac_name_file = './Data/IUPAC.csv'
