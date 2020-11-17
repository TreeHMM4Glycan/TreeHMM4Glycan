from treehmm import initHMM, baumWelch
from TreeHMM4Glycan.Glycan import Glycan
import csv
import numpy as np
import pandas as pd
import copy
import re

def get_iupcas(iupac_name_file):
    iupacs = {}
    with open(iupac_name_file) as file_in:
        csv_reader = csv.reader(file_in)
        for idx,row in enumerate(csv_reader):
            if idx == 0:
                continue
            id = row[0]
            # remove right most part'(a1-sp14'
            iupac = re.split(r"\([^\)]*$", row[1], 1)[0]
            count = int(row[2])
            if count > 0:
                iupacs[id] = iupac
    return iupacs

def create_and_run_treehmm(glycan:Glycan, number_state = 2, emissions = None):
    sample_tree = glycan.get_adj_matrix()

    # Declaring the emission_observation list
    emission_observation = glycan.get_emssions()

    #states = ['P','N']
    # create states
    states = [ str(i) for i in range(number_state)]
    #state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states, emissions, sample_tree)


    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.baumWelchRecursion(hmm, emission_observation)

def example1():
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
    #example1()
    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
    gylcans = {}
    monos = []
    for id in iupacs:
        inpuac_text = iupacs[id]
        #print(inpuac_text)
        gylcans[id] = Glycan(inpuac_text)
        mono = gylcans[id].get_emssions()
        if None in mono:
            print(inpuac_text)
            print(mono)
        monos += mono
    
    #go over each gylcans
    print(set(monos))