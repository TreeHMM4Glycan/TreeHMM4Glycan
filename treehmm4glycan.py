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

def create_and_run_treehmm(glycan:Glycan, number_state, possible_emissions):
    sample_tree = glycan.get_adj_matrix()

    # Declaring the emission_observation list
    emission_observation = [glycan.get_emssions()]
    
    #states = ['P','N']
    # create states
    states = [ str(i) for i in range(number_state)]
    emissions = []
    for i in range(number_state):
        emissions.append(possible_emissions)

    print(sample_tree)
    print(states)
    print(emissions)
    print(emission_observation)
    #state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states, emissions, sample_tree)

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.baumWelchRecursion(hmm, emission_observation)

    print("newparam :", newparam)

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
        monos += mono
    emissions = list(set(monos))

    #go over each gylcans
    id = '60'
    if gylcans[id].get_num_mono() > 1:
        create_and_run_treehmm(gylcans[id], 2, emissions)
