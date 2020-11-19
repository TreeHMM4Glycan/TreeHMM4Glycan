from re import VERBOSE
from typing import Dict, List
from treehmm import initHMM, baumWelch
from TreeHMM4Glycan.Glycan import Glycan
import csv
from scipy.linalg import block_diag
import copy
import re

#
# Method read a file and then retunr a Dict of iupac names
#
def get_iupcas(iupac_name_file:str) -> Dict[int, str]:
    iupacs = {}
    with open(iupac_name_file) as file_in:
        csv_reader = csv.reader(file_in)
        for idx,row in enumerate(csv_reader):
            if idx == 0:
                continue
            id = int(row[0])
            # remove right most part'(a1-sp14'
            iupac = re.split(r"\([^\)]*$", row[1], 1)[0]
            count = int(row[2])
            if count > 0:
                iupacs[id] = iupac
    return iupacs


# Method create foreset iputs from a dict collection of glycans
# Input:
#   glycans_dict: dict of glycans
# Return:
#   joint_adj_matrix - joint adjcent matrix 
#   joint_monosaccharide_emissions - joint emissions for monosaccharides types
#   joint_linkage_emissions - joint emissions for linkage types
def create_forest_inputs(glycans_dict:Dict[int, Glycan]):
    adj_matrices = []
    joint_monosaccharide_emissions = []
    joint_linkage_emissions = []
    for id in glycans_dict:
        glyan = glycans_dict[id]
        # update list of adj_matrices we are going to join along the diagonal
        adj_matrices.append(glyan.get_adj_matrix())
        # update joint_monosaccharide_emissions and joint_linkage_emissions
        joint_monosaccharide_emissions = joint_monosaccharide_emissions + glyan.get_monosaccharide_emssions()
        joint_linkage_emissions = joint_linkage_emissions + glyan.get_linkage_emssions()
    # join adj_martrix along diagonal
    joint_adj_matrix = block_diag(*adj_matrices)
    return joint_adj_matrix, joint_monosaccharide_emissions, joint_linkage_emissions

def create_and_train_treehmm(joint_adj_matrix, joint_emissions, number_state, possible_emissions):
    # TODO : add random init for both state trans matrix and emission matrix
    sample_tree = joint_adj_matrix

    # Declaring the emission_observation list
    emission_observation = [joint_emissions]
    
    #states = ['P','N']
    # create states
    states = [ str(i) for i in range(number_state)]
    emissions = [possible_emissions]

    #print(sample_tree)
    #print(states)
    #print(init_emission_matrix)
    #print(emission_observation)
    
    #state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states, emissions, sample_tree)

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.hmm_train_and_test(hmm, emission_observation)
    #newparam = baumWelch.baumWelchRecursion(hmm, emission_observation)

    print(newparam["Emission_Matrix"][0])
    print(newparam["Emission_Matrix"][0].to_numpy())
    return newparam["Transition_Matrix"], newparam["Emission_Matrix"][0].to_numpy()

if __name__ == "__main__":
    #pass
    #example1()
    
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
            #print(gylcans[id].get_monosaccharide_emssions())
            if mono_count not in counts:
                counts[mono_count] = 0
            counts[mono_count] += 1
    emissions = list(set(monos))

    #print(counts)
    #go over each gylcans
    trans_matrix = None
    emission_matrix = None
    create_forest_inputs(gylcans)
    #len(emissions)
    #print(emissions)
    #trans_matrix, emission_matrix = create_and_run_treehmm_for_one_glycan2(gylcans['21'], 10, emissions, trans_matrix)
    #trans_matrix, emission_matrix = create_and_run_treehmm_for_one_glycan(gylcans['21'], 2, emissions, trans_matrix, emission_matrix)
    #print(trans_matrix)
    #print(emission_matrix)
    
    '''
    for id in gylcans:
        if gylcans[id].get_num_nosaccharides() > 1 and id > 50:
            print('Process: {}'.format(id))
            trans_matrix, emission_matrix = create_and_run_treehmm_for_one_glycan(gylcans[id], 4, emissions, trans_matrix, emission_matrix)
            print(trans_matrix)
            print(emission_matrix)
    '''