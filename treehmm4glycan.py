from typing import Dict, List, Tuple
from treehmm import initHMM, baumWelch
from treehmm import fwd_seq_gen, forward
from TreeHMM4Glycan.Glycan import Glycan
import csv
from scipy.linalg import block_diag
import numpy as np
import re
import argparse
from scipy.sparse.csr import csr_matrix
from scipy.special import logsumexp
import logging
import sys

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

# method to get a dict of glycans form input iupac snfg
def get_glycans(iupacs:Dict[int, str] , total = None) -> Tuple[Dict[int, Glycan], List[str], List[str]]:
    gylcans_dict = {}
    monos = []
    links = []
    for id in iupacs:
        inpuac_text = iupacs[id]
        gylcan = Glycan(inpuac_text)
        
        if gylcan.get_num_nosaccharides() > 2:
            gylcans_dict[id] = gylcan
            monos += gylcan.get_filtered_monosaccharide_emssions()
            links += gylcan.get_filtered_linkage_emssions()
        
        if total is not None and len(gylcans_dict) == total:
            break

    mono_emissions = list(set(monos))
    link_emissions = list(set(links))

    return gylcans_dict, mono_emissions, link_emissions

# Method create foreset iputs from a dict collection of glycans
# Input:
#   glycans_dict: dict of glycans
# Return:
#   joint_adj_matrix - joint adjcent matrix 
#   joint_monosaccharide_emission_observations - joint emissions for monosaccharides types
#   joint_linkage_emission_observations - joint emissions for linkage types
def create_forest_inputs(glycans_dict:Dict[int, Glycan]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    adj_matrices = []
    joint_monosaccharide_emission_observations = []
    joint_linkage_emission_observations = []
    for id in glycans_dict:
        glyan = glycans_dict[id]
        # update list of adj_matrices we are going to join along the diagonal
        adj_matrices.append(glyan.get_adj_matrix())
        # update joint_monosaccharide_emission_observations and joint_linkage_emission_observations
        joint_monosaccharide_emission_observations = joint_monosaccharide_emission_observations + glyan.get_filtered_monosaccharide_emssions()
        joint_linkage_emission_observations = joint_linkage_emission_observations + glyan.get_filtered_linkage_emssions()
    # join adj_martrix along diagonal
    joint_adj_matrix = block_diag(*adj_matrices)
    return joint_adj_matrix, joint_monosaccharide_emission_observations, joint_linkage_emission_observations

# Method learn a treehmm from a forset
# Input:
#   joint_adj_matrix - joint adjcent matrix 
#   joint_emissions_observations - emissions observations for this foreset N * M, N is the number of emissions group eg,monosaccharide, linkage,
#       M is number of emssions groups eg.  monosaccharide_emissions for group 1 and linkage_emissions for group 2
#   number_state - number of states
#   possible_emissions -  possible_ emissions for this foreset N * M, N is the number of emissions group eg,monosaccharide, linkage, 
#       M is number of emssions groups   eg.  monosaccharide_emissions for group 1 and linkage_emissions for group 2
def create_and_train_treehmm(number_state:int, iupacs, include_linkage = False, max_iterations=50, delta=1e-5):

    gylcans, possible_monosaccharide_emissions, possible_linkage_emissions = get_glycans(iupacs, 100)
    
    joint_adj_matrix, joint_monosaccharide_emission_observations, joint_linkage_emission_observations = create_forest_inputs(gylcans)

    #print(possible_monosaccharide_emissions)
    # monosaccharide_emission_observations_counts = {}
    # for item in possible_monosaccharide_emissions:
    #     monosaccharide_emission_observations_counts[item] = 0
    # for item in joint_monosaccharide_emission_observations:
    #     monosaccharide_emission_observations_counts[item] += 1
    # print(monosaccharide_emission_observations_counts)

    # possible_linkage_emissions_counts = {}
    # for item in possible_linkage_emissions:
    #     possible_linkage_emissions_counts[item] = 0
    # for item in joint_linkage_emission_observations:
    #     possible_linkage_emissions_counts[item] += 1
    # print(possible_linkage_emissions_counts)

    # we only use monosaccharide
    if not include_linkage:
        joint_emissions_observations = [joint_monosaccharide_emission_observations]
        possible_emissions = [possible_monosaccharide_emissions]
    else:
    # if we want to use both of them
        joint_emissions_observations = [joint_monosaccharide_emission_observations, joint_linkage_emission_observations]
        possible_emissions = [possible_monosaccharide_emissions, possible_linkage_emissions]
        
    forest = csr_matrix(joint_adj_matrix)
    # create states
    states = [ str(i) for i in range(number_state)]
    
    #state_transition_probabilities = np.array([0.1,0.9,0.1,0.9]).reshape(2,2)
    hmm = initHMM.initHMM(states, possible_emissions)

    # The baumWelch part: To find the new parameters and result statistics
    newparam = baumWelch.hmm_train_and_test(hmm, forest, joint_emissions_observations, maxIterations = max_iterations, delta = delta)
    #newparam = baumWelch.baumWelchRecursion(hmm, emission_observation)


    hmm_trained = initHMM.initHMM(states, possible_emissions, state_transition_probabilities=newparam['hmm']['state_transition_probabilities'],
                                              emission_probabilities=newparam['hmm']['emission_probabilities'])
    
    #fwd_tree_sequence = fwd_seq_gen.forward_sequence_generator(forest)
    #bind_fwd_probs = forward.forward(hmm_trained, forest, joint_emissions_observations, fwd_tree_sequence)
    #print(joint_emissions_observations)
    
    ll = 0
    for glycan_idx in gylcans:
        glycan = gylcans[glycan_idx]
        #print(glycan.get_filtered_monosaccharide_emssions())
        glycan_tree = csr_matrix(glycan.get_adj_matrix())
        fwd_tree_sequence = fwd_seq_gen.forward_sequence_generator(glycan_tree)
        emssions = glycan.get_filtered_monosaccharide_emssions()
        if include_linkage:
            emssions += glycan.get_filtered_linkage_emssions()
            
        bind_fwd_probs = forward.forward(hmm_trained, glycan_tree, [emssions], fwd_tree_sequence)
        case_ll = logsumexp(bind_fwd_probs.iloc[:, -1])
        ll += case_ll
        
    print(ll)
    return newparam

if __name__ == "__main__":

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', help='num of hidden states', type = int, default = 2)
    parser.add_argument('--include_linkage', help='0 for no, 1 for yes', type = int, default = 0)
    args = parser.parse_args()

    num_states = args.num_states
    include_linkage = True if args.include_linkage > 0 else False

    # read iupac and get gylcans
    iupac_name_file = './Data/IUPAC.csv'
    iupacs = get_iupcas(iupac_name_file)
 
    create_and_train_treehmm(num_states, iupacs, max_iterations = 1)
    create_and_train_treehmm(num_states, iupacs, max_iterations = 10)
    create_and_train_treehmm(num_states, iupacs, max_iterations = 20)