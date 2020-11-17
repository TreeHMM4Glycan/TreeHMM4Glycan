import re
import numpy as np
from typing import List

class Node(object):
    def __init__(self, id, parent_id, mono_type = None, modification = None, linkage_type = None, remaning_text = None) -> None:
        # idx in the node list
        self._id = id
        self._parent_id = parent_id
        self._modification_type = modification
        self._mono_type = mono_type
        self._linkage_type = linkage_type
        self._remaning_text = remaning_text
        self._children = []
    
    def __str__(self):
        return '{}, {}, {} ,{}, {}'.format(self._id, self._remaning_text, self._mono_type, self._modification, self._linkage_type)

    def set_mono_type(self, mono_type):
        self._mono_type = mono_type

    def get_mono_type(self):
        return self._mono_type

    def set_linkage_type(self, linkage_type):
        self._linkage_type = linkage_type

    def get_linkage_type(self):
        return self._linkage_type

    def set_modification_type(self, modification_type):
        self._modification_type = modification_type

    def get_modification_type(self):
        return self._modification_type
    
    def add_child(self, child_node):
        self._children.append(child_node)
        
    def get_children(self) -> List:
        return self._children
        
    def get_remaning_text(self):
        return self._remaning_text

    def get_parent_id(self):
        return self._parent_id
    
    def get_id(self):
        return self._id

class Glycan(object):

    pattern = re.compile(
        r'''(?P<modification>(\(?[0-9][A-Z]\)?)*)
            (?P<base_type>([a-zA-Z]{3}[0-9]?[a-zA-Z]{0,3}))
            (?P<linkage>-?\((?P<anomer>[ab]?)[0-9]+-[0-9]+\))?$''', re.VERBOSE)

    def __init__(self, iupac_text:str) -> None:
        self._nodes = []
        self._adj_matrix = None
        self._adj_list = {} # adj list is dict
        self.glycan_from_iupac(iupac_text)

    def get_adj_list(self):
        return self._adj_list

    def get_root(self):
        return self._nodes[0]
    
    def get_emssions(self):
        return [node.get_mono_type() for node in self._nodes]

    def get_emssions_with_linkage(self):
        return ['{} {}{}'.format(node.get_linkage_type(),node.get_modification_type(),node.get_mono_type()) for node in self._nodes]

    def get_adj_matrix(self):
        adj_matrix = np.zeros((len(self._nodes),len(self._nodes))).astype(int)
        for from_id in self._adj_list:
            for to_id in self._adj_list[from_id]:
                adj_matrix[from_id, to_id] = 1

        return adj_matrix
    
    def _add_node(self, current_node, next_node_remaning_text, node_stack, is_branch = False):
        next_node_idx = len(self._nodes)
        current_node_parent_id = current_node.get_parent_id() if is_branch else current_node.get_id()
        branch_out_node = Node(next_node_idx, current_node_parent_id, remaning_text = next_node_remaning_text)
        # add to glacn nodes
        self._nodes.append(branch_out_node)
        # add this node as a child to current node
        current_node.add_child(branch_out_node)
        # add to stack 
        node_stack.append(branch_out_node)
        # update adj_list

        if current_node_parent_id not in self._adj_list:
            self._adj_list[current_node_parent_id] = []
        self._adj_list[current_node_parent_id].append(next_node_idx)

    def glycan_from_iupac(self, iupac_text:str):
        #iupac_text = re.sub(r"\((\d*|\?)->?$", "", iupac_text)
        #new_branch_open = re.compile(r"(\]-?)$")
        root = Node(0, -1, remaning_text = iupac_text)
        self._nodes.append(root)
        node_stack = [root]

        while len(node_stack) > 0: 
            current_node = node_stack.pop()
            current_iupac_text = current_node.get_remaning_text()

            #print('current_iupac_text', current_iupac_text)
            # check if we have a new branch
            branch_out_texts = []
            while current_iupac_text[-1] == ']':
                # find macthing brackets
                count = 0
                branch_left = 0
                for idx, char in enumerate(current_iupac_text[::-1]):
                    count += 1 if char == ']' else 0
                    count -= 1 if char == '[' else 0
                    if count == 0:
                        branch_left = len(current_iupac_text) - idx
                        break
                branch_text = current_iupac_text[branch_left:-1]
                branch_out_texts.append(branch_text)
                
                # get current_inpuac_text for current branch
                current_iupac_text = current_iupac_text[:branch_left-1] if branch_left-1 > 0 else ''
            
            if not current_iupac_text == '':
                # extact node and edge info from text
                match = self.pattern.search(current_iupac_text)
                if match:
                    mono_type = match.group('base_type') if match.group('base_type') else ''
                    current_node.set_mono_type(mono_type)
                    linkage_type = match.group('linkage') if match.group('linkage') else ''
                    current_node.set_linkage_type(linkage_type)
                    modification_type = match.group('modification') if match.group('modification') else ''
                    current_node.set_modification_type(modification_type)

                    #print('mono: {}, modification: {}, linkage: {} '. format(mono_type, modification_type, linkage_type))
                    remaining_text = current_iupac_text[:match.start()]
                    if len(remaining_text) > 0:
                        #print('add to the stack:' + remaining_text)
                        # create next node and added as a child of current node
                        self._add_node(current_node, remaining_text, node_stack, False)
            
            # add others branchs
            for branch_out_text in branch_out_texts:
                # create branch out node and added as a child of current node
                self._add_node(current_node, branch_out_text, node_stack, True)
        
    def get_num_mono(self):
        return len(self._nodes)