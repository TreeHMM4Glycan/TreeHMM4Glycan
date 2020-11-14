import re
import numpy as np
from typing import List

class Glycan(object):
    def __init__(self) -> None:
        self.nodes = None
        self.adj_matrix = None
        self.root = None
    
    def get_root(self):
        return self.root

class Node(object):
    def __init__(self, mono_type = None, modification = None, linkage_type = None, remaning_text = None) -> None:
        #self._node_id = node_id
        self._modification = modification
        self._mono_type = mono_type
        self._linkage_type = linkage_type
        self._remaning_text = remaning_text
        self._children = []
    
    def set_mono_type(self, mono_type):
        self._mono_type = mono_type

    def set_linkage_type(self, linkage_type):
        self._linkage_type = linkage_type

    def set_modification_type(self, modification_type):
        self._linkage_type = modification_type

    def add_child(self, child_node):
        self._children.append(child_node)
    
    def get_children(self) -> List:
        return self._children
    
    def get_remaning_text(self):
        return self._remaning_text

    def get_parent(self):
        return self._parent

def glycan_from_iupac(iupac_text:str):

    pattern = re.compile(
        r'''(?P<modification>([\(]+[a-zA-Z0-9_\-,]*[\)]+)*)
            (?P<base_type>(?:[A-Z][a-z]{2}?|(?:[a-z]{3}[A-Z][a-z]{2})))
            #(?P<substituent>[^-]*?)
            #(?P<derivatization>\^[^\s-]*?)?
            (?P<linkage>-?\((?P<anomer>[ab?o]?)[0-9?/]+->?[0-9?/]+\)-?)?$''', re.VERBOSE)

    #iupac_text = re.sub(r"\((\d*|\?)->?$", "", iupac_text)
    #new_branch_open = re.compile(r"(\]-?)$")

    root = Node(remaning_text = iupac_text)
    node_stack = [root]
    #text_stack = [iupac_text]

    while len(node_stack) > 0: 
        current_node = node_stack.pop()
        current_iupac_text = current_node.get_remaning_text()

        print('current_iupac_text', current_iupac_text)
        # check if we have a new branch
        if current_iupac_text[-1] == ']':
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
            # create branch out node and added as a child of current node
            branch_out_node = Node(remaning_text = branch_text)
            current_node.add_child(branch_out_node)

            node_stack.append(branch_out_node)
            # get current_inpuac_text for current branch
            current_iupac_text = current_iupac_text[:branch_left-1] if branch_left-1 > 0 else ''
            if current_iupac_text == '':
                break 

        # extact node and edge info from text
        match = pattern.search(current_iupac_text)
        if match:
            mono_type = match.group('base_type')
            current_node.set_mono_type(mono_type)
            linkage_type = match.group('linkage')
            current_node.set_linkage_type(linkage_type)
            modification_type = match.group('modification')
            current_node.set_modification_type(modification_type)
            
            print('mono: {}, modification: {}, linkage: {} '. format(mono_type, modification_type, linkage_type))
            remaining_text = current_iupac_text[:match.start()]
            if len(remaining_text) > 0:
                # create next node and added as a child of current node
                next_node = Node(remaning_text = remaining_text)
                node_stack.append(next_node)
                current_node.add_child(next_node)

glycan_from_iupac('(3S)Gal(b1-4)[Fuc(a1-3)]Glc')