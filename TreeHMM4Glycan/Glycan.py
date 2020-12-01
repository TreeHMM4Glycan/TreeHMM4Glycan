import re
import numpy as np
from typing import List, Dict

class Node(object):
    def __init__(self, id:int, parent_id:int, monosaccharide_type:str = None, modification:str = None, linkage_type:str = None, remaning_text:str = None) -> None:
        # idx in the node list
        self._id:int = id
        self._parent_id:int = parent_id
        self._modification_type:str = modification
        self._monosaccharide_type:str = monosaccharide_type
        self._linkage_type:str = linkage_type
        self._remaning_text:str = remaning_text
        self._children:List['Node'] = []
    
    def __str__(self):
        return '{}, {}, {} ,{}, {}'.format(self._id, self._remaning_text, self._monosaccharide_type, self._modification, self._linkage_type)

    def set_monosaccharide_type(self, monosaccharide_type:str):
        self._monosaccharide_type = monosaccharide_type

    def get_monosaccharide_type(self) -> str:
        return self._monosaccharide_type

    def set_linkage_type(self, linkage_type:str):
        self._linkage_type = linkage_type

    def get_linkage_type(self) -> str:
        return self._linkage_type

    def set_modification_type(self, modification_type:str):
        self._modification_type = modification_type

    def get_modification_type(self) -> str:
        return self._modification_type
    
    def add_child(self, child_node:'Node'):
        self._children.append(child_node)
        
    def get_children(self) -> List['Node']:
        return self._children
        
    def get_remaning_text(self) -> str:
        return self._remaning_text

    def get_parent_id(self) -> int:
        return self._parent_id
    
    def get_id(self) -> int:
        return self._id

class Glycan(object):

    pattern = re.compile(
        r'''(?P<modification>(\(?[0-9][A-Z]\)?)*)
            (?P<base_type>([a-zA-Z]{3}[0-9]?[a-zA-Z]{0,3}))
            (?P<linkage>-?\((?P<anomer>[ab]?)[0-9]+-[0-9]+\))?$''', re.VERBOSE)

    def __init__(self, iupac_text:str, single_end = True) -> None:
        self._nodes:List[Node] = []
        self._adj_matrix:np.ndarray = None
        self._adj_list:Dict[int, List[int]] = {} # adj list is dict
        self._end_indices = []
        self.glycan_from_iupac(iupac_text, single_end)
        

    # methods add an end node to the tree
    # end of each branch are linked to the same end node
    # because this tree is dircted, this wil create a polytree
    def _add_end_node(self, root:Node):
        end_node_id = len(self._nodes)
        # use -1 as parent id is not used in thi stage 
        # and we can have more then one parent for end node
        end_node = Node(end_node_id, -1, monosaccharide_type= 'End', linkage_type= 'End-Linkage')
        self._nodes.append(end_node)
        self._add_end_node_recursive(root, end_node)
        self._end_indices.append(end_node_id)

    def _add_end_node_recursive (self, node:Node, end_node:Node):
        if len(node.get_children()) == 0:
            node.add_child(end_node)
            self._update_adj_list_after_add_node(node.get_id(), end_node.get_id())
        else:
            for child in node.get_children():
                self._add_end_node_recursive(child, end_node)

    # methods add an end node to each end
    def _add_end_nodes(self, node:Node):
        if len(node.get_children()) == 0:
            end_node_id = len(self._nodes)
            end_node = Node(end_node_id, node.get_id(), monosaccharide_type= 'End', linkage_type= 'End-Linkage')
            node.add_child(end_node)
            self._nodes.append(end_node)
            self._end_indices.append(end_node_id)
            self._update_adj_list_after_add_node(node.get_id(), end_node_id)
        else:
            for child in node.get_children():
                self._add_end_nodes(child)

    def get_adj_list(self) -> Dict[int, List[int]]:
        return self._adj_list

    def get_root(self) -> Node:
        return self._nodes[0]
    
    def get_monosaccharide_emssions(self) -> List[str]:
        return [node.get_monosaccharide_type() for node in self._nodes]

    def get_linkage_emssions(self) -> List[str]:
        return [node.get_linkage_type() for node in self._nodes]

    def get_modification_emssions(self) -> List[str]:
        return [node.get_modification_type() for node in self._nodes]

    def get_filtered_monosaccharide_emssions(self)-> List[str]:
        emissions = self.get_monosaccharide_emssions()
        for idx, item in enumerate(emissions):
            if item in ['Neu5Gc','KDN','GlcA']:
                emissions[idx] = 'Other'
        return emissions

    def get_filtered_linkage_emssions(self)-> List[str]:
        emissions = self.get_linkage_emssions()
        for idx, item in enumerate(emissions):
            if item in ['(b2-6)','(a2-8)']:
                emissions[idx] = 'Other'
        return emissions

    #def get_monosaccharide_emssions_with_linkage(self):
    #    return ['{} {}{}'.format(node.get_linkage_type(),node.get_modification_type(),node.get_monosaccharide_type()) for node in self._nodes]

    def get_adj_matrix(self) -> np.ndarray:
        adj_matrix = np.zeros((len(self._nodes),len(self._nodes))).astype(int)
        for from_id in self._adj_list:
            for to_id in self._adj_list[from_id]:
                adj_matrix[from_id, to_id] = 1

        return adj_matrix
    
    def get_end_nodes_indices(self) -> List[int]:
        return self._end_indices
    
    def _update_adj_list_after_add_node(self, parent_id, next_node_idx):
        if parent_id not in self._adj_list:
            self._adj_list[parent_id] = []
        self._adj_list[parent_id].append(next_node_idx)

    def _glycan_from_iupac_add_node(self, current_node, next_node_remaning_text, node_stack, is_branch = False):
        next_node_id = len(self._nodes)
        parent_id = current_node.get_parent_id() if is_branch else current_node.get_id()
        new_node = Node(next_node_id, parent_id, remaning_text = next_node_remaning_text)
        # add to glacn nodes
        self._nodes.append(new_node)
        # add this node as a child to current node
        current_node.add_child(new_node)
        # add to stack 
        node_stack.append(new_node)
        # update adj_list
        self._update_adj_list_after_add_node(parent_id, next_node_id)


    def glycan_from_iupac(self, iupac_text:str, single_end:bool = True):
        #iupac_text = re.sub(r"\((\d*|\?)->?$", "", iupac_text)
        #new_branch_open = re.compile(r"(\]-?)$")
        root = Node(0, -1, remaning_text = iupac_text)
        root.set_linkage_type('Root-Linkage')
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
                    monosaccharide_type = match.group('base_type') if match.group('base_type') else ''
                    current_node.set_monosaccharide_type(monosaccharide_type)
                    linkage_type = match.group('linkage') if match.group('linkage') else ''
                    current_node.set_linkage_type(linkage_type)
                    modification_type = match.group('modification') if match.group('modification') else ''
                    current_node.set_modification_type(modification_type)

                    #print('mono: {}, modification: {}, linkage: {} '. format(monosaccharide_type, modification_type, linkage_type))
                    remaining_text = current_iupac_text[:match.start()]
                    if len(remaining_text) > 0:
                        #print('add to the stack:' + remaining_text)
                        # create next node and added as a child of current node
                        self._glycan_from_iupac_add_node(current_node, remaining_text, node_stack, False)
            
            # add others branchs
            for branch_out_text in branch_out_texts:
                # create branch out node and added as a child of current node
                self._glycan_from_iupac_add_node(current_node, branch_out_text, node_stack, True)
        
        if single_end:
            # add end node and void arc
            self._add_end_node(self._nodes[0])
        else:
            self._add_end_nodes(self._nodes[0])

    def get_num_nosaccharides(self):
        return len(self._nodes)

if __name__ == "__main__":
    test_input =  'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)]Man(b1-4)GlcNAc(b1-4)GlcNAc'
    glycan = Glycan(test_input)
    print(glycan.get_filtered_monosaccharide_emssions())
    print(glycan.get_monosaccharide_emssions())