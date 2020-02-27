from treelib import Node, Tree
import os, sys
from joblib import *
from pandas import read_csv, DataFrame
import numpy as np
import pickle5 as pickle
#from ete3 import NCBITaxa

# new_tree = Tree(tree.subtree(tree.root), deep=True)

class species_tree(Tree):


	def init_nodes_data(self, value = 0):
		for id in self.expand_tree(mode=1):
			tree[id].data = value
		print('succeed !')

	def from_paths(self, paths: list):
		# check duplicated son-fathur relationship
		stree = species_tree()
		stree.root = 'root'
		for path in paths:
			current_node = stree.root
			for nid in path:
				children_ids = [n.identifier for n in stree.children(current_node)]
				if nid not in children_ids: stree.create_node(nid, parent=current_node)
				current_node = nid
		return stree

	def from_child_father(self, ):
		return None
	
	def from_pickle(self, file: str):
		with open(file, 'rb') as f: 
			stree = pickle.load(f)
		return stree

	def path_to_node(self, node_id: str):
		nid = node_id
		path_r = []
		while nid != 'root':
			path_r.append(nid)
			nid = self[nid].bpointer
		path_r.append('root')
		path_r.reverse()
		return path_r
	
	def fill_with(self, data: dict):
		for nid, val in data.items():
			self[nid].data = val

	def update_value(self, ):
		all_nodes = [nid for nid in self.expand_tree(mode=2)][::-1]
		for nid in all_nodes:
			d = sum([node.data for node in self.children(nid)])
			self[nid].data = self[nid].data + d

	def to_pickle(self, file: str):
		with open(file, 'wb') as f:
			pickle.dump(self, f)

	def get_matrix(self, dtype: object=np.float32):
		paths_to_leaves = self.paths_to_leaves()
		ncol = self.depth()
		nrow = len(paths_to_leaves)
		Matrix = np.zeros(ncol*nrow, dtype=dtype).reshape(nrow, ncol)

		# for each row in matrix
		for row, path in enumerate(paths_to_leaves):		
			# for each element in row
			for col, nid in enumerate(path):
				Matrix[row, col]= tree[nid].data
		return Matrix
		
	def to_matrix_npy(self, file: str, dtype: object=np.float32 ):
		matrix = self.get_matrix(self, dtype=dtype)
		np.save(file, matrix)

	def copy(self, ):
		return species_tree(self.subtree(self.root), deep=True)

	def remove_levels(self, level: int):
		# print('this will remove ')
		for nid in self.expand_tree(mode=1):
			if self.level(nid) == level:
				self.remove_node(nid)  # check

	def save_paths_to_csv(self, file: str, fill_na=True):
		paths = self.paths_to_leaves()
		df = pd.DataFrame(paths)
		if fill_na:
			df.fillna('')
		df.to_csv(file, sep = ',')

	def from_paths_csv(self, file: str):
		df = read_csv(file, header=0, sep=',')
		def remove_null_str(x): 
			while '' in x: x.remove('')
			return x
		paths = map(remove_null_str, [list(df.iloc(0)[i]) for i in df.axes[0]])
		return self.from_paths(paths)

	def from_ete3_species(self, name: str):
		'''
		return a subtree of species name, data is retrived from NCBI Taxonomy database
		'''
		return None




