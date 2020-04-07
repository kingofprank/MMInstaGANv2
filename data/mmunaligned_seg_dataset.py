import sys
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class MMUnalignedSegDataset(BaseDataset):
	def name(self):
		return 'MMUnalignedSegDataset'

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.cat_num = opt.cat_num
		self.dir, self.cat_size, self.cat_path = list(), list(), list()
		self.max_instances = 20  # default: 20
		self.seg_dir = 'seg'  # default: 'seg'
		for i in range(opt.cat_num):
			self.dir.append(os.path.join(opt.dataroot, opt.phase + chr(i + 65)))
			self.cat_path.append(sorted(make_dataset(self.dir[i])))
			self.cat_size.append(len(self.cat_path[i]))
		self.transform = get_transform(opt)

	def fixed_transform(self, image, seed):
		random.seed(seed)
		return self.transform(image)

	def read_segs(self, seg_path, seed):
		segs = list()
		for i in range(self.max_instances):
			path = seg_path.replace('.png', '_{}.png'.format(i))
			if os.path.isfile(path):
				seg = Image.open(path).convert('L')
				seg = self.fixed_transform(seg, seed)
				segs.append(seg)
			else:
				segs.append(-torch.ones(segs[0].size()))
		return torch.cat(segs)

	def __getitem__(self, index):
		cat_index = list()
		for i in range(self.cat_num):
			if i == 0: # first as base index
				cat_index.append(index % self.cat_size[i])
			else:
				if self.opt.serial_batches:
					cat_index.append(index % self.cat_size[i])
				else:
					cat_index.append(random.randint(0, self.cat_size[i] - 1))
		# index_A = index % self.A_size
		# if self.opt.serial_batches:
		# 	index_B = index % self.B_size
		# else:
		# 	index_B = random.randint(0, self.B_size - 1)
		seed = random.randint(-sys.maxsize, sys.maxsize)
		imgs_list, segs_list = list(), list()
		for i in range(self.cat_num):
			cur_path = self.cat_path[i][cat_index[i]]
			cur_seg_path = cur_path.replace(chr(i + 65), chr(i + 65) + '_{}'.format(self.seg_dir))
			cur_idx = cur_path.split('/')[-1].split('.')[0]
			img = Image.open(cur_path).convert('RGB')
			imgs_list.append(self.fixed_transform(img, seed))
			segs_list.append(self.read_segs(cur_seg_path, seed))


		# A_path = self.A_paths[index_A]
		# B_path = self.B_paths[index_B]
		# A_seg_path = A_path.replace('A', 'A_{}'.format(self.seg_dir))
		# B_seg_path = B_path.replace('B', 'B_{}'.format(self.seg_dir))
		#
		# A_idx = A_path.split('/')[-1].split('.')[0]
		# B_idx = B_path.split('/')[-1].split('.')[0]

		# print('(A, B) = (%d, %d)' % (index_A, index_B))
		# seed = random.randint(-sys.maxsize, sys.maxsize)
		#
		# A = Image.open(A_path).convert('RGB')
		# B = Image.open(B_path).convert('RGB')
		# A = self.fixed_transform(A, seed)
		# B = self.fixed_transform(B, seed)
		#
		# A_segs = self.read_segs(A_seg_path, seed)
		# B_segs = self.read_segs(B_seg_path, seed)

		# if self.opt.direction == 'BtoA':
		# 	input_nc = self.opt.output_nc
		# 	output_nc = self.opt.input_nc
		# else:
		# 	input_nc = self.opt.input_nc
		# 	output_nc = self.opt.output_nc
		#
		# if input_nc == 1:  # RGB to gray
		# 	tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
		# 	A = tmp.unsqueeze(0)
		# if output_nc == 1:  # RGB to gray
		# 	tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
		# 	B = tmp.unsqueeze(0)

		# return {'A': A, 'B': B,
		# 		'A_idx': A_idx, 'B_idx': B_idx,
		# 		'A_segs': A_segs, 'B_segs': B_segs,
		# 		'A_paths': A_path, 'B_paths': B_path}
		return {'imgs_list': imgs_list, 'segs_list': segs_list}

	def __len__(self):
		return max(self.cat_size)
