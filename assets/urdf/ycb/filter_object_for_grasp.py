import os
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../../../util/')
sys.path.append(util_dir)

import obj_reader

def write_out_index(file_name, indexs):
	container = []
	with open(file_name, 'r') as f:
		start_index = 0
		lines = f.readlines()
		for ind in indexs:
			container.append(lines[ind])
	f.close()

	with open(file_name[:-4] + '_grasp.txt', 'w') as f:
		for element in container:
			f.write(element)
	f.close()


def filter():
	indexs = []
	start_index = 0
	with open('object_urdf.txt', 'r') as f:
		lines = f.readlines()
		for line in lines:
			data = line[:-1].split('/')
			ids = data[0]
			collision_data = ids + '/' + 'textured_vhacd.obj'
			obj_ins = obj_reader.obj_reader(collision_data)
			vertices = obj_ins.get_vertices()
			x_range = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
			y_range = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
			z_range = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

			if (z_range > 0.1): indexs.append(start_index)

			start_index += 1

	print (indexs)

	write_out_index('object_urdf.txt', indexs)
	write_out_index('object_offset.txt', indexs)
	write_out_index('object_collision.txt', indexs)



if __name__ == '__main__':
	filter()
