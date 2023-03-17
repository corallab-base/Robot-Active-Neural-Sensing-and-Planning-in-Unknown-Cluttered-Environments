#
# file   obj_reader.py
# brief  read vertices and faces info from .obj file
# author Hanwen Ren -- ren221@purdue.edu
# date   2022-01-11
#

import sys
import numpy as np

class obj_reader:

    #constructor
    def __init__(self, ifile_name):
        self.vertices_ = None
        self.faces_ = None
        
        f = open(ifile_name)

        vertices = []
        faces = []
        for line in f:
            if line[0] == 'v':
                div = line[2:-1].split(' ')
                vertices.append([float(x) for x in div])
            elif line[0] == 'f':
                div = line[2:-1].split(' ')
                faces.append([int(x)-1 for x in div if x])
        
        self.vertices_ = np.array(vertices)
        self.faces_ = np.array(faces)
        self.current_vertices_ = np.array(vertices)

    #scale vertices
    def set_scale(self, scale):
        if self.vertices_.any() and self.faces_.any():
            for i in range(len(self.vertices_)):
                self.vertices_[i] *= scale
        else:
            print ("Not enough data to define a mesh!")
            sys.exit(1)

    def add_offset(self, offset):
        temp_offset = np.array(offset)
        if self.vertices_.any() and self.faces_.any():
            for i in range(len(self.vertices_)):
                self.vertices_[i] += temp_offset
        else:
            print ("Not enough data to define a mesh!")
            sys.exit(1)
        self.current_vertices_ = np.array(self.vertices_)


    #translate vertices
    def set_offset(self, offset):
        temp_offset = np.array(offset)
        if self.vertices_.any() and self.faces_.any():
            for i in range(len(self.vertices_)):
                self.current_vertices_[i] = self.vertices_[i] + temp_offset
        else:
            print ("Not enough data to define a mesh!")
            sys.exit(1)

    #get vertices defined by the mesh
    def get_vertices(self):
        return self.current_vertices_

    #get faces defined by the mesh
    def get_faces(self):
        return self.faces_

    def get_center(self):
        dx, dy, dz = self.get_bounding_box()
        x_min, y_min, z_min = sys.maxsize, sys.maxsize, sys.maxsize
        for x, y, z in self.current_vertices_:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            z_min = min(z_min, z)
        return np.array([x_min + dx/2, y_min + dy/2, z_min + dz/2])

    def get_bounding_box(self):
        x_min, y_min, z_min = sys.maxsize, sys.maxsize, sys.maxsize
        x_max, y_max, z_max = -sys.maxsize, -sys.maxsize, -sys.maxsize
        for x, y, z in self.current_vertices_:
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            z_min = min(z_min, z)
            z_max = max(z_max, z)
        return (x_max - x_min, y_max - y_min, z_max - z_min)

    def get_bounding_box_mesh(self):
        cx, cy, cz = self.get_center()
        dx, dy, dz = self.get_bounding_box()
        vertices = np.array([[cx - dx/2.0, cy - dy/2.0, cz - dz/2.0],
				                     [cx - dx/2.0, cy + dy/2.0, cz - dz/2.0],
														 [cx + dx/2.0, cy + dy/2.0, cz - dz/2.0],
														 [cx + dx/2.0, cy - dy/2.0, cz - dz/2.0],
                             [cx - dx/2.0, cy - dy/2.0, cz + dz/2.0],
				                     [cx - dx/2.0, cy + dy/2.0, cz + dz/2.0],
														 [cx + dx/2.0, cy + dy/2.0, cz + dz/2.0],
														 [cx + dx/2.0, cy - dy/2.0, cz + dz/2.0]])
        faces = np.array([[0, 2, 1], [0, 2, 3],
				                  [4, 6, 5], [4, 6, 7],
													[5, 2, 1], [5, 2, 6],
													[7, 2, 3], [7, 2, 6],
													[4, 3, 0], [4, 3, 7],
													[4, 1, 0], [4, 1, 5]])
        return vertices, faces


if __name__ == "__main__":
    reader = obj_reader("mug_collision.obj")
