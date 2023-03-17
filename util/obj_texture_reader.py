#
# file   obj_reader.py
# brief  read vertices and faces info from .obj file
# author Hanwen Ren -- ren221@purdue.edu
# date   2022-01-11
#

import numpy as np

class obj_texture_reader:

    #constructor
    def __init__(self, ifile_name):
        self.vertices_ = None
        
        f = open(ifile_name)

        vertices = []
        faces = []
        for line in f:
            if line[:2] == 'v ':
                div = line[2:-1].split(' ')
                vertices.append([float(x) for x in div])
        
        self.vertices_ = np.array(vertices)

    #translate vertices
    def set_offset(self, offset):
        temp_offset = np.array(offset)
        if self.vertices_.any():
            for i in range(len(self.vertices_)):
                self.vertices_[i] += temp_offset
        else:
            print ("Not enough data to define a mesh!")
            sys.exit(1)

    #get vertices defined by the mesh
    def get_vertices(self):
        return self.vertices_

if __name__ == "__main__":
    reader = obj_texture_reader('textured.obj')
