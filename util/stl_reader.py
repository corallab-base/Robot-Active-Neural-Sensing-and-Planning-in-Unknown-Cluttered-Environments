#
# file   stl_reader.py
# brief  read vertices and faces info from .stl file
# author Hanwen Ren -- ren221@purdue.edu
# date   2022-01-04
#

import numpy as np
from stl import mesh
import sys
from scipy.spatial.transform import Rotation as R

class stl_reader:

    #constructor
    def __init__(self, ifile_name):
        self.vertices_ = None
        self.faces_ = None

        mesh_info = mesh.Mesh.from_file(ifile_name)
        dicts = {}
        reverse_dicts = {}
        face_data = mesh_info.vectors
        num_face, _, _, = face_data.shape
        index = 0
        for i in range(num_face):
            for j in range(3):
                vertex = tuple(face_data[i][j])
                if vertex not in dicts:
                    dicts[vertex] = index
                    reverse_dicts[index] = vertex
                    index += 1

        #use np.array append function is less efficient
        vertices = []
        for i in range(len(reverse_dicts)):
            vertices.append(reverse_dicts[i])
        self.vertices_ = np.array(vertices)

        faces = []
        for i in range(num_face):
            face_index = []
            for j in range(3):
                face_index.append(dicts[tuple(face_data[i][j])])
            faces.append(face_index)
        self.faces_ = np.array(faces)

    #translation should be defined in cartesian space(x,y,z) dim(1,3)
    def transform(self, rotation, translation):
        if self.vertices_.any() and self.faces_.any():
            temp_trans = np.array(translation)
            for i in range(len(self.vertices_)):
                self.vertices_[i] = rotation.apply(self.vertices_[i])
                self.vertices_[i] += temp_trans
        else:
            print("Not enough data to define a mesh!")
            sys.exit(1)

    #get vertices defined by the mesh
    def get_vertices(self):
        return self.vertices_

    #get faces defined by the mesh
    def get_faces(self):
        return self.faces_

    #write out the mesh using the vertices and faces info extracted
    #used to check the performance of stl reader
    def write_to_file(self, ofile_name):
        if self.vertices_.any() and self.faces_.any():
            omesh = mesh.Mesh(np.zeros(self.faces_.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(self.faces_):
                for j in range(3):
                    omesh.vectors[i][j] = self.vertices_[f[j],:]

            omesh.save(ofile_name)
        else:
            print("Not enough data to define a mesh!")
            sys.exit(1)

if __name__ == "__main__":
    reader = stl_reader('shoulder.stl')
    reader.write_to_file('test.stl')
