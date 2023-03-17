import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv

def cast_index_to_real(i, j, k, unit, offset):
    digit_location = np.array([i,j,k], dtype = np.float32) * unit
    center = np.array([0.005, 0.005, 0.005])
    return digit_location + offset + center

def cast_real_to_index(digit_location, unit, offset):
    center = np.array([0.005, 0.005, 0.005])
    shifted_location = (digit_location - center - offset)/unit
    return (round(shifted_location[0]), round(shifted_location[1]), round(shifted_location[2]))

def cast_pc_to_index(pc):
    center = np.array([0.0005, 0.0005, 0.0005])
    pc = (digit_location - center)/0.001

def cast_index_to_pc(index):
    center = np.array([0.0005, 0.0005, 0.0005])
    return index * 0.001 + center



class YCB_object:
    def __init__(self):
        self.seen_ = {}
        self.completion_ = {}
        self.pcd_ = None
        self.center_ = None

    def add_seen(self, seen):
        for key,value in seen.items():
            if key not in self.seen_:
                self.seen_[key] = value

        points = [list(x) for x in self.seen_.keys()]
        points = np.array(points)

        self.center_ = np.mean(points, axis = 0)

        self.pcd_ = o3d.geometry.PointCloud()
        self.pcd_.points = o3d.utility.Vector3dVector(points)


    def add_completion(self, comp):
        self.completion_ = {}
        average_colors = np.array([list(x) for x in self.seen_.values()])
        average_colors = np.mean(average_colors, axis = 0)
        for element in comp:
            if tuple(element) not in self.seen_:
                if tuple(element) not in self.completion_:
                    self.completion_[tuple(element)] = list(average_colors)

    def get_seen(self):
        return self.seen_

    def get_completion(self):
        return self.completion_

    def is_part_of_current_pc(self, new_pc):
        dists = self.pcd_.compute_point_cloud_distance(new_pc)
        flag = False
        for dis in dists:
            if dis <= 1e-3:
                flag = True
                break
        return flag

    def get_distance_between_center(self, new_center):
        return np.sum((self.center_ - new_center)**2)

    def get_collision_mesh(self):
        points = [list(x) for x in self.seen_.keys()]
        points += [list(x) for x in self.completion_.keys()]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        aabb = pcd.get_axis_aligned_bounding_box()

        all_points = {}
        x_min, x_max = sys.maxsize, -sys.maxsize
        y_min, y_max = sys.maxsize, -sys.maxsize
        z_min, z_max = sys.maxsize, -sys.maxsize
        ind = 0
        for points in aabb.get_box_points():
            temp_x, temp_y, temp_z = points
            temp_x = round(temp_x, 3)
            temp_y = round(temp_y, 3)
            temp_z = round(temp_z, 3)
            x_min = min(x_min, temp_x)
            x_max = max(x_max, temp_x)
            y_min = min(y_min, temp_y)
            y_max = max(y_max, temp_y)
            z_min = min(z_min, temp_z)
            z_max = max(z_max, temp_z)
            all_points[ind] = [temp_x, temp_y, temp_z]
            ind += 1

        vertices = []
        faces = []
        limits = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        for t in range(3):
            group1, group2 = [], []
            maxi, mini = limits[t]
            for key,value in all_points.items():
                if value[t] == maxi:
                    group1.append(key)
                else:
                    group2.append(key)

            faces.append([group1[0], group1[1], group1[2]])
            faces.append([group1[0], group1[1], group1[3]])
            faces.append([group1[0], group1[2], group1[3]])
            faces.append([group1[1], group1[2], group1[3]])
            faces.append([group2[0], group2[1], group2[2]])
            faces.append([group2[0], group2[1], group2[3]])
            faces.append([group2[0], group2[2], group2[3]])
            faces.append([group2[1], group2[2], group2[3]])

        for t in range(8):
            vertices.append(all_points[t])

        return np.array(vertices), np.array(faces)






        #pcd = pcd.uniform_down_sample(1)
        #pcd.estimate_normals()
        #radii = [0.005, 0.01]
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        #return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        #points = pv.PolyData(points)
        #surf = points.reconstruct_surface()

        #mesh_vertices = np.asarray(surf.points)

        #mesh_faces = []

        #start = 0

        #pl = pv.Plotter(shape = (1,2))
        #pl.add_mesh(points)
        #pl.add_title("PC")
        #pl.subplot(0, 1)
        #pl.add_mesh(surf, color = True, show_edges = True)
        #pl.add_title("Mesh")
        #pl.show()

        #while start < len(surf.faces):
        #    num = surf.faces[start]
        #    mesh_faces.append(surf.faces[start+1 : start + 1 + num])
        #    start += (num + 1)
        #mesh_faces = np.asarray(mesh_faces)

        #return mesh_vertices, mesh_faces
        
