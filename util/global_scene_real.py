import os
import sys

import numpy as np
import open3d as o3d

file_dir = os.path.dirname(__file__) 
root_dir = os.path.join(file_dir, '..')
sys.path.append(root_dir)
from camera_module.camera_view_real import camera_real

class global_scene_real:

    def __init__(self, length, width, height, offset, x_limit, y_limit, z_limit, ground_height):
        self.dim_x_ = round(length / 0.01 + 1)
        self.dim_y_ = round(width / 0.01 + 1)
        self.dim_z_ = round(height / 0.01 + 1)

        self.x_limit_ = round(x_limit / 0.01 + 1)
        self.y_limit_ = round(y_limit / 0.01 + 1)
        self.z_limit_ = round(z_limit / 0.01 + 1)
        self.g_height_ = round(ground_height / 0.01)
        self.y_left_ = int((self.dim_y_ - self.y_limit_)/2)

        self.offset_ = offset
        self.scene_ = np.full((self.dim_x_ * self.dim_y_ * self.dim_z_), -1).reshape(self.dim_x_, self.dim_y_, self.dim_z_, 1)
        for i in range(min(self.x_limit_, self.dim_x_)):
            for j in range(self.y_left_, min(self.y_left_ + self.y_limit_, self.dim_y_)):
                for k in range(self.g_height_, min(self.g_height_ + self.z_limit_, self.dim_z_)):
                    self.scene_[i][j][k] = 0




    def register_camera_view(self, camera_rotation, camera_translation, depth_image, object_dict, file_prefix):
        scene_name = file_prefix + '_scene.npy'
        camera_name = file_prefix + '_camera.npy'
        with open(scene_name, 'wb') as f:
            np.save(f, self.scene_)
        

        for i in range(min(self.x_limit_, self.dim_x_)):
            for j in range(self.y_left_, min(self.y_left_ + self.y_limit_, self.dim_y_)):
                for k in range(self.g_height_, min(self.g_height_ + self.z_limit_, self.dim_z_)):
                    if self.scene_[i][j][k] == 2:
                        self.scene_[i][j][k] = 0


        #labeling
        #seen-object - 3, unseen-completion - 2, seen-empty - 1
        for ids, object_handler in object_dict.items():
            seen_data = object_handler.get_seen()
            comp_data = object_handler.get_completion()
            
            for loc_x, loc_y, loc_z in seen_data:
                index_x = round((loc_x - self.offset_[0])/0.01)
                index_y = round((loc_y - self.offset_[1])/0.01)
                index_z = round((loc_z - self.offset_[2])/0.01)
                if 0 <= index_x < min(self.x_limit_, self.dim_x_) and \
                   self.y_left_ <= index_y < min(self.y_left_ + self.y_limit_, self.dim_y_) and \
                   self.g_height_ <= index_z < min(self.g_height_ + self.z_limit_, self.dim_z_):
                       self.scene_[index_x][index_y][index_z] = 3
            for loc_x, loc_y, loc_z in comp_data:
                index_x = round((loc_x - self.offset_[0])/0.01)
                index_y = round((loc_y - self.offset_[1])/0.01)
                index_z = round((loc_z - self.offset_[2])/0.01)
                if 0 <= index_x < min(self.x_limit_, self.dim_x_) and \
                   self.y_left_ <= index_y < min(self.y_left_ + self.y_limit_, self.dim_y_) and \
                   self.g_height_ <= index_z < min(self.g_height_ + self.z_limit_, self.dim_z_):
                       if self.scene_[index_x][index_y][index_z] < 2:
                           self.scene_[index_x][index_y][index_z] = 2 

        cam = camera_real(camera_translation, camera_rotation)
        cam.build()

        for i in range(min(self.x_limit_, self.dim_x_)):
            for j in range(self.y_left_, min(self.y_left_ + self.y_limit_, self.dim_y_)):
                for k in range(self.g_height_, min(self.g_height_ + self.z_limit_, self.dim_z_)):
                    target_location = np.array([i*0.01, j*0.01, k*0.01]) + self.offset_
                    flag, depth, location = cam.inside_frame(target_location)
                    if flag:
                        loc_x, loc_y = location
                        loc_x += 641
                        loc_y += 352
                        if 0 <= loc_x < 1280 and 0 <= loc_y < 720:
                            if depth * 1000 - np.asarray(depth_image[loc_y][loc_x]) < 5 and \
                               self.scene_[i][j][k] < 1:
                                self.scene_[i][j][k] = 1

        score = 0
        for i in range(min(self.x_limit_, self.dim_x_)):
            for j in range(self.y_left_, min(self.y_left_ + self.y_limit_, self.dim_y_)):
                for k in range(self.g_height_, min(self.g_height_ + self.z_limit_, self.dim_z_)):
                    if self.scene_[i][j][k] > 0:
                        score += 1
        score = (score * 1.0 / (self.x_limit_ * self.y_limit_ * self.z_limit_))

        camera_data = camera_rotation + camera_translation + [score]
        camera_data = np.array(camera_data)
        print (score)
        with open(camera_name, 'wb') as f:
            np.save(f, camera_data)
        return score

    def get_surface_collision_mesh(self):
        neighbor = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i != 0 or j != 0 or k != 0:
                        neighbor.append([i,j,k])
        points = []
        for i in range(self.dim_x_):
            for j in range(self.dim_y_):
                for k in range(self.dim_z_):
                    if self.scene_[i][j][k] <= 0:
                        counter = 0
                        flag = False
                        for di, dj, dk in neighbor:
                            new_i = i + di
                            new_j = j + di
                            new_k = k + di
                            if 0 <= new_i < self.dim_x_ and \
                               0 <= new_j < self.dim_y_ and \
                               0 <= new_k < self.dim_z_:
                                   counter += 1
                                   if self.scene_[new_i][new_j][new_k] > 0:
                                       flag = True
                                       break
                        if counter < 26 or flag:
                            points.append([i*0.01 + self.offset_[0], 
                                           j*0.01 + self.offset_[1], 
                                           k*0.01 + self.offset_[2]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.uniform_down_sample(1)
        pcd.estimate_normals()
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        return mesh





    def vis_scene(self, prefix):
        arr = []
        color = []
        for i in range(self.dim_x_):
            for j in range(self.dim_y_):
                for k in range(self.dim_z_):
                    flag = self.scene_[i][j][k]
                    target_location = np.array([i*0.01, j*0.01, k*0.01]) + self.offset_
                    if flag > 0:
                        arr.append(target_location)
                        color.append([1, 0, 0])
                    #arr.append(target_location)
                    #if flag == 0:
                    #    color.append([0.9, 0.9, 0.9])
                    #elif flag == 1:
                    #    color.append([1, 0, 0])
                    #elif flag == 2:
                    #    color.append([1, 1, 1])
                    #else:
                    #    color.append([0, 0, 1])

        environment_pc = []
        environment_colors = []
        for i in range(min(self.x_limit_, self.dim_x_)):
            for j in range(self.y_left_, min(self.y_left_ + self.y_limit_, self.dim_y_)):
                environment_pc.append(np.array([i*0.01, j*0.01, self.g_height_*0.01]) + self.offset_)
                environment_pc.append(np.array([i*0.01, j*0.01, (min(self.g_height_ + self.z_limit_, self.dim_z_)-1)*0.01]) + self.offset_)
                environment_colors.append([0.5, 0.5, 0.5])
                environment_colors.append([0.5, 0.5, 0.5])

        for i in range(min(self.x_limit_, self.dim_x_)):
            for k in range(self.g_height_, min(self.g_height_ + self.z_limit_, self.dim_z_)):
                environment_pc.append(np.array([i*0.01, self.y_left_*0.01, k*0.01]) + self.offset_)
                environment_pc.append(np.array([i*0.01, (min(self.y_left_ + self.y_limit_, self.dim_y_)-1)*0.01, k*0.01]) + self.offset_)
                environment_colors.append([0.5, 0.5, 0.5])
                environment_colors.append([0.5, 0.5, 0.5])


        env_pc = o3d.geometry.PointCloud()
        env_pc.points = o3d.utility.Vector3dVector(environment_pc)
        env_pc.colors = o3d.utility.Vector3dVector(environment_colors)

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.array(arr))
        scene_pcd.colors = o3d.utility.Vector3dVector(np.array(color))

        with open(prefix + '_scene_pcd.npy', 'wb') as f:
            np.save(f, np.array(arr))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.4, origin = [0, 0, 0])
        o3d.visualization.draw_geometries([scene_pcd, mesh_frame, env_pc])


    def visualize_scene(self, data_file):
        self.scene_ = np.load(data_file)
        self.dim_x_, self.dim_y_, self.dim_z_, _ = self.scene_.shape
        print (self.scene_.shape)
        arr = []
        color = []
        for i in range(self.dim_x_):
            for j in range(self.dim_y_):
                for k in range(self.dim_z_):
                    flag = self.scene_[i][j][k]
                    target_location = np.array([i*0.01, j*0.01, k*0.01]) + self.offset_
                    arr.append(target_location)
                    if flag == 0:
                        color.append([0.9, 0.9, 0.9])
                    elif flag == 1:
                        color.append([1, 0, 0])
                    elif flag == 2:
                        color.append([1, 0, 0])
                    else:
                        color.append([1, 0, 0])

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.array(arr))
        scene_pcd.colors = o3d.utility.Vector3dVector(np.array(color))

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.4, origin = [0, 0, 0])
        o3d.visualization.draw_geometries([scene_pcd, mesh_frame])



if __name__ == '__main__':
    test_scene = global_scene(1.0, 1.5, 0.5, np.array([0.1, -0.58, 0.05]))
    test_scene.visualize_scene('../sim/scene_generation_final/env1_sequence4Rscene.npy')
    camera_info = np.load('../sim/scene_generation/env10_sequence3_camera.npy')
    print (camera_info)
    #depth_image = o3d.io.read_image('../sim/data_generation/depth_test0_0.png')
    #test_scene.get_camera_view([0.0, -0.4900, 0.5306, -0.6916], [0.2959, 0.1362, 0.4913], depth_image)
    #depth_image = o3d.io.read_image('../sim/data_generation/depth_test1_0.png')
    #test_scene.get_camera_view([0.0, 0.5219, 0.4823, 0.7035], [0.7084, 0.1242, 0.4423], depth_image)
    #depth_image = o3d.io.read_image('../sim/data_generation/depth_test8_0.png')
    #test_scene.get_camera_view([0.0, -0.2666, 0.2660, -0.9264], [0.4371, 0.3841, 0.3966], depth_image)
    #depth_image = o3d.io.read_image('../sim/data_generation/depth_test10_0.png')
    #test_scene.get_camera_view([0.0, 0.3675, -0.4891, 0.7910], [0.2708, -0.1053, 0.3027], depth_image)
    #test_scene.visualize_scene()

