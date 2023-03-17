#
# File:          ur5e_refactor.py
# Brief:         main program for ur5e simulation
# Author:        Hanwen Ren -- ren221@purdue.edu
# Date:          2023-03-01
# Last Modified: 2023-03-01
#

from scipy.spatial.transform import Rotation as R
import math
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from PIL import Image
import numpy as np
from trac_ik_python.trac_ik import IK
import sys
import os
import open3d as o3d
import fcl
import cv2
from scipy.stats import norm


file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
learning_dir = os.path.join(file_dir, '../ScoreNet')
dt_learning_dir = os.path.join(file_dir, '../VPformer')
sys.path.append(util_dir)
sys.path.append('/home/corallab3/Documents/Hanwen/ompl-1.5.2/py-bindings')
sys.path.append(learning_dir)
sys.path.append(dt_learning_dir)
import ompl.base as ob
import ompl.util as ou
import ompl.geometric as og
from stl_reader import stl_reader
from obj_reader import obj_reader
from pc_extractor_real_cam import pc_extractor_real_cam
from pc_extractor_real_cam import visualize_scene
from pc_extractor_real_cam import save_object
from global_scene_real import global_scene_real
from runner import feed_forward
from dt_runner import nbv_feed_forward

#define parameters
#*************************************************************************************************#
#global settings
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

#env settings
table_dims = gymapi.Vec3(np.random.rand()*0.2 + 0.8, np.random.rand()*0.3 + 1.2, 
                         np.random.rand()*0.15 + 0.05)
table_dims = gymapi.Vec3(0.65, 1.12, 0.15)
piece_width = 0.03
min_num_of_objects = 0
max_num_of_objects = 0
max_scaling_factor = 0
fall_height = table_dims.z
max_drawer_height = 0.33
min_drawer_height = 0.33
ADD_COVER = True
#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
def get_best_cam_pose(scene, camera_pose_list):
    best_score = - sys.maxsize
    best_index = None
    for t in range(len(camera_pose_list)):
        pose_candidate = camera_pose_list[t]
        candidate_score = feed_forward(scene.scene_, pose_candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_index = t
    print (f'best score is : {best_score}')
    return best_index

def get_random_arm_angle():
    return [np.random.rand()*6.28*2 - 6.28,
            np.random.rand()*6.28*2 - 6.28,
            np.random.rand()*3.14*2 - 3.14,
            np.random.rand()*6.28*2 - 6.28,
            np.random.rand()*6.28*2 - 6.28,
            np.random.rand()*6.28*2 - 6.28]


def write_to_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    img = Image.fromarray(new_image, 'RGB')
    img.save(image_name)

def convert_rgb_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    return new_image

def write_to_seg_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    img = Image.fromarray(new_image)
    img.save(image_name)

def convert_seg_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    return new_image


def write_to_depth_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1))
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] >= -5:
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
            else:
                new_image[i][j][0] = 5000
    cv2.imwrite(image_name, new_image.astype(np.uint16))

def convert_depth_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1), dtype = np.uint16)
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] != mini:
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
            else:
                new_image[i][j][0] = 65535
    return new_image


def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)

def rotation_concat(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)

def get_random_loc(x_min, x_max, y_min, y_max, z_min, z_max):
    x_can = np.random.random()*(x_max - x_min) + x_min
    y_can = np.random.random()*(y_max - y_min) + y_min
    z_can = np.random.random()*(z_max - z_min) + z_min
    return gymapi.Vec3(x_can, y_can, z_can)

def dt_viewpoint_selection(sim, dt_states, dt_actions, dt_rewards, seq_counter):
    mean, std = nbv_feed_forward(dt_states, dt_actions, dt_rewards, seq_counter)
    camera_pose_list = []
    camera_setting_list = []
    std = [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
    while len(camera_pose_list) < 100:
        target_pos = gymapi.Vec3(np.random.normal(mean[4], std[4]),
                                 np.random.normal(mean[5], std[5]),
                                 np.random.normal(mean[6], std[6]))
        q1 = np.random.normal(mean[0], std[0])
        q2 = np.random.normal(mean[1], std[1])
        q3 = np.random.normal(mean[2], std[2])
        q4 = np.random.normal(mean[3], std[3])
        normal_term = math.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
        target_quat = gymapi.Quat(q1/normal_term, q2/normal_term, q3/normal_term, q4/normal_term)
        r = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        test_vec = r.apply(np.array([1, 0, 0]))
        test_vec2 = r.apply(np.array([0, 0, 1]))
        if 0.2 <= target_pos.x <= table_dims.x + 0.3 and\
            -table_dims.y*0.4 + 0.02 <= target_pos.y <= table_dims.y*0.4 - 0.02 and\
            table_dims.z <= target_pos.z <= table_dims.z + drawer_height - 0.02 and\
            test_vec[2] <= 0 and test_vec[0] >= 0 and test_vec2[2] >= 0:
            
            camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                    target_pos.x, target_pos.y, target_pos.z])

            r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
            cam_offset_vector = np.array([0.11, 0.03, 0.08])
            rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
            converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                     target_pos.y - rot_cam_offset_vector[1],
                                                     target_pos.z - rot_cam_offset_vector[2], 
                                                     ur5e_pose.p.x, 
                                                     ur5e_pose.p.y,
                                                     ur5e_pose.p.z)
            converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

            seed_state = [0.0]*ik_solver2.number_of_joints
            dof_result = ik_solver2.get_ik(seed_state, 
                                           converted_coord[0],
                                           converted_coord[1],
                                           converted_coord[2],
                                           converted_quat.x, 
                                           converted_quat.y,
                                           converted_quat.z,
                                           converted_quat.w)
            if dof_result:
                end_state_collision = ur5e_in_collision(dof_result, real_offset)
                if not end_state_collision:
                    camera_pose_list.append(camera_pose)
                    camera_setting_list.append([camera_loc, camera_focus])
         
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
    print (len(camera_pose_list))
    return camera_pose_list[best_cam_pose_index]




def random_sample_guided_selection(sim, env, test_cam, scene):
    camera_pose_list = []
    camera_setting_list = []
    while len(camera_pose_list) < 100:
        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                    -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                    table_dims.z, table_dims.z + drawer_height - 0.02)
        camera_focus = get_random_loc(0 + 0.3, table_dims.x + 0.3,
                                      -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                      table_dims.z, camera_loc.z)
        gym.set_camera_location(test_cam, env, 
                                camera_loc, 
                                camera_focus)
        target_pos = gym.get_camera_transform(sim, env, test_cam).p
        target_quat = gym.get_camera_transform(sim, env, test_cam).r
        camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                target_pos.x, target_pos.y, target_pos.z])

        r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        cam_offset_vector = np.array([0.11, 0.0, 0.08])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                 target_pos.y - rot_cam_offset_vector[1],
                                                 target_pos.z - rot_cam_offset_vector[2], 
                                                 ur5e_pose.p.x, 
                                                 ur5e_pose.p.y,
                                                 ur5e_pose.p.z)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

        seed_state = [0.0]*ik_solver2.number_of_joints
        dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                       converted_quat.w)
        if dof_result:
            end_state_collision = ur5e_in_collision(dof_result, real_offset)
            if not end_state_collision:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus])
                     
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1]

def new_mpc_based_on_single_point(sim, env, test_cam, scene):
    camera_loc, camera_focus = random_sample_guided_selection(sim, env, test_cam, scene)
    print ('initial point')
    print (camera_loc, camera_focus)
    cam_x_mean, cam_y_mean, cam_z_mean = camera_loc.x, camera_loc.y, camera_loc.z
    foc_x_mean, foc_y_mean, foc_z_mean = camera_focus.x, camera_focus.y, camera_focus.z
    total_sample = 1000
    elite_sample = [800, 500, 200, 100, 50]
    cam_x_std, cam_y_std, cam_z_std = 0.1, 0.1, 0.1
    foc_x_std, foc_y_std, foc_z_std = 0.1, 0.1, 0.1
    iteration = 6
    for t in range(iteration):
        print (f"iteraion: {t}\n")
        cam_x, cam_y, cam_z = [], [], []
        foc_x, foc_y, foc_z = [], [], []
        while len(cam_x) < total_sample:
            sample_x = np.random.normal(cam_x_mean, cam_x_std)
            sample_y = np.random.normal(cam_y_mean, cam_y_std)
            sample_z = np.random.normal(cam_z_mean, cam_z_std)
            if 0.2 <= sample_x <= 0.3 + table_dims.x and \
               -table_dims.y*0.4 + 0.02 <= sample_y <= table_dims.y*0.4 - 0.02 and\
               table_dims.z <= sample_z <= table_dims.z + drawer_height - 0.02:
                distance = (sample_x - ur5e_pose.p.x)**2 + \
                           (sample_y - ur5e_pose.p.y)**2 + \
                           (sample_z - ur5e_pose.p.z)**2
                if distance <= 1:
                    cam_x.append(sample_x)
                    cam_y.append(sample_y)
                    cam_z.append(sample_z)
        while len(foc_x) < total_sample:
            sample_x = np.random.normal(foc_x_mean, foc_x_std)
            if 0.3 <= sample_x <= 0.3 + table_dims.x:
                foc_x.append(sample_x)
        while len(foc_y) < total_sample:
            sample_y = np.random.normal(foc_y_mean, foc_y_std)
            if -table_dims.y*0.4 + 0.02 <= sample_y <= table_dims.y*0.4 - 0.02:
                foc_y.append(sample_y)
        while len(foc_z) < total_sample:
            sample_z = np.random.normal(foc_z_mean, foc_z_std)
            if table_dims.z <= sample_z <= table_dims.z + drawer_height - 0.02 and sample_z <= cam_z[len(foc_z)]:
                foc_z.append(sample_z)
        scores = []
        for k in range(total_sample):
            cam_location = gymapi.Vec3(cam_x[k], cam_y[k], cam_z[k])
            cam_focus = gymapi.Vec3(foc_x[k], foc_y[k], foc_z[k])
            gym.set_camera_location(test_cam, env, cam_location, cam_focus)

            target_pos = gym.get_camera_transform(sim, env, test_cam).p
            target_quat = gym.get_camera_transform(sim, env, test_cam).r

            pose_candidate = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                       target_pos.x, target_pos.y, target_pos.z])
            candidate_score = feed_forward(scene.scene_, pose_candidate)
            scores.append([candidate_score, k])
        scores.sort(key = lambda x : x[0], reverse = True)
        if t != iteration - 1:
            fit_cam_x, fit_cam_y, fit_cam_z = [], [], []
            fit_foc_x, fit_foc_y, fit_foc_z = [], [], []
            for k in range(elite_sample[t]):
                fit_cam_x.append(cam_x[scores[k][1]])
                fit_cam_y.append(cam_y[scores[k][1]])
                fit_cam_z.append(cam_z[scores[k][1]])
                fit_foc_x.append(foc_x[scores[k][1]])
                fit_foc_y.append(foc_y[scores[k][1]])
                fit_foc_z.append(foc_z[scores[k][1]])
            cam_x_mean, cam_x_std = norm.fit(fit_cam_x)
            cam_y_mean, cam_y_std = norm.fit(fit_cam_y)
            cam_z_mean, cam_z_std = norm.fit(fit_cam_z)
            foc_x_mean, foc_x_std = norm.fit(fit_foc_x)
            foc_y_mean, foc_y_std = norm.fit(fit_foc_y)
            foc_z_mean, foc_z_std = norm.fit(fit_foc_z)
        else:
            res = []
            for t in range(20):
                best_index = scores[t][1]
                res.append([gymapi.Vec3(cam_x[best_index], cam_y[best_index], cam_z[best_index]), gymapi.Vec3(foc_x[best_index], foc_y[best_index], foc_z[best_index])])
            return res


def mpc_viewpoint_selection_new_grad(sim, env, test_cam, scene):
    elite_sample = [800, 500, 200, 50, 20]
    total_sample = 1000
    iteration = 6
    cam_x_mean, cam_x_std = 0.3 + table_dims.x/2.0, table_dims.x/2.0 + 0.1
    cam_y_mean, cam_y_std = 0, table_dims.y*0.4 - 0.02
    cam_z_mean, cam_z_std = table_dims.z + drawer_height/2.0, drawer_height/2.0
    foc_x_mean, foc_x_std = 0.3 + table_dims.x/2.0, table_dims.x/2.0
    foc_y_mean, foc_y_std = 0, table_dims.y*0.4 - 0.02
    foc_z_mean, foc_z_std = table_dims.z + drawer_height/2.0, drawer_height/2.0
    for t in range(iteration):
        print (f"iteraion: {t}\n")
        cam_x, cam_y, cam_z = [], [], []
        foc_x, foc_y, foc_z = [], [], []
        while len(cam_x) < total_sample:
            sample_x = np.random.normal(cam_x_mean, cam_x_std)
            sample_y = np.random.normal(cam_y_mean, cam_y_std)
            sample_z = np.random.normal(cam_z_mean, cam_z_std)

            sample_fx = np.random.normal(foc_x_mean, foc_x_std)
            sample_fy = np.random.normal(foc_y_mean, foc_y_std)
            sample_fz = np.random.normal(foc_z_mean, foc_z_std)

            if 0.2 <= sample_x <= table_dims.x + 0.3 and\
               0.3 <= sample_fx <= table_dims.x + 0.3 and\
               -table_dims.y*0.4 + 0.02 <= sample_y <= table_dims.y*0.4 - 0.02 and\
               -table_dims.y*0.4 + 0.02 <= sample_fy <= table_dims.y*0.4 - 0.02 and\
               table_dims.z <= sample_z <= table_dims.z + drawer_height - 0.02 and\
               table_dims.z <= sample_fz <= table_dims.z + drawer_height - 0.02 and\
               sample_fz <= sample_z:
                gym.set_camera_location(test_cam, env, 
                                        gymapi.Vec3(sample_x, sample_y, sample_z), 
                                        gymapi.Vec3(sample_fx, sample_fy, sample_fz))
                target_pos = gym.get_camera_transform(sim, env, test_cam).p
                target_quat = gym.get_camera_transform(sim, env, test_cam).r
                camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                        target_pos.x, target_pos.y, target_pos.z])

                r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                cam_offset_vector = np.array([0.11, 0.0, 0.08])
                rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
                converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                         target_pos.y - rot_cam_offset_vector[1],
                                                         target_pos.z - rot_cam_offset_vector[2], 
                                                         ur5e_pose.p.x, 
                                                         ur5e_pose.p.y,
                                                         ur5e_pose.p.z)
                converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

                seed_state = [0.0]*ik_solver2.number_of_joints
                dof_result = ik_solver2.get_ik(seed_state, 
                                               converted_coord[0],
                                               converted_coord[1],
                                               converted_coord[2],
                                               converted_quat.x, 
                                               converted_quat.y,
                                               converted_quat.z,
                                               converted_quat.w)
                if dof_result:
                    end_state_collision = ur5e_in_collision(dof_result, real_offset)
                    if not end_state_collision:
                        cam_x.append(sample_x)
                        cam_y.append(sample_y)
                        cam_z.append(sample_z)
                        foc_x.append(sample_fx)
                        foc_y.append(sample_fy)
                        foc_z.append(sample_fz)

            #if 0.2 <= sample_x <= 0.3 + table_dims.x and \
            #   -table_dims.y*0.4 + 0.02 <= sample_y <= table_dims.y*0.4 - 0.02 and\
            #   table_dims.z <= sample_z <= table_dims.z + drawer_height - 0.02:
            #    distance = (sample_x - ur5e_pose.p.x)**2 + \
            #               (sample_y - ur5e_pose.p.y)**2 + \
            #               (sample_z - ur5e_pose.p.z)**2
            #    if distance <= 1:
            #        cam_x.append(sample_x)
            #        cam_y.append(sample_y)
            #        cam_z.append(sample_z)
        #while len(foc_x) < total_sample:
        #    sample_x = np.random.normal(foc_x_mean, foc_x_std)
        #    if 0.3 <= sample_x <= 0.3 + table_dims.x:
        #        foc_x.append(sample_x)
        #while len(foc_y) < total_sample:
        #    sample_y = np.random.normal(foc_y_mean, foc_y_std)
        #    if -table_dims.y*0.4 + 0.02 <= sample_y <= table_dims.y*0.4 - 0.02:
        #        foc_y.append(sample_y)
        #while len(foc_z) < total_sample:
        #    sample_z = np.random.normal(foc_z_mean, foc_z_std)
        #    if table_dims.z <= sample_z <= table_dims.z + drawer_height - 0.02 and sample_z <= cam_z[len(foc_z)]:
        #        foc_z.append(sample_z)
        scores = []
        for k in range(total_sample):
            cam_location = gymapi.Vec3(cam_x[k], cam_y[k], cam_z[k])
            cam_focus = gymapi.Vec3(foc_x[k], foc_y[k], foc_z[k])
            gym.set_camera_location(test_cam, env, cam_location, cam_focus)

            target_pos = gym.get_camera_transform(sim, env, test_cam).p
            target_quat = gym.get_camera_transform(sim, env, test_cam).r

            pose_candidate = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                       target_pos.x, target_pos.y, target_pos.z])
            candidate_score = feed_forward(scene.scene_, pose_candidate)
            scores.append([candidate_score, k])
        scores.sort(key = lambda x : x[0], reverse = True)
        if t != iteration - 1:
            fit_cam_x, fit_cam_y, fit_cam_z = [], [], []
            fit_foc_x, fit_foc_y, fit_foc_z = [], [], []
            for k in range(elite_sample[t]):
                fit_cam_x.append(cam_x[scores[k][1]])
                fit_cam_y.append(cam_y[scores[k][1]])
                fit_cam_z.append(cam_z[scores[k][1]])
                fit_foc_x.append(foc_x[scores[k][1]])
                fit_foc_y.append(foc_y[scores[k][1]])
                fit_foc_z.append(foc_z[scores[k][1]])
            cam_x_mean, cam_x_std = norm.fit(fit_cam_x)
            cam_y_mean, cam_y_std = norm.fit(fit_cam_y)
            cam_z_mean, cam_z_std = norm.fit(fit_cam_z)
            foc_x_mean, foc_x_std = norm.fit(fit_foc_x)
            foc_y_mean, foc_y_std = norm.fit(fit_foc_y)
            foc_z_mean, foc_z_std = norm.fit(fit_foc_z)
        else:
            res = []
            for t in range(10):
                best_index = scores[t][1]
                res.append([gymapi.Vec3(cam_x[best_index], cam_y[best_index], cam_z[best_index]), gymapi.Vec3(foc_x[best_index], foc_y[best_index], foc_z[best_index]), scores[t][0]])
            return res


def mpc_viewpoint_selection(sim, env, test_cam, scene):
    elite_sample = 5
    total_sample = 100
    iteration = 10
    cam_x_mean, cam_x_std = 0.3 + table_dims.x/2.0, table_dims.x/2.0 + 0.1
    cam_y_mean, cam_y_std = 0, table_dims.y*0.4 - 0.02
    cam_z_mean, cam_z_std = table_dims.z + drawer_height/2.0, drawer_height/2.0
    foc_x_mean, foc_x_std = 0.3 + table_dims.x/2.0, table_dims.x/2.0
    foc_y_mean, foc_y_std = 0, table_dims.y*0.4 - 0.02
    foc_z_mean, foc_z_std = 0.5, 0.5
    for t in range(iteration):
        print (f"iteraion: {t}\n")
        cam_x, cam_y, cam_z = [], [], []
        foc_x, foc_y, foc_z = [], [], []
        while len(cam_x) < 100:
            sample_x = np.random.normal(cam_x_mean, cam_x_std)
            if 0.2 <= sample_x <= 0.86:
                cam_x.append(sample_x)
        while len(foc_x) < 100:
            sample_x = np.random.normal(foc_x_mean, foc_x_std)
            if 0.3 <= sample_x <= 0.86:
                foc_x.append(sample_x)
        while len(cam_y) < 100:
            sample_y = np.random.normal(cam_y_mean, cam_y_std)
            if -0.46 <= sample_y <= 0.46:
                cam_y.append(sample_y)
        while len(foc_y) < 100:
            sample_y = np.random.normal(foc_y_mean, foc_y_std)
            if -0.46 <= sample_y <= 0.46:
                foc_y.append(sample_y)
        while len(cam_z) < 100:
            sample_z = np.random.normal(cam_z_mean, cam_z_std)
            if 0.15 <= sample_z <= 0.82:
                cam_z.append(sample_z)
        while len(foc_z) < 100:
            sample_z = np.random.normal(foc_z_mean, foc_z_std)
            if 0 <= sample_z <= 1:
                foc_z.append(cam_z[len(foc_z)]*sample_z/1)
        scores = []
        for k in range(100):
            cam_location = gymapi.Vec3(cam_x[k], cam_y[k], cam_z[k])
            cam_focus = gymapi.Vec3(foc_x[k], foc_y[k], foc_z[k])
            gym.set_camera_location(test_cam, env, cam_location, cam_focus)

            target_pos = gym.get_camera_transform(sim, env, test_cam).p
            target_quat = gym.get_camera_transform(sim, env, test_cam).r

            pose_candidate = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                       target_pos.x, target_pos.y, target_pos.z])
            candidate_score = feed_forward(scene.scene_, pose_candidate)
            scores.append([candidate_score, k])
        scores.sort(key = lambda x : x[0], reverse = True)
        if t != iteration - 1:
            fit_cam_x, fit_cam_y, fit_cam_z = [], [], []
            fit_foc_x, fit_foc_y, fit_foc_z = [], [], []
            for k in range(elite_sample):
                fit_cam_x.append(cam_x[scores[k][1]])
                fit_cam_y.append(cam_y[scores[k][1]])
                fit_cam_z.append(cam_z[scores[k][1]])
                fit_foc_x.append(foc_x[scores[k][1]])
                fit_foc_y.append(foc_y[scores[k][1]])
                fit_foc_z.append(foc_z[scores[k][1]])
            cam_x_mean, cam_x_std = norm.fit(fit_cam_x)
            cam_y_mean, cam_y_std = norm.fit(fit_cam_y)
            cam_z_mean, cam_z_std = norm.fit(fit_cam_z)
            foc_x_mean, foc_x_std = norm.fit(fit_foc_x)
            foc_y_mean, foc_y_std = norm.fit(fit_foc_y)
            foc_z_mean, foc_z_std = norm.fit(fit_foc_z)
        else:
            best_index = scores[0][1]
            print (f'best score: {scores[0][0]:.4f}')
            return [gymapi.Vec3(cam_x[best_index], cam_y[best_index], cam_z[best_index]), gymapi.Vec3(foc_x[best_index], foc_y[best_index], foc_z[best_index])]






#*************************************************************************************************#


#initialize gym
#*************************************************************************************************#
gym = gymapi.acquire_gym()
#*************************************************************************************************#

#parse arguments

#*************************************************************************************************#
args = gymutil.parse_arguments(description="ur5e example", custom_parameters = [{'name':'--env_id', 'type':int, 'help':'env_id', 'default':0}, {'name':'--vp_method', 'type':int, 'help':'viewpoint choosing method: 0-random, 1-random guided, 2-MPC style', 'default': 0}])
env_id = int(args.env_id)
vp_method = int(args.vp_method)

print (vp_method)
#*************************************************************************************************#

#create a simulator
#*************************************************************************************************#
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
#*************************************************************************************************#

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.8)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, 
                     args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()
#*************************************************************************************************#

#configure a ground plane
#*************************************************************************************************#
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
#*************************************************************************************************#

#all assets
#*************************************************************************************************#
asset_root = "/home/corallab3/Documents/active_sensing/assets/"
ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real.urdf"
ur5e_collision_parts = ["urdf/ur5e/meshes/collision/base.stl",
                        "urdf/ur5e/meshes/collision/shoulder.stl",
                        "urdf/ur5e/meshes/collision/upperarm.stl",
                        "urdf/ur5e/meshes/collision/forearm.stl",
                        "urdf/ur5e/meshes/collision/wrist1.stl",
                        "urdf/ur5e/meshes/collision/wrist2.stl",
                        "urdf/ur5e/meshes/collision/wrist3.stl",
                        "urdf/ur5e/meshes/collision/camera_and_frame.stl"]

object_asset_files = []
object_collision_files = []
object_offset = []
object_centroid_m = []
object_common_prefix = "urdf/ycb/"
with open(asset_root + "urdf/ycb/object_urdf_new.txt") as f:
    for line in f:
        object_asset_files.append(object_common_prefix + line[:-1])
with open(asset_root + "urdf/ycb/object_collision_new.txt") as f:
    for line in f:
        object_collision_files.append(object_common_prefix + line[:-1])
with open(asset_root + "urdf/ycb/object_offset_new.txt") as f:
    for line in f:
        div = line[:-1].split(" ")
        object_offset.append([float(x) for x in div])
with open(asset_root + "urdf/ycb/all_centroid_m_new.txt") as f:
    for line in f:
        div = line[:-1].split(" ")
        div_f = [float(x) for x in div]
        object_centroid_m.append([np.array([div_f[0], div_f[1], div_f[2]]), div_f[3]])


#object_asset_files = ["urdf/ycb/025_mug/025_mug.urdf",
#                      "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
#                      "urdf/ycb/006_mustard_bottle/textured.urdf"]
#object_collision_files = ["urdf/ycb/025_mug/mug_collision.obj",
#                          "urdf/ycb/010_potted_meat_can/collision.obj",
#                          "urdf/ycb/006_mustard_bottle/textured_vhacd.obj"]


#setup all collision meshes
#setup is done outside the env loop since all robots are the same
ur5e_collision_models = []
ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                  R.from_euler('xy', [90, 180], degrees = True),
                  R.from_euler('xy', [180, 180], degrees = True),
                  R.from_euler('z',  [-180], degrees = True),
                  R.from_euler('x',  [-180], degrees = True),
                  R.from_euler('x',  [90], degrees = True),
                  R.from_euler('z',  [-90], degrees = True),
                  R.from_euler('z', [180], degrees = True)]
ur5e_translations = [[0, 0, 0], 
                     [0, 0, 0],
                     [0, -0.138, 0],
                     [0, -0.007, 0],
                     [0, 0.127, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]
for i in range(len(ur5e_collision_parts)):
    parts_path = ur5e_collision_parts[i]
    collision_mesh = stl_reader(asset_root + parts_path)
    m = fcl.BVHModel()
    collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
    verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    ur5e_collision_models.append(m)

object_collision_lib = []
#*************************************************************************************************#


#calculate Inverse Kinematics
#*************************************************************************************************#
urdf_str = ''
with open("../assets/urdf/ur5e/ur5e_mimic_real.urdf") as f:
    urdf_str = f.read()
#ik_solver = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
#seed_state = [0.0]*ik_solver.number_of_joints
#test_quat = gymapi.Quat()
#quat_result = test_quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi)
#print (quat_result)
#dof_result = ik_solver.get_ik(seed_state, 
#                              0.5, 0.5, 0.5,
#                              quat_result.x, 
#                              quat_result.y,
#                              quat_result.z,
#                              quat_result.w)
#*************************************************************************************************#

# create viewer using the default camera properties
#*************************************************************************************************#
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')
#*************************************************************************************************#

#set up the environment grid
#*************************************************************************************************#
spacing = 2
env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, 0)
#*************************************************************************************************#

#load asset
#*************************************************************************************************#
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.use_mesh_materials = True

ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
table_asset = gym.create_box(sim, table_dims.x,
                                  table_dims.y,
                                  table_dims.z,
                                  asset_options)

#size of left/right cover will be decided by table size
drawer_height = np.random.random()*(max_drawer_height - min_drawer_height) + min_drawer_height
side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
left_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                       side_cover_dims.y,
                                       side_cover_dims.z,
                                       asset_options)
right_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                        side_cover_dims.y,
                                        side_cover_dims.z,
                                        asset_options)

#upper cover
upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y*0.8 + piece_width, 0.03)
upper_cover_asset = gym.create_box(sim, upper_cover_dims.x,
                                        upper_cover_dims.y,
                                        upper_cover_dims.z,
                                        asset_options)

asset_options.fix_base_link = False
object_assets = []
test_assets = []
for t in range(len(object_asset_files)):
    object_assets.append(gym.load_asset(sim, asset_root, object_asset_files[t], asset_options))
asset_options.fix_base_link = True
test_assets.append(gym.load_asset(sim, asset_root, object_asset_files[0], asset_options))
asset_options.fix_base_link = False

#*************************************************************************************************#


#initial pose
#*************************************************************************************************#
ur5e_pose = gymapi.Transform()
ur5e_pose.p = gymapi.Vec3(np.random.rand()*0.4 - 0.2, np.random.rand()*0.4 - 0.2, 0.0)
ur5e_pose.p = gymapi.Vec3(-0.4, 0.3, 0)
ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5)

left_cover_pose = gymapi.Transform()
left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y*0.4, 
                                table_dims.z + side_cover_dims.z/2.0)

right_cover_pose = gymapi.Transform()
right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y*0.4, 
                                 table_dims.z + side_cover_dims.z/2.0)

upper_cover_pose = gymapi.Transform()
upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z)

camera_focus = gymapi.Vec3(0, 0, 0)
camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 70.25
camera_props.width = 1280
camera_props.height = 720

#set all environment collision models

plane_normal = np.array([0.0, 0.0, 1.0])
col_plane = fcl.Plane(plane_normal, 0)
plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
trans_table = fcl.Transform(np.array([table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5]))
table_obj = fcl.CollisionObject(col_table, trans_table)

col_left_cover = fcl.Box(side_cover_dims.x,
                         side_cover_dims.y,
                         side_cover_dims.z)
trans_left_cover = fcl.Transform(np.array([table_pose.p.x, table_dims.y*0.4, 
                                           table_dims.z + side_cover_dims.z/2.0]))
left_cover_obj = fcl.CollisionObject(col_left_cover, trans_left_cover)

col_right_cover = fcl.Box(side_cover_dims.x,
                          side_cover_dims.y,
                          side_cover_dims.z)
trans_right_cover = fcl.Transform(np.array([table_pose.p.x, -table_dims.y*0.4, 
                                            table_dims.z + side_cover_dims.z/2.0]))
right_cover_obj = fcl.CollisionObject(col_right_cover, trans_right_cover)

object_collision_models = [table_obj, left_cover_obj, right_cover_obj]

if ADD_COVER:
    col_upper_cover = fcl.Box(upper_cover_dims.x,
                              upper_cover_dims.y,
                              upper_cover_dims.z)
    trans_upper_cover = fcl.Transform(np.array([table_pose.p.x, 0.0, 
                                      table_dims.z + side_cover_dims.z]))
    upper_cover_obj = fcl.CollisionObject(col_upper_cover, trans_upper_cover)
    object_collision_models.append(upper_cover_obj)

#*************************************************************************************************#


#create environment
#*************************************************************************************************#
envs = []
ur5e_handles = []
body_cam_handles = []
camera_candidates = []
chosen_object = []
chosen_scale = []
object_normalize = []
num_of_objects = np.random.randint(min_num_of_objects, max_num_of_objects+1)
for i in range(num_of_envs):
    envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
    ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

    #get joint handler
    spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
    ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
    wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")

    #attach body camera sensor
    cam_link = gym.find_actor_rigid_body_handle(envs[-1], ur5e_handles[-1], "wrist_3_link")

    #right in front of D435i model
    cam_offset_x = 0.11
    cam_offset_z = 0.08
    body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
    body_cam_transform = gymapi.Transform()
    body_cam_transform.p = gymapi.Vec3(cam_offset_x, 0, cam_offset_z)
    gym.attach_camera_to_body(body_cam_handles[-1], envs[-1], cam_link, body_cam_transform, 
                              gymapi.CameraFollowMode.FOLLOW_TRANSFORM)


    #print (gym.get_camera_proj_matrix(sim, envs[-1], body_cam_handles[-1]))

    gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
    gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
    gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
    if ADD_COVER:
        gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)

    object_index = np.random.randint(len(object_asset_files), size=num_of_objects)
    chosen_object.append(object_index)
    object_index = [1, 8, 9, 11, 12, 26, 28]
    #object_index = [0]
    object_loc = np.random.normal(0, table_dims.x/10.0, size=(2, num_of_objects))
    object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = num_of_objects)/10.0 + 1.0

    chosen_scale.append(object_scaling_factor)
    object_handles = []

    with open("object_name.txt", 'a') as f:
        for k in range(num_of_objects):
            f.write(object_asset_files[object_index[k]])
    

    fixed_object_loc = [gymapi.Vec3(0.65, 0, 0.4), 
                        gymapi.Vec3(0.7, -0.3, 0.15),
                        gymapi.Vec3(0.4, -0.1, 0.15),
                        gymapi.Vec3(0.4, -0.3, 0.15),
                        gymapi.Vec3(0.4, 0.1, 0.15),
                        gymapi.Vec3(0.5, 0.3, 0.15),
                        gymapi.Vec3(0.7, 0.2, 0.15)]

    #set up objects
    for k in range(num_of_objects):
        object_pose = gymapi.Transform()
        #object_pose.p = get_random_loc(0.3 + table_dims.x*0.2, 0.3 + table_dims.x*0.8,
        #                               -table_dims.y*0.4, table_dims.y*0.4,
        #                               table_dims.z, table_dims.z + drawer_height*0.5)
        object_pose.p = fixed_object_loc[k]
        object_handles.append(gym.create_actor(envs[-1], 
                                               object_assets[object_index[k]], 
                                               object_pose, 
                                               "object" + str(k) + str(i), 0, 2**(k+1), k+1))
        gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
        object_normalize.append(object_centroid_m[object_index[k]])
        file_path = object_collision_files[object_index[k]]
        collision_mesh = obj_reader(asset_root + file_path)
        m = fcl.BVHModel()
        collision_mesh.set_scale(object_scaling_factor[k])
        collision_mesh.set_offset(object_offset[object_index[k]])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        object_collision_lib.append(m)


    #set up global camera to record configuration
    body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
    viewpoint_candidate = gymapi.Vec3(2.2, 0, 0.3)
    gym.set_camera_location(body_cam_handles[-1], envs[-1], 
                            viewpoint_candidate, 
                            camera_focus)

        #target_pos = gym.get_camera_transform(sim, envs[-1], local_camera_handles[-1]).p
        #target_quat = gym.get_camera_transform(sim, envs[-1], local_camera_handles[-1]).r
        #converted_coord = global_coord_converter(viewpoint_candidate.x - cam_offset_x,
        #                                         viewpoint_candidate.y,
        #                                         viewpoint_candidate.z - cam_offset_z, 0, 0, 0)
        #converted_quat = quaternion_multiply(gymapi.Quat(-0.707, 0, 0, 0.707), target_quat)

        #ik_solver = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
        #seed_state = [0.0]*ik_solver.number_of_joints
        #dof_result = ik_solver.get_ik(seed_state, 
        #                              converted_coord[0],
        #                              converted_coord[1],
        #                              converted_coord[2],
        #                              converted_quat.x, 
        #                              converted_quat.y,
        #                              converted_quat.z,
        #                              converted_quat.w)


        #if dof_result:
        #    #gym.set_dof_target_position(envs[-1], spj, dof_result[0]) 
        #    #gym.set_dof_target_position(envs[-1], slj, dof_result[1]) 
        #    #gym.set_dof_target_position(envs[-1], ej,  dof_result[2]) 
        #    #gym.set_dof_target_position(envs[-1], wj1, dof_result[3]) 
        #    #gym.set_dof_target_position(envs[-1], wj2, dof_result[4]) 
        #    #gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
        #    #final_ori = gym.get_joint_transform(envs[-1], wj3)
        #    #test_ori = gym.get_rigid_transform(envs[-1], cam_link)
        #    #res = gym.get_actor_rigid_body_states(envs[-1], ur5e_handles[-1], 1)
        #    print (converted_coord, converted_quat)
        #    print (dof_result)
        #    #print (test_ori.p, test_ori.r)
        #    print ("found solution")
        ##else:
        ##    gym.set_dof_target_position(envs[-1], ej, -math.pi * 0.5)
        ##    #gym.set_dof_target_position(envs[-1], wj1, 0.966)
        ##    #gym.set_dof_target_position(envs[-1], wj2, math.pi * 0.5)


#*************************************************************************************************#


#*************************************************************************************************#
cam_pos = gymapi.Vec3(6, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(-1.0, 0.0, 0.0))
gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(1.0, 0.0, 0.0))
#*************************************************************************************************#

#*************************************************************************************************#
def get_pose_from_dof(dofs):
    #link1 pose
    trans1, rot1 = [0, 0, 0], [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
    
    #link2 pose
    rot2_initial = [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot2_new = R.from_euler('z', dofs[0]).as_quat().tolist()
    rot2_final = rotation_concat(rot2_new, rot2_initial)
    trans2, rot2 = [0, 0, 0.1625], rot2_final

    #link3 pose
    #rot3_initial = [0, -0, 1, 0]
    rot3_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot3_vector = R.from_quat(rot2_new).apply([0, 1, 0])
    rot3_final = rotation_concat(rot2_new, rot3_initial)
    rot3_new = R.from_rotvec(dofs[1]*rot3_vector).as_quat().tolist()
    rot3_final = rotation_concat(rot3_new, rot3_final)
    trans3, rot3 = [0, 0, 0.1625], rot3_final

    #link4 pose
    #rot4_initial = [0, -0, 1, 0]
    rot4_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot4_vector = rot3_vector
    rot4_final = rot3_final
    rot4_offset = R.from_quat(rot3_final).apply([0, 0, 0.425])
    rot4_new = R.from_rotvec(dofs[2]*rot4_vector).as_quat().tolist()
    rot4_final = rotation_concat(rot4_new, rot4_final)
    trans4, rot4 = trans3 + rot4_offset, rot4_final

    #link5 pose
    rot5_offset = R.from_quat(rot4_final).apply([0, -0.1333, 0.3915])
    rot5_initial = [0, -0, 1, 0]
    rot5_vector = rot4_vector
    rot5_final = rotation_concat(rot2_new, rot5_initial)
    rot5_final = rotation_concat(rot3_new, rot5_final)
    rot5_final = rotation_concat(rot4_new, rot5_final)
    rot5_new = R.from_rotvec(dofs[3]*rot5_vector).as_quat().tolist()
    rot5_final = rotation_concat(rot5_new, rot5_final)
    trans5, rot5 = trans4 + rot5_offset, rot5_final

    #link6 pose
    rot6_offset = R.from_quat(rot5_final).apply([0, 0, 0])
    rot6_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot6_final = rotation_concat(rot2_new, rot6_initial)
    rot6_final = rotation_concat(rot3_new, rot6_final)
    rot6_final = rotation_concat(rot4_new, rot6_final)
    rot6_final = rotation_concat(rot5_new, rot6_final)
    rot6_vector = [0, 0, -1]
    rot6_vector = R.from_quat(rot2_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot3_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot4_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot5_new).apply(rot6_vector)
    rot6_new = R.from_rotvec(dofs[4]*rot6_vector).as_quat().tolist()
    rot6_final = rotation_concat(rot6_new, rot6_final)
    trans6, rot6 = trans5 + rot6_offset, rot6_final

    #link7 pose
    rot7_offset = R.from_quat(rot6_final).apply([0, -0.0996, 0])
    rot7_initial = [math.sqrt(2)/2, math.sqrt(2)/2, 0, 0]
    rot7_final = rotation_concat(rot2_new, rot7_initial)
    rot7_final = rotation_concat(rot3_new, rot7_final)
    rot7_final = rotation_concat(rot4_new, rot7_final)
    rot7_final = rotation_concat(rot5_new, rot7_final)
    rot7_final = rotation_concat(rot6_new, rot7_final)
    rot7_vector = [0, 1, 0]
    rot7_vector = R.from_quat(rot2_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot3_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot4_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot5_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot6_new).apply(rot7_vector)
    rot7_new = R.from_rotvec(dofs[5]*rot7_vector).as_quat().tolist()
    rot7_final = rotation_concat(rot7_new, rot7_final)
    trans7, rot7 = trans6 + rot7_offset, rot7_final

    #camera pose
    rot8_offset = R.from_quat(rot7_final).apply([0.065, 0, 0.04])
    rot8_final = rot7
    trans8, rot8 = trans7 + rot8_offset, rot8_final

    return [[trans1, rot1],
            [trans2, rot2],
            [trans3, rot3],
            [trans4, rot4],
            [trans5, rot5],
            [trans6, rot6],
            [trans7, rot7],
            [trans8, rot8]]




class ur5e_valid(ob.StateValidityChecker):
    def __init__(self, si, real_offset):
        super().__init__(si)
        self.real_offset_ = real_offset

    def isValid(self, dof_state):
        pose_array = get_pose_from_dof(dof_state)

        ur5e_self_col = []
        rots = []
        trans = []

        #print (dof_state[0], dof_state[1], dof_state[2], dof_state[3], dof_state[4], dof_state[5])
        #real_offset = np.array(state_tensor[0][:3])
        for t in range(8):
            rotation = np.array(pose_array[t][1])
            translation = np.array(pose_array[t][0] + self.real_offset_)
            rots.append(rotation)
            trans.append(translation)
            r1 = R.from_quat(rotation)
            tf = fcl.Transform(r1.as_matrix(), translation)
            ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))


        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        self_collision_flag = False

        for t in range(8):
            if t != 0:
                if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
                    self_collision_flag = True
                    break
            col_with_other_part = False
            for q in range(8):
                if q < t-1 or q > t + 1:
                    if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                        col_with_other_part = True
                        break
            if col_with_other_part:
                self_collision_flag = True
                break

        env_collision_flag = False
        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(ur5e_self_col)
        manager1.setup()

        manager2 = fcl.DynamicAABBTreeCollisionManager()
        manager2.registerObjects(object_collision_models + flexible_collision_models)
        manager2.setup()

        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
        rdata = fcl.CollisionData(request = req)
        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
        if rdata.result.is_collision:
            env_collision_flag = True

        return self_collision_flag == False and env_collision_flag == False

def ur5e_in_collision(dof_result, real_offset):

    pose_array = get_pose_from_dof(dof_result)

    ur5e_self_col = []
    rots = []
    trans = []
    #real_offset = np.array(state_tensor[0][:3])
    for t in range(8):
        rotation = np.array(pose_array[t][1])
        translation = np.array(pose_array[t][0] + real_offset)
        rots.append(rotation)
        trans.append(translation)
        r1 = R.from_quat(rotation)
        tf = fcl.Transform(r1.as_matrix(), translation)
        ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))


    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    self_collision_flag = False

    for t in range(8):
        if t != 0:
            if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
                self_collision_flag = True
                break
        col_with_other_part = False
        for q in range(8):
            if q < t-1 or q > t + 1:
                if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                    col_with_other_part = True
                    break
        if col_with_other_part:
            self_collision_flag = True
            break

    env_collision_flag = False
    manager1 = fcl.DynamicAABBTreeCollisionManager()
    manager1.registerObjects(ur5e_self_col)
    manager1.setup()

    manager2 = fcl.DynamicAABBTreeCollisionManager()
    manager2.registerObjects(object_collision_models + flexible_collision_models)
    manager2.setup()

    req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
    rdata = fcl.CollisionData(request = req)
    manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
    if rdata.result.is_collision:
        env_collision_flag = True

    return self_collision_flag == True or env_collision_flag == True



space = ob.RealVectorStateSpace(0)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)
space.addDimension(-3.14, 3.14)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)



#*************************************************************************************************#

real_position = False
#test_dof = [-5.61, -90.02, 80.96, -112.31, -101.32, 3.28]
#test_dof = [x*1.0/180*math.pi for x in test_dof]
#set to zero position at first
for t in range(4000):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)


    gym.sync_frame_time(sim)

    if not real_position:
        gym.set_dof_target_position(envs[-1], spj, 0) 
        gym.set_dof_target_position(envs[-1], slj, -math.pi/2) 
        gym.set_dof_target_position(envs[-1], ej,  0) 
        gym.set_dof_target_position(envs[-1], wj1, -math.pi/2) 
        gym.set_dof_target_position(envs[-1], wj2, 0) 
        gym.set_dof_target_position(envs[-1], wj3, 0)
        real_position = True

    if t == 3000:
        gym.render_all_camera_sensors(sim)
        color_image = gym.get_camera_image(sim, envs[-1], 
                                                           body_cam_handles[0], 
                                                           gymapi.IMAGE_COLOR)
         

        #write_to_image(color_image, "test_focal.jpg")
        print ('Done')



def get_real_rotation(rx, ry, rz):

    vector = np.array([rx, ry, rz])
    mag = np.linalg.norm(vector)
    vector /= mag
    rot1 = R.from_rotvec(mag * vector)
    rot2 = R.from_euler("YX", [-math.pi/2, math.pi/2])

    rot2 = rot1 * rot2

    t1, t2, t3, t4 = rot2.as_quat()

    rot3 = R.from_quat([-t1, -t2, t3, t4])

    print (rot3.apply([-1, 0, 0]))
    print (rot3.apply([0, -1, 0]))
    print (rot3.apply([0, 0, 1]))

    rot4 = R.from_euler('Z', math.pi)
    rot4 = rot3*rot4

    print (rot4.apply([1, 0, 0]))
    print (rot4.apply([0, 1, 0]))
    print (rot4.apply([0, 0, 1]))

    print(rot4.as_quat())

    return rot4.as_quat()

def get_real_location(x, y, z, rotation):
    offset = R.from_quat(rotation).apply([0, 0.03, 0.08])
    return [-0.4 - x + offset[0], 0.3 - y + offset[1], z + offset[2]]



#*************************************************************************************************#
counter = 0
viewpoint_counter = 0
ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
test_cam = gym.create_camera_sensor(envs[-1], camera_props)
first_view_obtained = False
track_last_pose = None
#body_cam_handles.append(test_cam)
flag = True
need_acquire = False
acquire_counter = 0
drawn = False
loc_tracker = None
rot_tracker = None
object_status = []
track_cameras = []
flexible_collision_models = []
sequence_count = 0
final_rotation, final_translation = None, None
object_dict = {}
scene = global_scene_real(1.0, 1.2, 0.85, np.array([0.3, -0.6, 0.05]), table_dims.x, table_dims.y*0.8 - 0.04, drawer_height - 0.02, table_dims.z - 0.05)
end_state_collision = True
coverage_score = 0
track_angle_path = []

dt_states = np.zeros(shape = (1, 8, 1, 101, 121, 86))
dt_actions = np.zeros(shape = (1, 8, 7))
dt_rewards = np.zeros(shape = (1, 8, 1))

dt_camera_pose = []

v1_set = [0.05041503, 0.10564561, -0.43281302, 0.89385128, 0.21217731, 0.41443005, 0.70991528]
v2_set = [-0.04168901, -0.20197801, -0.55242831, -0.80764461, 0.20088865, -0.33497143, 0.70227611]
v3_set = [0.0, 0.55776453, -0.21592074, 0.80142182, 0.26792365, 0.14650505, 0.69469910]

view_sets = [v1_set, v2_set, v3_set]
while not gym.query_viewer_has_closed(viewer):

    if coverage_score >= 0.95 or sequence_count >= 15: break

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    offset = 4

    if counter > 1000:
        if need_acquire:
            gym.clear_lines(viewer)
            if acquire_counter > 500:
                _body_tensor = gym.acquire_rigid_body_state_tensor(sim)
                body_tensor = gymtorch.wrap_tensor(_body_tensor)



                gym.render_all_camera_sensors(sim)
                for t in range(num_of_envs):
                    for q in body_cam_handles:
                        color_image = gym.get_camera_image(sim, envs[-1], 
                                                           body_cam_handles[q], 
                                                           gymapi.IMAGE_COLOR)
                        depth_image = gym.get_camera_image(sim, envs[-1], 
                                                           body_cam_handles[q], 
                                                           gymapi.IMAGE_DEPTH)
                        seg_image = gym.get_camera_image(sim, envs[-1], 
                                                         body_cam_handles[q], 
                                                         gymapi.IMAGE_SEGMENTATION)
                        projection_matrix = np.matrix(gym.get_camera_proj_matrix(sim,
                                                                                 envs[-1],
                                                                                 body_cam_handles[q]))

                        write_to_image(color_image, "real_exp/" + str(vp_method) + "/" + str(env_id) + "/color_test" \
                                       + str(sequence_count) + '_' + str(q) + ".jpg")
                        write_to_depth_image(depth_image, "real_exp/" + str(vp_method) + "/" + str(env_id) + "/depth_test" \
                                             + str(sequence_count) + '_' + str(q) + ".png")
                        write_to_seg_image(seg_image, "real_exp/" + str(vp_method) + "/" + str(env_id) + "/seg_test" \
                                           + str(sequence_count) + '_' + str(q) + ".png")
                        print ("Image Captured")
                        if (q == 0):
                            print (q, t, need_acquire, acquire_counter)
                            #extract point cloud

                            temp_cam = body_cam_handles[q]
                            cam_rotation = gym.get_camera_transform(sim, envs[-1],
                                              temp_cam).r
                            cam_translation = gym.get_camera_transform(sim, envs[-1], 
                                              temp_cam).p
                            new_rgb_image = convert_rgb_image(color_image)
                            new_seg_image = convert_seg_image(seg_image)
                            new_depth_image = convert_depth_image(depth_image)
                            
                            new_cam_rotation = np.array([cam_rotation.x,
                                                         cam_rotation.y,
                                                         cam_rotation.z,
                                                         cam_rotation.w])
                            new_cam_translation = np.array([cam_translation.x,
                                                            cam_translation.y,
                                                            cam_translation.z])

                            dist1 = np.linalg.norm(final_rotation - new_cam_rotation)
                            dist3 = np.linalg.norm(final_rotation - new_cam_rotation*-1)
                            dist2 = np.linalg.norm(final_translation - new_cam_translation)
                            print (new_cam_rotation, new_cam_translation, dist1, dist3, dist2)
                            if (dist1 < 1e-2 or dist3 < 1e-2) and dist2 < 1e-2:
                                
                                next_id = input("Enter the next id: ")

                                div = next_id.split()

                                real_id = div[0]

                                robot_rot = [float(x) for x in div[1:4]]
                                robot_trans = [(float)(x) for x in div[4:7]]

                                rotation = get_real_rotation(robot_rot[0], robot_rot[1], robot_rot[2])

                                translation = get_real_location(robot_trans[0], robot_trans[1], robot_trans[2], rotation)
                                
                                print ("real rotation: ", rotation)
                                print ("real translation:", translation)
                                img_color_file = 'real_exp/' + str(vp_method) + '/' + str(env_id) + '/test_rgb_' + real_id + '.jpg'
                                img_depth_file = 'real_exp/' + str(vp_method) + '/' + str(env_id) + '/test_depth_' + real_id + '.png'
                                img_seg_file = 'real_exp/' + str(vp_method) + '/' + str(env_id) + '/test_seg_' + real_id + '.png'
                                new_rgb_image = o3d.io.read_image(img_color_file)
                                new_depth_image = o3d.io.read_image(img_depth_file)
                                new_seg_image = o3d.io.read_image(img_seg_file)


                                res = pc_extractor_real_cam(new_rgb_image, new_depth_image, new_seg_image, 
                                               rotation, translation, object_dict, True)
                                #print (len(object_dict))

                                visualize_scene(object_dict, True, True)
                                visualize_scene(object_dict, False, True)

                                file_prefix = 'real_exp/' + str(vp_method) + '/' + str(env_id) + '/env' + str(env_id) + "_sequence" + str(sequence_count)

                                coverage_score = scene.register_camera_view(list(rotation), \
                                                           list(translation), \
                                                           np.asarray(new_depth_image), object_dict, file_prefix)

                                scene.vis_scene()
                                save_object(object_dict, file_prefix)

                                if sequence_count < 8:
                                    dt_actions[0, sequence_count] = np.array(list(new_cam_rotation) + list(new_cam_translation))
                                if sequence_count < 7:
                                    dt_rewards[0, sequence_count + 1] = coverage_score

                                sequence_count += 1



                need_acquire = False
                flag = True
                acquire_counter = 0
                print ("end acquire")
            else:
                acquire_counter += 1
        else:
            _state_tensor = gym.acquire_rigid_body_state_tensor(sim)
            state_tensor = gymtorch.wrap_tensor(_state_tensor)

            real_offset = np.array(state_tensor[0][:3])

            end_state_collision = True

            if counter % 2 == 0 and flag:

                gym.clear_lines(viewer)

                scene_mesh = scene.get_surface_collision_mesh()
                scene_vertices = np.asarray(scene_mesh.vertices)
                scene_faces = np.asarray(scene_mesh.triangles)

                
                #add scene surface mesh
                flexible_collision_models = []
                m = fcl.BVHModel()
                m.beginModel(len(scene_vertices), len(scene_faces))
                m.addSubModel(scene_vertices, scene_faces)
                m.endModel()
                flexible_collision_models.append(fcl.CollisionObject(m))

                print ("finish adding surface collision")

                #add object mesh
                #print (object_dict)
                for obj_id, obj_ins in object_dict.items():
                    if obj_id != 0:
                        object_vertices, object_faces = obj_ins.get_collision_mesh()
                        m = fcl.BVHModel()
                        m.beginModel(len(object_vertices), len(object_faces))
                        m.addSubModel(object_vertices, object_faces)
                        m.endModel()
                        flexible_collision_models.append(fcl.CollisionObject(m))

                print ("finish adding object collision mesh")

                #    object_lines = []

                #    for v1, v2, v3 in object_faces:
                #        object_lines += list(object_vertices[v1])
                #        object_lines += list(object_vertices[v2])
                #        object_lines += list(object_vertices[v1])
                #        object_lines += list(object_vertices[v3])
                #        object_lines += list(object_vertices[v2])
                #        object_lines += list(object_vertices[v3])

                #    gym.add_lines(viewer, envs[-1], len(object_lines)//6, object_lines, [1, 0, 0])

                all_lines = []

                for v1, v2, v3 in scene_faces:
                    all_lines += list(scene_vertices[v1])
                    all_lines += list(scene_vertices[v2])
                    all_lines += list(scene_vertices[v1])
                    all_lines += list(scene_vertices[v3])
                    all_lines += list(scene_vertices[v2])
                    all_lines += list(scene_vertices[v3])

                gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])
                
                for show_index in range(50):
                    # step the physics
                    gym.simulate(sim)
                    gym.fetch_results(sim, True)
                
                    # update the viewer
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, True)
                
                
                    gym.sync_frame_time(sim)

                cache_index = 10
                view_cache = []
                stuck_counter = 0

                while end_state_collision:
                    camera_loc, camera_focus = None, None
                    if vp_method == 0:
                        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                                    -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                                    table_dims.z, table_dims.z + drawer_height - 0.02)
                        camera_focus = get_random_loc(0 + 0.3, table_dims.x + 0.3,
                                                      -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                                      table_dims.z, camera_loc.z)
                    elif vp_method == 1:
                    #random sample guided viewpoint selection
                        camera_loc, camera_focus = random_sample_guided_selection(sim, envs[-1], test_cam, scene)
                    elif vp_method == 2:
                        if stuck_counter == 0:
                                #mpc_result = mpc_viewpoint_selection_new_grad(sim, envs[-1], test_cam, scene)
                            mpc_result = new_mpc_based_on_single_point(sim, envs[-1], test_cam, scene)
                        if stuck_counter < 20:
                            camera_loc, camera_focus = mpc_result[stuck_counter]

                                #target_pos = gymapi.Vec3(t1, t2, t3)
                                #target_quat = gymapi.Quat(r1, r2, r3, r4)
                            stuck_counter += 1

                        else:
                            camera_loc, camera_focus = random_sample_guided_selection(sim, envs[-1], test_cam, scene)

                    elif vp_method == 3:
                        if sequence_count <= 1:
                            print ('Decision transformer selection')
                            temp_scene = np.transpose(scene.scene_, (3, 0, 1, 2))
                            temp_scene = np.where(temp_scene == -1, 255, temp_scene)
                            dt_states[0, sequence_count] = temp_scene
                            dt_camera_pose = dt_viewpoint_selection(sim, dt_states, dt_actions, dt_rewards, sequence_count)
                        else:
                            camera_loc, camera_focus = random_sample_guided_selection(sim, envs[-1], test_cam, scene)
                            gym.set_camera_location(test_cam, envs[-1], camera_loc, camera_focus)
                            target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
                            target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r


                    else:
                        camera_loc, camera_focus = mpc_viewpoint_selection_new_grad(sim, envs[-1], test_cam , scene)
                    #gym.set_camera_location(test_cam, envs[-1], camera_loc, camera_focus)
                    #target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
                    #target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r
                    if vp_method == 3:
                        if sequence_count <= 1:
                            target_pos = gymapi.Vec3(dt_camera_pose[4], 
                                                     dt_camera_pose[5],
                                                     dt_camera_pose[6])
                            target_quat = gymapi.Quat(dt_camera_pose[0], 
                                                      dt_camera_pose[1],
                                                      dt_camera_pose[2],
                                                      dt_camera_pose[3])
                        else:
                            target_pos = target_pos
                            target_quat = target_quat

                    r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                    cam_offset_vector = np.array([0.11, 0.0, 0.08])
                    rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
                    converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                             target_pos.y - rot_cam_offset_vector[1],
                                                             target_pos.z - rot_cam_offset_vector[2], 
                                                             ur5e_pose.p.x, 
                                                             ur5e_pose.p.y,
                                                             ur5e_pose.p.z)
                    converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)



                    seed_state = [0.0]*ik_solver2.number_of_joints
                    dof_result = ik_solver2.get_ik(seed_state, 
                                                   converted_coord[0],
                                                   converted_coord[1],
                                                   converted_coord[2],
                                                   converted_quat.x, 
                                                   converted_quat.y,
                                                   converted_quat.z,
                                                   converted_quat.w)
                    if dof_result:
                        end_state_collision = ur5e_in_collision(dof_result, real_offset)
                        if end_state_collision:
                            pass
                        else:
                            print ("found solution")
                            pose_array = get_pose_from_dof(dof_result[:6])

                            rots = []
                            trans = []
                            for t in range(8):
                                rotation = np.array(pose_array[t][1])
                                translation = np.array(pose_array[t][0] + real_offset)
                                rots.append(rotation)
                                trans.append(translation)

                            print (target_pos, target_quat)
                            loc_tracker = target_pos
                            rot_tracker = target_quat
                            final_translation = np.array([target_pos.x, target_pos.y, target_pos.z])
                            final_rotation = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w])

                            need_acquire = True

                            gym.clear_lines(viewer)

                            for obj_id, obj_ins in object_dict.items():
                                if obj_id != 0:
                                    object_vertices, object_faces = obj_ins.get_collision_mesh()

                                    object_lines = []

                                    for v1, v2, v3 in object_faces:
                                        object_lines += list(object_vertices[v1])
                                        object_lines += list(object_vertices[v2])
                                        object_lines += list(object_vertices[v1])
                                        object_lines += list(object_vertices[v3])
                                        object_lines += list(object_vertices[v2])
                                        object_lines += list(object_vertices[v3])

                                    gym.add_lines(viewer, envs[-1], len(object_lines)//6, object_lines, [1, 0, 0])

                            all_lines = []

                            for v1, v2, v3 in scene_faces:
                                all_lines += list(scene_vertices[v1])
                                all_lines += list(scene_vertices[v2])
                                all_lines += list(scene_vertices[v1])
                                all_lines += list(scene_vertices[v3])
                                all_lines += list(scene_vertices[v2])
                                all_lines += list(scene_vertices[v3])

                            gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])


                            for i in range(8):
                                parts_path = ur5e_collision_parts[i]
                                collision_mesh = stl_reader(asset_root + parts_path)
                                collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
                                verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
                                all_lines = []
                                r2 = R.from_quat(rots[i])
                                for t in range(len(verts)):
                                    verts[t] = r2.apply(verts[t])
                                    verts[t] += np.array(trans[i])


                                for v1, v2, v3 in tris:
                                    all_lines += list(verts[v1])
                                    all_lines += list(verts[v2])
                                    all_lines += list(verts[v1])
                                    all_lines += list(verts[v3])
                                    all_lines += list(verts[v2])
                                    all_lines += list(verts[v3])

                                gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])

                            

                            for draw_index in range(400):
                                gym.simulate(sim)
                                gym.fetch_results(sim, True)

                                # update the viewer
                                gym.step_graphics(sim)
                                gym.draw_viewer(viewer, sim, True)


                                gym.sync_frame_time(sim)


                            
                            si = ob.SpaceInformation(space)
                            
                            validityChecker = ur5e_valid(si, real_offset)
                            si.setStateValidityChecker(validityChecker)
                            si.setStateValidityCheckingResolution(0.05)

                            si.setup()


                            col_free_path_found = False

                            trial_counter = 0

                            pathlist = []

                            while trial_counter <= 5:

                                trial_counter += 1

                                start = ob.State(space)
                                if not first_view_obtained:
                                    arm_angle_sample = get_random_arm_angle()
                                    arm_angle_sample_collide = ur5e_in_collision(arm_angle_sample, real_offset)
                                    while arm_angle_sample_collide:
                                        arm_angle_sample = get_random_arm_angle()
                                        arm_angle_sample_collide = ur5e_in_collision(arm_angle_sample, real_offset)
                                    start[0] = arm_angle_sample[0]
                                    start[1] = arm_angle_sample[1]
                                    start[2] = arm_angle_sample[2]
                                    start[3] = arm_angle_sample[3]
                                    start[4] = arm_angle_sample[4]
                                    start[5] = arm_angle_sample[5]
                                else:
                                    start[0] = track_last_pose[0]
                                    start[1] = track_last_pose[1]
                                    start[2] = track_last_pose[2]
                                    start[3] = track_last_pose[3]
                                    start[4] = track_last_pose[4]
                                    start[5] = track_last_pose[5]



                                goal = ob.State(space)
                                goal[0] = dof_result[0]
                                goal[1] = dof_result[1]
                                goal[2] = dof_result[2]
                                goal[3] = dof_result[3]
                                goal[4] = dof_result[4]
                                goal[5] = dof_result[5]

                                pdef = ob.ProblemDefinition(si)
                                pdef.setStartAndGoalStates(start, goal)
                                
                                shortestPathObjective = ob.PathLengthOptimizationObjective(si)
                                pdef.setOptimizationObjective(shortestPathObjective)

                                optimizingPlanner = og.RRTConnect(si)

                                optimizingPlanner.setProblemDefinition(pdef)

                                optimizingPlanner.setRange(1000000)

                                optimizingPlanner.setup()

                                temp_res = optimizingPlanner.solve(100)

                                res = None
                                
                                path = pdef.getSolutionPath()

                                pathlist = []

                                for i in range(path.getStateCount()):
                                    state = path.getState(i)
                                    pathlist.append((state[0], state[1],
                                                     state[2], state[3],
                                                     state[4], state[5]))

                                res = pathlist

                                temp_difference = 0
                                
                                for dof_index in range(len(pathlist[-1])):
                                    temp_difference += (pathlist[-1][dof_index] - dof_result[dof_index])**2


                                col_free_path_found = (temp_difference < 1e-4)

                                if col_free_path_found: break

                            
                            if col_free_path_found:
                                _body_tensor = gym.acquire_rigid_body_state_tensor(sim)
                                body_tensor = gymtorch.wrap_tensor(_body_tensor)
                                flag = False
                                need_acquire = True
                                    
                                print ("Solution found, start moving arm")

                                for element in pathlist:
                                    print ([round(x*1.0/math.pi*180,2) for x in element])


                                print ('End solution')

                                track_last_pose = dof_result


                                first_view_obtained = True

                                track_angle_path.append(pathlist)

                                for pathseg in pathlist:

                                    for path_seg_index in range(400):
                                        # step the physics
                                        gym.simulate(sim)
                                        gym.fetch_results(sim, True)

                                        # update the viewer
                                        gym.step_graphics(sim)
                                        gym.draw_viewer(viewer, sim, True)


                                        gym.sync_frame_time(sim)

                                        gym.set_dof_target_position(envs[-1], spj, pathseg[0]) 
                                        gym.set_dof_target_position(envs[-1], slj, pathseg[1]) 
                                        gym.set_dof_target_position(envs[-1], ej,  pathseg[2]) 
                                        gym.set_dof_target_position(envs[-1], wj1, pathseg[3]) 
                                        gym.set_dof_target_position(envs[-1], wj2, pathseg[4]) 
                                        gym.set_dof_target_position(envs[-1], wj3, pathseg[5])

                                #assign tensor here

                                for assign_pose_index in range(7):

                                    temp_trans, temp_rot = pose_array[assign_pose_index]

                                    state_tensor[assign_pose_index+2][0] = temp_trans[0]
                                    state_tensor[assign_pose_index+2][1] = temp_trans[1]
                                    state_tensor[assign_pose_index+2][2] = temp_trans[2]
                                    state_tensor[assign_pose_index+2][3] = temp_rot[0]
                                    state_tensor[assign_pose_index+2][4] = temp_rot[1]
                                    state_tensor[assign_pose_index+2][5] = temp_rot[2]
                                    state_tensor[assign_pose_index+2][6] = temp_rot[3]

                                for tensor_assign_index in range(400):
                                    gym.simulate(sim)
                                    gym.fetch_results(sim, True)

                                    # update the viewer
                                    gym.step_graphics(sim)
                                    gym.draw_viewer(viewer, sim, True)


                                    gym.sync_frame_time(sim)



                                break
                            else:
                                end_state_collision = True

                            #for t in range(100000000):
                            #    gym.simulate(sim)
                            #    gym.fetch_results(sim, True)

                            #    # update the viewer
                            #    gym.step_graphics(sim)
                            #    gym.draw_viewer(viewer, sim, True)


                            #    gym.sync_frame_time(sim)


                viewpoint_counter += 1
                
    counter += 1


    #check the range of movement
    #radius = 4
    #rand_x = np.random.random()*radius - radius/2.0
    #rand_y = np.random.random()*radius - radius/2.0
    #rand_z = max(0, np.random.random()*radius)
    #converted_coord = global_coord_converter(rand_x, rand_y, rand_z, 0, 0, 0)
    #ik_solver = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
    #seed_state = [0.0]*ik_solver.number_of_joints
    #test_quat = gymapi.Quat()
    #quat_result = test_quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi)
    ##print (quat_result)
    #dof_result = ik_solver.get_ik(seed_state, 
    #                              converted_coord[0],
    #                              converted_coord[1],
    #                              converted_coord[2],
    #                              quat_result.x, 
    #                              quat_result.y,
    #                              quat_result.z,
    #                              quat_result.w)
    #if dof_result:
    #    print ("here")
    #    print (converted_coord)
    #    gym.add_lines(viewer, envs[-1], 1, [rand_x,
    #                                       rand_y,
    #                                       rand_z, 
    #                                       0, 0, 0], [1, 0, 0])
    #    maxi = max(maxi, math.sqrt(rand_x**2 + rand_y**2 + rand_z**2))

    #counter += 1
    

    gym.sync_frame_time(sim)
#*************************************************************************************************#
#with open("track_object.txt", 'w') as f:
#    for i in range(num_of_objects):
#        rotation, translation, scale = object_status[i]
#        f.write(' '.join([str(x) for x in rotation]))
#        f.write(" ")
#        f.write(' '.join([str(x) for x in translation]))
#        f.write(" ")
#        f.write(str(scale))
#        f.write(" ")
#        f.write(str(object_index[i]))
#        f.write("\n")
#
#with open("track_camera.txt", 'w') as f:
#    for element in track_cameras:
#        print (element)
#        f.write(' '.join([str(x) for x in [element[0].x, element[0].y, element[0].z, element[0].w]]))
#        f.write(" ")
#        f.write(' '.join([str(x) for x in [element[1].x, element[1].y, element[1].z]]))
#        f.write('\n')


#write out all paths
path_file_name = 'saved_data_paper/' + str(vp_method) + '/' + str(env_id) + '/track_dof_path.txt'
with open(path_file_name, 'w') as f:
    for path in track_angle_path:
        f.write('start**********************\n')
        for element in path:
            f.write(str(element) + '\n')
        f.write('end***********************\n')




print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


