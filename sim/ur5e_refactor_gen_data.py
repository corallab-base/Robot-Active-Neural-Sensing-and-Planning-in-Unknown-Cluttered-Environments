#
# File:          ur5e_refactor.py
# Brief:         main program for ur5e simulation
# Author:        Hanwen Ren -- ren221@purdue.edu
# Date:          2022-01-04
# Last Modified: 2022-02-28
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


file_dir = os.path.dirname(__file__)
print (file_dir)
util_dir = os.path.join(file_dir, '../util')
sys.path.append(util_dir)
from stl_reader import stl_reader
from obj_reader import obj_reader
from pc_extractor import pc_extractor
from pc_extractor import visualize_scene
from global_scene import global_scene

#define parameters
#*************************************************************************************************#
#global settings
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

#env settings
table_dims = gymapi.Vec3(np.random.rand()*0.5 + 0.5, np.random.rand()*0.4 + 1.1, 
                         np.random.rand()*0.15 + 0.05)
#table_dims = gymapi.Vec3(0.5, 1.1, 0.05)
piece_width = 0.03
min_num_of_objects = 3
max_num_of_objects = 10
max_scaling_factor = 0
fall_height = table_dims.z
max_drawer_height = 0.3
min_drawer_height = 0.7
ADD_COVER = True
#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
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
            if raw_image[i][j] != mini:
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
            else:
                new_image[i][j][0] = 65535
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
#*************************************************************************************************#


#initialize gym
#*************************************************************************************************#
gym = gymapi.acquire_gym()
#*************************************************************************************************#

#parse arguments

#*************************************************************************************************#
args = gymutil.parse_arguments(description="ur5e example", custom_parameters = [{'name':'--env_id', 'type':int, 'help':'env_id', 'default':0}])
env_id = int(args.env_id)
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
ur5e_asset_file = "urdf/ur5e/ur5e_final_no_grasp.urdf"
ur5e_collision_parts = ["urdf/ur5e/meshes/collision/base.stl",
                        "urdf/ur5e/meshes/collision/shoulder.stl",
                        "urdf/ur5e/meshes/collision/upperarm.stl",
                        "urdf/ur5e/meshes/collision/forearm.stl",
                        "urdf/ur5e/meshes/collision/wrist1.stl",
                        "urdf/ur5e/meshes/collision/wrist2.stl",
                        "urdf/ur5e/meshes/collision/wrist3.stl"]

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
                  R.from_euler('z',  [-90], degrees = True)]
ur5e_translations = [[0, 0, 0], 
                     [0, 0, 0],
                     [0, -0.138, 0],
                     [0, -0.007, 0],
                     [0, 0.127, 0],
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
with open("../assets/urdf/ur5e/ur5e_calcversion.urdf") as f:
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

back_cover_dims = gymapi.Vec3(0.1, table_dims.y*0.8 + piece_width, table_dims.z + drawer_height)
back_cover_asset = gym.create_box(sim, back_cover_dims.x,
                                       back_cover_dims.y,
                                       back_cover_dims.z,
                                       asset_options)

asset_options.fix_base_link = False
object_assets = []
test_assets = []
for ob in object_asset_files:
    object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
asset_options.fix_base_link = True
test_assets.append(gym.load_asset(sim, asset_root, object_asset_files[0], asset_options))
asset_options.fix_base_link = False

#*************************************************************************************************#


#initial pose
#*************************************************************************************************#
ur5e_pose = gymapi.Transform()
#ur5e_pose.p = gymapi.Vec3(np.random.rand()*0.4 - 0.2, np.random.rand()*0.4 - 0.2, 0.0)
ur5e_pose.p = gymapi.Vec3(100, 100, 0)
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

back_cover_pose = gymapi.Transform()
back_cover_pose.p = gymapi.Vec3(0.3, 0, (table_dims.z + drawer_height)/2.0)

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


    color = gymapi.Vec3(170/255.0, 170/255.0, 170/255.0)
    w1 = gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
    gym.set_rigid_body_color(envs[-1], w1, 0, gymapi.MESH_VISUAL_AND_COLLISION,color)
    w2 = gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
    gym.set_rigid_body_color(envs[-1], w2, 0, gymapi.MESH_VISUAL_AND_COLLISION,color)
    w3 = gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
    gym.set_rigid_body_color(envs[-1], w3, 0, gymapi.MESH_VISUAL_AND_COLLISION,color)

    w5 = gym.create_actor(envs[-1], back_cover_asset, back_cover_pose, "back_cover" + str(i), 0, 1)
    gym.set_rigid_body_color(envs[-1], w5, 0, gymapi.MESH_VISUAL_AND_COLLISION,color)
    if ADD_COVER:
        w4 = gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)
        gym.set_rigid_body_color(envs[-1], w4, 0, gymapi.MESH_VISUAL_AND_COLLISION,color)

    object_index = np.random.randint(len(object_asset_files), size=num_of_objects)
    chosen_object.append(object_index)
    #object_index = [0]
    object_loc = np.random.normal(0, table_dims.x/10.0, size=(2, num_of_objects))
    object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = num_of_objects)/10.0 + 1.0

    chosen_scale.append(object_scaling_factor)
    object_handles = []

    with open("object_name.txt", 'a') as f:
        for k in range(num_of_objects):
            f.write(object_asset_files[object_index[k]])
    

    #set up objects
    for k in range(num_of_objects):
        object_pose = gymapi.Transform()
        object_pose.p = get_random_loc(0.3 + table_dims.x*0.2, 0.3 + table_dims.x*0.8,
                                       -table_dims.y*0.4, table_dims.y*0.4,
                                       table_dims.z, table_dims.z + drawer_height*0.5)
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
    viewpoint_candidate = gymapi.Vec3(3, 0, 0.6)
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
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(-1.0, 0.0, 0.0))
gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(1.0, 0.0, 0.0))
#*************************************************************************************************#


#*************************************************************************************************#
counter = 0
viewpoint_counter = 0
ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
test_cam = gym.create_camera_sensor(envs[-1], camera_props)
#body_cam_handles.append(test_cam)
flag = True
need_acquire = True
acquire_counter = 0
drawn = False
loc_tracker = None
rot_tracker = None
object_status = []
track_cameras = []
sequence_count = 0
final_rotation, final_translation = None, None
object_dict = {}
scene = global_scene(1.0, 1.2, 0.85, np.array([0.3, -0.6, 0.05]), table_dims.x, table_dims.y*0.8 - 0.04, drawer_height - 0.02, table_dims.z - 0.05)
coverage_score = 0
while not gym.query_viewer_has_closed(viewer):

    if coverage_score >= 0.95 or sequence_count >= 20: break

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    offset = 4

    if counter > 1000:
        if need_acquire:
            if acquire_counter > 500:
                _body_tensor = gym.acquire_rigid_body_state_tensor(sim)
                body_tensor = gymtorch.wrap_tensor(_body_tensor)

                object_rots = []
                object_trans = []
                for i in range(len(object_handles)):
                    element = object_handles[i]
                    all_state = gym.get_actor_rigid_body_states(envs[-1], element, 1)
                    rotation = np.array(all_state[0][0][1])
                    translation = np.array(all_state[0][0][0])
                    rotation = np.array(rotation.item())
                    translation = np.array(translation.item())
                    object_rots.append(rotation)
                    object_trans.append(translation)
                    if i not in object_status:
                        object_status.append([rotation, translation, object_scaling_factor[i]])
                    r1 = R.from_quat(rotation)
                    tf = fcl.Transform(r1.as_matrix(), translation)
                    object_collision_models.append(fcl.CollisionObject(object_collision_lib[i], tf))

                ur5e_self_col = []
                rots = []
                trans = []
                for t in range(7):
                    rotation = np.array(body_tensor[t+2][3:7])
                    translation = np.array(body_tensor[t+2][:3])
                    rots.append(rotation)
                    trans.append(translation)
                    r1 = R.from_quat(rotation)
                    tf = fcl.Transform(r1.as_matrix(), translation)
                    ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))

                #visualize mesh in fcl BVHModel() for debugging usage
                #if not drawn:
                #    drawn = True
                #    for i in range(7):
                #        parts_path = ur5e_collision_parts[i]
                #        collision_mesh = stl_reader(asset_root + parts_path)
                #        collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
                #        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
                #        all_lines = []
                #        r2 = R.from_quat(rots[i])
                #        for t in range(len(verts)):
                #            verts[t] = r2.apply(verts[t])
                #            verts[t] += np.array(trans[i])


                #        for v1, v2, v3 in tris:
                #            all_lines += list(verts[v1])
                #            all_lines += list(verts[v2])
                #            all_lines += list(verts[v1])
                #            all_lines += list(verts[v3])
                #            all_lines += list(verts[v2])
                #            all_lines += list(verts[v3])

                #        gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])

                #    for i in range(num_of_objects):
                #        file_path = object_collision_files[chosen_object[-1][i]]
                #        collision_mesh = obj_reader(asset_root + file_path)
                #        collision_mesh.set_offset(object_offset[chosen_object[-1][i]])
                #        collision_mesh.set_scale(chosen_scale[-1][i])
                #        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()

                #        all_lines = []
                #        r2 = R.from_quat(object_rots[i])
                #        for t in range(len(verts)):
                #            verts[t] = r2.apply(verts[t])
                #            verts[t] += np.array(object_trans[i])


                #        for v1, v2, v3 in tris:
                #            all_lines += list(verts[v1])
                #            all_lines += list(verts[v2])
                #            all_lines += list(verts[v1])
                #            all_lines += list(verts[v3])
                #            all_lines += list(verts[v2])
                #            all_lines += list(verts[v3])

                #        gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])

                #request = fcl.CollisionRequest()
                #result = fcl.CollisionResult()
                #self_collision_flag = False
                #for t in range(7):
                #    if t != 0:
                #        if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
                #            self_collision_flag = True
                #            break
                #    for q in range(7):
                #        if q < t-1 or q > t + 1:
                #            if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                #                self_collision_flag = True
                #                break
                #print ("self collision: {0}".format(self_collision_flag))


                #env_collision_flag = False
                ##manager1 stores robot collision model
                #manager1 = fcl.DynamicAABBTreeCollisionManager()
                #manager1.registerObjects(ur5e_self_col)
                #manager1.setup()
                ##manager2 stores environment collision model
                #manager2 = fcl.DynamicAABBTreeCollisionManager()
                #manager2.registerObjects(object_collision_models)
                #manager2.setup()

                ##check collision between manager1 and manager2
                #req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
                #rdata = fcl.CollisionData(request = req)
                #manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
                #if rdata.result.is_collision:
                #    env_collision_flag = True

                #print ("env collision: {0}".format(env_collision_flag))

                gym.render_all_camera_sensors(sim)
                self_collision_flag = False
                env_collision_flag = False
                if (not self_collision_flag and not env_collision_flag):
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

                            write_to_image(color_image, "data_generation/color_test" \
                                           + str(sequence_count) + '_' + str(q) + ".jpg")
                            write_to_depth_image(depth_image, "data_generation/depth_test" \
                                                 + str(sequence_count) + '_' + str(q) + ".png")
                            write_to_seg_image(seg_image, "data_generation/seg_test" \
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
                                    res = pc_extractor(new_rgb_image, new_depth_image, new_seg_image, 
                                                   new_cam_rotation, new_cam_translation, object_dict, True)
                                    #print (len(object_dict))

                                    visualize_scene(object_dict)

                                    file_prefix = 'scene_generation_0602/env' + str(env_id) + "_sequence" + str(sequence_count)

                                    coverage_score = scene.register_camera_view(list(new_cam_rotation), \
                                                               list(new_cam_translation), \
                                                               new_depth_image, object_dict, file_prefix)

                                    scene.vis_scene()

                                    sequence_count += 1

                            #if q == 0:
                            #    pc_extractor = point_cloud_extractor("../sim/captured_images/color_test4.jpg",
                            #                                         "../sim/captured_images/depth_test4.png",
                            #                                         "../sim/captured_images/seg_test4.png",
                            #                                         [temp_pos.x,
                            #                                          temp_pos.y, 
                            #                                          temp_pos.z], 
                            #                                         [temp_quat.x,
                            #                                          temp_quat.y,
                            #                                          temp_quat.z,
                            #                                          temp_quat.w])

                            #    point_clouds = pc_extractor.get_point_cloud()
                            #    all_lines = []
                            #    for c1, c2, c3 in np.asarray(point_clouds.points):
                            #        all_lines.append(c1)
                            #        all_lines.append(c2)
                            #        all_lines.append(c3)
                            #        all_lines.append(c1+0.001)
                            #        all_lines.append(c2+0.001)
                            #        all_lines.append(c3+0.001)
                            #    gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])



                need_acquire = False
                flag = True
                acquire_counter = 0
                print ("end acquire")
            else:
                acquire_counter += 1
        else:
            if counter % 2 == 0 and flag:
                camera_loc = get_random_loc(0 + 0.2, 0.32,
                                            -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                            table_dims.z, table_dims.z + drawer_height - 0.02)
                camera_focus = get_random_loc(0 + 0.3, table_dims.x + 0.3,
                                            -table_dims.y*0.4 + 0.02, table_dims.y*0.4 - 0.02,
                                            table_dims.z, camera_loc.z)
                #camera_loc = gymapi.Vec3(0.6, 0, 0.4)
                #camera_focus = gymapi.Vec3(0.6, 0.7, 0.0)

                gym.set_camera_location(test_cam, envs[-1], 
                                        camera_loc, 
                                        camera_focus)
                target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
                target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r
                track_cameras.append([target_quat, target_pos])
                r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                cam_offset_vector = np.array([0.11, 0, 0.08])
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
                    print ("found solution")
                    print (target_pos, target_quat)
                    loc_tracker = target_pos
                    rot_tracker = target_quat
                    final_translation = np.array([target_pos.x, target_pos.y, target_pos.z])
                    final_rotation = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                    #gym.add_lines(viewer, envs[-1], 1, [target_pos.x, 
                    #                                    target_pos.y, 
                    #                                    target_pos.z, 
                    #                                    target_pos.x + 2*rot_cam_offset_vector[0],
                    #                                    target_pos.y + 2*rot_cam_offset_vector[1],
                    #                                    target_pos.z + 2*rot_cam_offset_vector[2]],
                    #                                    [1, 0, 0])
                    gym.set_dof_target_position(envs[-1], spj, dof_result[0]) 
                    gym.set_dof_target_position(envs[-1], slj, dof_result[1]) 
                    gym.set_dof_target_position(envs[-1], ej,  dof_result[2]) 
                    gym.set_dof_target_position(envs[-1], wj1, dof_result[3]) 
                    gym.set_dof_target_position(envs[-1], wj2, dof_result[4]) 
                    gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
                    
                    _body_tensor = gym.acquire_rigid_body_state_tensor(sim)
                    body_tensor = gymtorch.wrap_tensor(_body_tensor)
                    flag = False
                    need_acquire = True

                


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




print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


