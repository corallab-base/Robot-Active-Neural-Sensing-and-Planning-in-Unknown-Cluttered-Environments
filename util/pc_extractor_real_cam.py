import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import math

file_dir = os.path.dirname(__file__)
root_dir = os.path.join(file_dir, '..')
sys.path.append(root_dir)
root_dir = os.path.join(file_dir, '../../util')
sys.path.append(root_dir)
from camera_module.camera_view import camera
from tools import object_completion_network
from YCB_object import YCB_object

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
    offset = R.from_quat(rotation).apply([0, 0, 0.07])
    return [-0.4 - x + offset[0], -0.3 - y + offset[1], z + offset[2]]




def visualize_scene(object_dict, flag, bg):
    all_data = []

    #env_points = []
    #env_colors = []
    #for t in range(300, 1000):
    #    for k in range(-400, 400):
    #        env_points.append([t*0.001, k*0.001, 0.15])
    #        env_colors.append([255, 0, 0])
    

    environment_pc = []
    environment_colors = []
    for i in range(0, 65):
        for j in range(-42, 43):
            environment_pc.append([i*0.01 + 0.3, j*0.01, 0.15])
            environment_pc.append([i*0.01 + 0.3, j*0.01, 0.82])
            environment_colors.append([0.5, 0.5, 0.5])
            environment_colors.append([0.5, 0.5, 0.5])
    for i in range(0, 65):
        for k in range(15, 81):
            environment_pc.append([i*0.01 + 0.3, -0.42, k*0.01])
            environment_pc.append([i*0.01 + 0.3, 0.42, k*0.01])
            environment_colors.append([0.5, 0.5, 0.5])
            environment_colors.append([0.5, 0.5, 0.5])
    env_pc = o3d.geometry.PointCloud()
    env_pc.points = o3d.utility.Vector3dVector(environment_pc)
    env_pc.colors = o3d.utility.Vector3dVector(environment_colors)




    for ids, object_handler in object_dict.items():
        seen_data = object_handler.get_seen()
        comp_data = object_handler.get_completion()

        if bg == False:
            if ids == 0:
                continue

    
        pcd_seen = o3d.geometry.PointCloud()
        pcd_seen.points = o3d.utility.Vector3dVector([list(x) for x in seen_data.keys()])
        pcd_seen.colors = o3d.utility.Vector3dVector([list(x) for x in seen_data.values()])
        all_data.append(pcd_seen)
        average_colors = np.array([list(x) for x in seen_data.values()])
        average_colors = np.mean(average_colors, axis = 0)

        if flag and len(comp_data) > 0:
            pcd_comp = o3d.geometry.PointCloud()
            average_location = np.array([list(x) for x in comp_data.keys()])
            average_location = np.mean(average_location, axis = 0)
            print (average_location)
            if average_location[1] < 0 and average_location[1] <= 0.6:
                points = [list(x) for x in comp_data.keys()]
                for i in range(len(points)):
                    points[i][1] += 0.02
                pcd_comp.points = o3d.utility.Vector3dVector(points)
            else:
                pcd_comp.points = o3d.utility.Vector3dVector([list(x) for x in comp_data.keys()])
            comp_colors = np.tile(average_colors, (len(comp_data.keys()), 1))
            pcd_comp.colors = o3d.utility.Vector3dVector(comp_colors)
            #o3d.visualization.draw_geometries([pcd_comp, mesh_frame])
            all_data.append(pcd_comp)
   
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[0, 0, 0])
    all_data.append(mesh_frame)
    if not bg:
        all_data.append(env_pc)

    o3d.visualization.draw_geometries(all_data)


def save_object(object_dict, file_prefix):
    seen_data = []
    comp_data = []
    seen_color = []
    comp_color = []

    for ids, object_handler in object_dict.items():
        seen_obj = object_handler.get_seen()
        comp_obj = object_handler.get_completion()

        for element in seen_obj.keys():
            seen_data.append(list(element))
        for element in comp_obj.keys():
            comp_data.append(list(element))
        for element in seen_obj.values():
            seen_color.append(list(element))
        for element in comp_obj.values():
            comp_color.append(list(element))

    seen_data = np.array(seen_data)
    comp_data = np.array(comp_data)
    seen_color = np.array(seen_color)
    comp_color = np.array(comp_color)

    with open(file_prefix + "_seen_object.npy", 'wb') as f:
        np.save(f, seen_data)
    
    with open(file_prefix + "_comp_object.npy", 'wb') as f:
        np.save(f, comp_data)

    with open(file_prefix + "_seen_color.npy", 'wb') as f:
        np.save(f, seen_color)

    with open(file_prefix + "_comp_color.npy", 'wb') as f:
        np.save(f, comp_color)


def save_object_no_bg(object_dict, file_prefix):
    seen_data = []
    comp_data = []
    seen_color = []
    comp_color = []

    for ids, object_handler in object_dict.items():
        if ids != 0:
            seen_obj = object_handler.get_seen()
            comp_obj = object_handler.get_completion()
            
            average_colors = np.array([list(x) for x in seen_obj.values()])
            average_colors = np.mean(average_colors, axis = 0)
            
            average_location = np.array([list(x) for x in comp_obj.keys()])
            average_location = np.mean(average_location, axis = 0)
            print (average_location)
            if len(comp_obj.keys()):
                if average_location[1] < 0 and average_location[0] <= 0.6:
                    print ('here')
                    points = [list(x) for x in comp_obj.keys()]
                    for i in range(len(points)):
                        points[i][1] += 0.02
                    comp_data += points
                else:
                    print ('jere')
                    for element in comp_obj.keys():
                        comp_data.append(list(element))

            for element in seen_obj.keys():
                seen_data.append(list(element))
            #for element in comp_obj.keys():
            #    comp_data.append(element)
            for element in seen_obj.values():
                seen_color.append(list(element))
            comp_colors = np.tile(average_colors, (len(comp_obj.keys()), 1))
            print (comp_colors)
            for element in comp_colors:
                comp_color.append(list(element))

    seen_data = np.array(seen_data)
    comp_data = np.array(comp_data)
    seen_color = np.array(seen_color)
    comp_color = np.array(comp_color)

    with open(file_prefix + "_seen_object_nbg.npy", 'wb') as f:
        np.save(f, seen_data)
    
    with open(file_prefix + "_comp_object_nbg.npy", 'wb') as f:
        np.save(f, comp_data)

    with open(file_prefix + "_seen_color_nbg.npy", 'wb') as f:
        np.save(f, seen_color)

    with open(file_prefix + "_comp_color_nbg.npy", 'wb') as f:
        np.save(f, comp_color)


    


class pc_extractor_real_cam:

    #constructor
    def __init__(self, color_image, depth_image, seg_image, cam_rotation, cam_translation, object_dict, comp_flag):


        environment_pc = []
        environment_colors = []
        for i in range(0, 57):
            for j in range(-46, 47):
                environment_pc.append([i*0.01 + 0.3, j*0.01, 0.15])
                environment_pc.append([i*0.01 + 0.3, j*0.01, 0.82])
                environment_colors.append([0.5, 0.5, 0.5])
                environment_colors.append([0.5, 0.5, 0.5])
        for i in range(0, 57):
            for k in range(15, 83):
                environment_pc.append([i*0.01 + 0.3, -0.46, k*0.01])
                environment_pc.append([i*0.01 + 0.3, 0.46, k*0.01])
                environment_colors.append([0.5, 0.5, 0.5])
                environment_colors.append([0.5, 0.5, 0.5])


        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_image)
        seg_raw = o3d.geometry.Image(seg_image)
       
        m, n = np.asarray(seg_raw).shape
        offset = np.array(cam_translation)
        rot = R.from_quat(cam_rotation)

        erosion_times = 10
        look_up_array = np.asarray(seg_raw)
        for k in range(erosion_times):
            new_array = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    if look_up_array[i][j] != 0:
                        counter = 0
                        target_id = look_up_array[i][j]
                        for ii in range(-1, 2):
                            for jj in range(-1, 2):
                                res_i = i + ii
                                res_j = j + jj
                                if res_i != i or res_j != j:
                                    if 0 <= res_i < m and 0 <= res_j < n and look_up_array[res_i][res_j] == target_id:
                                        counter += 1
                        if counter == 8:
                            new_array[i][j] = target_id
                        else:
                            new_array[i][j] = 0
            for i in range(m):
                for j in range(n):
                    look_up_array[i][j] = new_array[i][j]
        

        #plt.imshow(depth_raw)
        #plt.show()

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=0.4, origin=[0, 0, 0])


        object_list = set()
        object_list.add(0)
        for i in range(m):
            for j in range(n):
                object_id = np.asarray(seg_raw)[i][j]
                if object_id != 0 and object_id not in object_list:
                    object_list.add(object_id)
        print (object_list)

        all_data = []

        self.completion_network = object_completion_network()
        
        for ids in object_list:
            object_handler = None
            #if ids not in object_dict:
            #    object_handler = YCB_object()
            #    object_dict[ids] = object_handler
            #else:
            #    object_handler = object_dict[ids]
            color_copy = o3d.geometry.Image(color_raw)
            depth_copy = o3d.geometry.Image(depth_raw)
            for i in range(m):
                for j in range(n):
                    if np.asarray(seg_raw)[i][j] != ids:
                        np.asarray(color_copy)[i][j][0] = 0
                        np.asarray(color_copy)[i][j][1] = 0
                        np.asarray(color_copy)[i][j][2] = 0
                        np.asarray(depth_copy)[i][j] = 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_copy, depth_copy,
                                      convert_rgb_to_intensity = False)

            param = o3d.camera.PinholeCameraIntrinsic(1280, 720, 910, 910, 641, 353)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image,
                            o3d.camera.PinholeCameraIntrinsic(
                            param))
            #all_data.append(pcd)
            pcd_data = np.array(pcd.points, dtype = np.float32)
            pcd_color = np.array(pcd.colors, dtype = np.float32)
            temp_sets = set()
            new_data = []
            new_color = []
            num, dim = pcd_data.shape

            for i in range(num):
                element = np.round(pcd_data[i], 3)
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))
                    new_data.append(element)
                    new_color.append(pcd_color[i])
            if (len(new_data) < 2): continue

            #for element in np.round(pcd_data, 3):
            #    if tuple(element) not in temp_sets:
            #        temp_sets.add(tuple(element))
            #        new_data.append(element)
            pcd_data = np.array(new_data)


            pcd_data[:, [0,1,2]] = pcd_data[:, [2, 0, 1]]
            pcd_data[:, 1] *= -1
            pcd_data[:, 2] *= -1
            
            pcd_data = rot.apply(pcd_data)
            pcd_data += offset

            temp_sets = {}
            num, dim = pcd_data.shape
            for i in range(num):
                element = np.round(pcd_data[i], 3)
                if tuple(element) not in temp_sets:
                    temp_sets[tuple(element)] = new_color[i]


            if ids == 0:
                if 0 in object_dict:
                    object_dict[0].add_seen(temp_sets)
                    object_handler = object_dict[0]
                else:
                    object_handler = YCB_object()
                    object_dict[0] = object_handler
                    object_handler.add_seen(temp_sets)
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector([list(x) for x in object_handler.get_seen().keys()])
                temp_pcd.colors = o3d.utility.Vector3dVector([list(x) for x in object_handler.get_seen().values()])
                all_data.append(temp_pcd)
                continue

            print (ids)

            object_translation_from_pc = np.mean(pcd_data, axis = 0)
            #create pcd
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(pcd_data)
            new_pcd.colors = o3d.utility.Vector3dVector(new_color)
            #all_data.append(new_pcd)

            #find same object
            track_same_object = []
            for obj_id, obj_ins in object_dict.items():
                if obj_id != 0:
                    if obj_ins.is_part_of_current_pc(new_pcd):
                        track_same_object.append(obj_id)
            
            if track_same_object != []:
                parent_id = track_same_object[0]
                object_handler = object_dict[parent_id]
                for same_object_id in track_same_object[1:]:
                    seen_part = object_dict[same_object_id].get_seen()
                    object_handler.add_seen(seen_part)

                #since they are merged, we need to delete them
                for same_object_id in track_same_object[1:]:
                    del object_dict[same_object_id]

                print ("found same object")
                print (object_handler)
            else:
                #find min distance between objects
                min_distance = sys.maxsize
                if object_handler == None:
                    for obj_id, obj_ins in object_dict.items():
                        if obj_id != 0:
                            temp_distance = obj_ins.get_distance_between_center(object_translation_from_pc)
                            if temp_distance < min_distance:
                                min_distance = temp_distance
                                object_handler = obj_ins

                #need to create new object
                if min_distance > 1e-6:
                    temp_id = np.random.randint(1, 100)
                    while temp_id in object_dict:
                        temp_id = np.random.randint(1, 100)
                    object_handler = YCB_object()
                    object_dict[temp_id] = object_handler

            object_handler.add_seen(temp_sets)
            if not comp_flag:
                continue
            else:
                if len(object_handler.get_seen()) < 1856:
                    seen_data = object_handler.get_seen()

                    pcd_seen = o3d.geometry.PointCloud()
                    pcd_seen.points = o3d.utility.Vector3dVector([list(x) for x in seen_data.keys()])
                    pcd_seen.colors = o3d.utility.Vector3dVector([list(x) for x in seen_data.values()])
                    all_data.append(pcd_seen)

                    print (len(seen_data))
                    continue

                ids -= 1

                pcd_data = [list(x) for x in object_handler.get_seen().keys()]
                pcd_color = [list(x) for x in object_handler.get_seen().values()]
                pcd_data = np.array(pcd_data)
                object_translation_from_pc = np.mean(pcd_data, axis = 0)

                pcd_data = pcd_data - object_translation_from_pc
                pcd_data = pcd_data / 0.24

                pc_check1 = o3d.geometry.PointCloud()
                pc_check1.points = o3d.utility.Vector3dVector(pcd_data)
                #o3d.visualization.draw_geometries([mesh_frame, pc_check1])
                
                pcd_data = self.completion_network.complete(np.asarray([pcd_data]))
                
                pcd_data = pcd_data * 0.24
                pcd_data = pcd_data + object_translation_from_pc

                pc_check2 = o3d.geometry.PointCloud()
                pc_check2.points = o3d.utility.Vector3dVector(pcd_data)
                #o3d.visualization.draw_geometries([mesh_frame, pc_check2])

                temp_sets = set()
                new_data = []
                for element in np.round(pcd_data, 3):
                    if tuple(element) not in temp_sets:
                        temp_sets.add(tuple(element))
                        new_data.append(element)
                object_handler.completion_ = set()
                object_handler.add_completion(temp_sets)

                #debugging purpose
                temp_pc = o3d.geometry.PointCloud()
                seen_points = [list(x) for x in object_handler.get_seen().keys()]
                seen_colors = [list(x) for x in object_handler.get_seen().values()]
                comp_points = [list(x) for x in object_handler.get_completion().keys()]
                comp_colors = [list(x) for x in object_handler.get_completion().values()]
                temp_pc.points = o3d.utility.Vector3dVector(seen_points + comp_points)
                temp_pc.colors = o3d.utility.Vector3dVector(seen_colors + comp_colors)
                all_data.append(temp_pc)

        #dim_x, dim_y, dim_z = 100, 150, 50
        #offset = np.array([0, -0.75, 0])
        #scene_voxel = np.arange(dim_x * dim_y * dim_z * 3).reshape(dim_x * dim_y * dim_z, 3)
        #scene_voxel = scene_voxel.astype(np.float32)
        #for i in range(dim_x):
        #    for j in range(dim_y):
        #        for k in range(dim_z):
        #          scene_voxel[i*dim_y*dim_z + j*dim_z + k] = np.array([i,j,k], dtype = np.float32)*0.01 + \
        #                                                     np.array([0.005, 0.005, 0.005]) + \
        #                                                     offset
        #scene_pcd = o3d.geometry.PointCloud()
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_voxel)
        #scene_pcd.paint_uniform_color([0.9, 0.9, 0.9])

        #for items in all_data:
        #    for points in items.points:
        #        p1, p2, p3 = points
        #        ind1 = round(p1 / 0.01)
        #        ind2 = round(p2 / 0.01) + 75
        #        ind3 = round(p3 / 0.01)
        #        scene_pcd.colors[ind1 * dim_y * dim_z + ind2 * dim_z + ind3] = np.array([1, 1, 0])

        #cam = camera(np.array([0.190782, -0.272692, 0.333791]), np.array([0.000005, 0.225503, 0.250454, 0.941499]))
        #cam.build()

        #for i in range(len(scene_pcd.points)):
        #    query_point = scene_pcd.points[i]
        #    flag, depth, location = cam.inside_frame(query_point)
        #    if flag:
        #        loc_x, loc_y = location
        #        loc_x += 512
        #        loc_y += 512
        #        if 0 <= loc_x < 1024 and 0 <= loc_y < 1024:
        #            if depth*1000 < np.asarray(depth_raw)[loc_y][loc_x]:
        #                if scene_pcd.colors[i][0] == 0.9:
        #                    scene_pcd.colors[i] = np.array([1, 0, 0])
        #            else:
        #                if scene_pcd.colors[i][0] == 0.9:
        #                    scene_pcd.colors[i] = np.array([0, 0, 1])
        #
    
        env_pc = o3d.geometry.PointCloud()
        env_pc.points = o3d.utility.Vector3dVector(environment_pc)
        env_pc.colors = o3d.utility.Vector3dVector(environment_colors)
        o3d.visualization.draw_geometries([mesh_frame] +  all_data)
        #np.save("scene.npy", np.asarray(scene_pcd.points))

        
    def get_point_cloud(self):
        return self.pts_


if __name__ == "__main__":
    
    color_image = sys.argv[1]
    depth_image = sys.argv[2]
    seg_image = sys.argv[3]

    object_dict = {}
    color_raw = o3d.io.read_image(color_image)
    depth_raw = o3d.io.read_image(depth_image)
    seg_raw = o3d.io.read_image(seg_image)
    rot = get_real_rotation(3.661, 1.929, -1.581)
    trans = get_real_location(-0.605, -0.613, 0.436, rot)
    object_dict = {}
    extractor = pc_extractor_real_cam(color_raw, depth_raw, seg_raw, 
                                      rot,
                                      trans,
                                      object_dict, True)
    visualize_scene(object_dict, False)
    visualize_scene(object_dict, True)
