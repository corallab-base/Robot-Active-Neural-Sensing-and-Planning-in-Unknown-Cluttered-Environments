import open3d as o3d
import numpy as np
import math
import sys

unit_length = 0.05
unit_count = 20
focus = 910
image_size_y = 641
image_size_z = 352

def generate_random_object():
    center_location = np.array([10, 10, 10]) * unit_length
    res = []
    for i in range(unit_count):
        for j in range(unit_count):
            for k in range(unit_count):
                temp_location = np.array([i,j,k]) * unit_length
                distance = np.linalg.norm(temp_location - center_location)
                if distance <= 0.2:
                    res.append((k + j * unit_count + i * (unit_count ** 2), temp_location))
    return res

def create_environment(length, width, height):
    env_list = np.zeros(shape = (0,3))
    for i in range(length):
        for j in range(width):
            for k in range(height):
                env_list = np.concatenate((env_list, np.array([[i * unit_length,
                                                                j * unit_length,
                                                                k * unit_length]])))
    pcd_env = o3d.geometry.PointCloud()
    pcd_env.points = o3d.utility.Vector3dVector(env_list)
    pcd_env.paint_uniform_color([0.9, 0.9, 0.9])
    return pcd_env

def generate_random_rotation():
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    rotation = np.random.rand()*np.pi*2
    return np.array([x * np.sin(rotation//2), 
                     y * np.sin(rotation//2),
                     z * np.sin(rotation//2), 
                     np.cos(rotation//2)])

#this camera class will be used to setup a vitual camera and 
#check if a voxel can be seen in the picture frame
class camera_real:

    def __init__(self, location, rotation):
        self.location_ = np.array(location)
        self.rotation_ = np.array(rotation)
        print (self.location_, self.rotation_)
        self.inverse_rotation_ = np.array(self.rotation_)
        for i in range(3):
            self.inverse_rotation_[i] *= -1.0


    def hamilton_product(self, x1, x2):
        #print (x1, x2)
        a1, b1, c1, d1 = x1
        a2, b2, c2, d2 = x2
        return [d1*a2 + a1*d2 + b1*c2 -c1*b2,
                d1*b2 - a1*c2 + b1*d2 + c1*a2,
                d1*c2 + a1*b2 - b1*a2 + c1*d2,
                d1*d2 - a1*a2 - b1*b2 - c1*c2]

    def build(self):
        #calculate normal vector and image vector
        self.normal_vector_ = np.array(self.hamilton_product(self.hamilton_product(self.rotation_, [1, 0, 0, 0]), self.inverse_rotation_)[:3])
        self.pic_y_vector_ = np.array(self.hamilton_product(self.hamilton_product(self.rotation_, [0, 1, 0, 0]), self.inverse_rotation_)[:3])
        self.pic_z_vector_ = np.array(self.hamilton_product(self.hamilton_product(self.rotation_, [0, 0, 1, 0]), self.inverse_rotation_)[:3])
        print (np.linalg.norm(self.normal_vector_))
        self.normal_vector_ /= np.linalg.norm(self.normal_vector_)
        self.pic_y_vector_ /= np.linalg.norm(self.pic_y_vector_)
        self.pic_z_vector_ /= np.linalg.norm(self.pic_z_vector_)
        self.image_center_ = self.location_ - focus * 1.0 * self.normal_vector_
        self.object_list_ = {}

    def distance_calc(self, query_location):
        vec1 = self.image_center_ - query_location
        disy = np.linalg.norm(np.cross(vec1, self.pic_y_vector_))/np.linalg.norm(self.pic_y_vector_)
        disz = np.linalg.norm(np.cross(vec1, self.pic_z_vector_))/np.linalg.norm(self.pic_z_vector_)
        return disy, disz

    def register_object(self, object_list):
        for query_point in object_list:
          unit_vector = self.location_ - query_point
          if unit_vector.any() and abs(np.dot(unit_vector, self.normal_vector_)) >= 1e-6:
            unit_vector /= np.linalg.norm(unit_vector)
            k = (np.dot(self.normal_vector_, self.image_center_) - \
                 np.dot(self.normal_vector_, self.location_)) \
                *1.0/(np.dot(unit_vector, self.normal_vector_))
            if k >= 0:
                res_location = self.location_ + k * unit_vector
                proj_y = np.dot(res_location - self.image_center_, self.pic_y_vector_)
                proj_z = np.dot(res_location - self.image_center_, self.pic_z_vector_)
                if abs(proj_y) <= image_size_y and abs(proj_z) <= image_size_z:
                    tups = (round(proj_y), round(proj_z))
                    if tups not in self.object_list_:
                        self.object_list_[tups] = k
                    else:
                        self.object_list_[tups] = min(self.object_list_[tups], k)
                else:
                    pass
            else: pass
          else:
              pass

    def inside_frame(self, query_point):
        #return boolean meaning whether is voxel can be seen from camera
        unit_vector = self.location_ - query_point
        if unit_vector.any() and abs(np.dot(unit_vector, self.normal_vector_)) >= 1e-6:
          unit_vector /= np.linalg.norm(unit_vector)
          k = (np.dot(self.normal_vector_, self.image_center_) - \
               np.dot(self.normal_vector_, self.location_)) \
              *1.0/(np.dot(unit_vector, self.normal_vector_))
          if k >= 0:
              res_location = self.location_ + k * unit_vector
              proj_y = np.dot(res_location - self.image_center_, self.pic_y_vector_)
              proj_z = np.dot(res_location - self.image_center_, self.pic_z_vector_)
              if 0 <= proj_y + image_size_y < 1280 and 0 <= proj_z + image_size_z < 720:
                  tups = (round(proj_y), round(proj_z))
                  return (True, np.dot(query_point - self.location_, self.normal_vector_), tups)
              else:
                  return (False, None, None)
          else: return (False, None, None)
        else:
            return (False, None, None)

    #return line set visual comp for visual rendering
    def visualization(self):
        points = [self.location_, self.location_ + 0.5*self.normal_vector_,
                                  self.location_ - 0.2*self.normal_vector_]
        lines = [[0, 1], [0, 2]]
        colors = [[1, 0, 0], [0, 0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def print_info(self):
        print (self.normal_vector_, self.pic_y_vector_, self.pic_z_vector_, self.image_center_)


if __name__ == "__main__":
    pcd_env = create_environment(unit_count, unit_count, unit_count)
    object_list = generate_random_object()
    object_loc_track = set()
    for idx, loc in object_list:
        pcd_env.colors[idx] = np.array([1, 1, 0])
        object_loc_track.add(tuple(loc))
   
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1, origin = [-1, -1, 0])

    #line_set = set_camera(20, 20, 20)
    camera_location = None
    while True:
        candidate_location = np.random.randint(0, unit_count, size = 3)*unit_length
        if tuple(candidate_location) not in object_loc_track:
            camera_location = candidate_location
            break
    #cam = camera(camera_location, generate_random_rotation())
    cam = camera(np.array([0, 10, 10])*0.05, np.array([0.0, 0.0, 0.0, 1.0]))
    cam.build()
    cam.register_object(object_loc_track)
    cam.print_info()
    for i in range(unit_count):
        for j in range(unit_count):
            for k in range(unit_count):
                index = k + j*unit_count + i*(unit_count**2)
                if cam.inside_frame(np.array([i,j,k])*unit_length):
                    pcd_env.colors[index] = np.array([1, 0, 0])

    line_set = cam.visualization()

    o3d.visualization.draw_geometries([pcd_env, line_set, mesh_frame])
