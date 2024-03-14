import numpy as np
import sys
import open3d as o3d


if __name__ == "__main__":
    target_pc_num = 8000
    offsets = []
    with open("object_offset.txt") as f:
        for line in f:
            temp_offsets = [float(x) for x in line.split()]
            offsets.append(np.array(temp_offsets))
    index = 0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin = [0, 0, 0])
    names = []
    with open("all_names.txt") as f:
        for line in f:
            object_name = None
            if '-' in line:
                object_name = line[line.index('-')+1:-2]
            else:
                object_name = line[line.index('_')+1:-2]
            object_name = ''.join([x for x in object_name if (x).isalnum()])
            print (object_name)
            file_name = line[:-1] + "textured.obj"
            vertices_arr = []
            temp_offset = offsets[index]
            for line2 in open(file_name):
                if line2[:2] == 'v ':
                    div = line2[2:-1].split()
                    vertices_arr.append([float(x) for x in div])
                    vertices_arr[-1][0] += temp_offset[0]
                    vertices_arr[-1][1] += temp_offset[1]
                    vertices_arr[-1][2] += temp_offset[2]
            total_pc_num = len(vertices_arr)
            select_vertices_arr = []
            select = np.random.choice(total_pc_num, target_pc_num, replace = False)
            for element in select:
                select_vertices_arr.append(vertices_arr[element])


            print (index, object_name)
            for t in range(6):
                factor = 1 + t/10.0
                scale_vertices_arr = []
                for element in select_vertices_arr:
                    scale_vertices_arr.append(element)
                    scale_vertices_arr[-1][0] *= factor
                    scale_vertices_arr[-1][1] *= factor
                    scale_vertices_arr[-1][2] *= factor
                    vertices = np.array(scale_vertices_arr)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(vertices)
                    #print (type(vertices))
                    #sys.exit(1)
                file_name = str(index) + "-" + object_name + "s" + str(t) + ".npy"
                np.save("./pcs/" + file_name, vertices)
                names.append(file_name)
                #o3d.visualization.draw_geometries([pcd, mesh_frame])
            index += 1
    with open("train.txt", 'a') as f:
        for na in names:
            f.write(na + '\n')


