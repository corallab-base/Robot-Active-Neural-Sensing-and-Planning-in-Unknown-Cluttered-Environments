import os
import sys
import open3d as o3d
import numpy as np


if __name__ == '__main__':
    file1_points = sys.argv[1]
    file1_colors = sys.argv[2]

    file2_points = sys.argv[3]
    file2_colors = sys.argv[4]

    data1_points = np.load(file1_points)
    data1_colors = np.load(file1_colors)

    data2_points = np.load(file2_points)
    data2_colors = np.load(file2_colors)

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(data1_points)
    pc1.colors = o3d.utility.Vector3dVector(data1_colors)


    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(data2_points)
    pc2.colors = o3d.utility.Vector3dVector(data2_colors)

    #o3d.visualization.draw_geometries([pc1, pc2])

    threshold = 0.001
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], \
                            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    evaluation = o3d.registration.evaluate_registration(pc1, pc2, threshold, trans_init)

    print (evaluation)


    reg_p2p = o3d.registration.registration_icp(
                pc1, pc2, threshold, trans_init,
                    o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

