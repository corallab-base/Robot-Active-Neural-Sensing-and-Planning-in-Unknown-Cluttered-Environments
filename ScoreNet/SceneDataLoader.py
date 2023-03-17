import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SceneDataset(Dataset):
    def __init__(self, path = None):
        if not path:
            sys.exit(1)
        else:
            self.path_ = path
            self.names_ = set()
            print (path)
            for root, dirs, files in os.walk(path):
                for f in files:
                    if 'scene' in f:
                        prefix = f[:f.index('scene')-1]
                        self.names_.add(prefix)
                    elif 'camera' in files:
                        prefix = f[:f.index('camera')-1]
                        self.names_.add(prefix)
            self.names_ = list(self.names_)
            self.min_pose_ = sys.maxsize
            
    def __getitem__(self, idx):
        prefix = self.path_ + self.names_[idx]
        prefix_scene = prefix + '_scene.npy'
        prefix_camera = prefix + '_camera.npy'
        scene_graph = np.load(prefix_scene)
        scene_graph = np.where(scene_graph == -1, 255, scene_graph)
        #print (scene_graph[1][2][3][0])
        scene_graph = np.transpose(scene_graph, (3, 0, 1, 2))
        #print (scene_graph[0][1][2][3])
        camera_info = np.load(prefix_camera)
        camera_pose = camera_info[:7]
        score = np.array([camera_info[7]])
        self.min_pose_ = min(self.min_pose_, abs(camera_pose[3]))
        return scene_graph, camera_pose, score


    def __len__(self):
        return len(self.names_)

if __name__ == '__main__':
    train_dataset = SceneDataset('./scene_train/')
    dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

    print (train_dataset.min_pose_)
    data_iter = iter(dataloader)
    for i, (scene, camera_pose, score) in enumerate(dataloader):
        pass
    print (train_dataset.min_pose_)

