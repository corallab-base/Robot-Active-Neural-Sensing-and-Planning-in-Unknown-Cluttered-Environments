import os
import sys
import numpy as np

names = set()
for root, dirs, files in os.walk('./scene_generation_0602/'):
    for f in files:
        if 'scene' in f:
            prefix = f[:f.index('scene') - 1]
            names.add(prefix)
        elif 'camera' in files:
            prefix = f[:f.index('camera') - 1]
            names.add(prefix)

total_size = len(names)
chosen_size = int(total_size*0.15)

test_ind = np.random.choice([x for x in range(len(names))], size = chosen_size, replace = False)
test_sets = set()
for element in test_ind:
    test_sets.add(element)
validation_sets = set()
while len(validation_sets) < chosen_size:
    candidate = np.random.randint(0, len(names))
    if candidate not in test_sets and candidate not in validation_sets:
        validation_sets.add(candidate)
names = list(names)

with open('test_list.txt', 'w') as f:
    for ids in test_ind:
        print (ids)
        scene_file = names[ids] + '_scene.npy\n'
        camera_file = names[ids] + '_camera.npy\n'
        f.write(scene_file)
        f.write(camera_file)
f.close()

with open('validation_list.txt', 'w') as f:
    for ids in validation_sets:
        print (ids)
        scene_file = names[ids] + '_scene.npy\n'
        camera_file = names[ids] + '_camera.npy\n'
        f.write(scene_file)
        f.write(camera_file)
f.close()
