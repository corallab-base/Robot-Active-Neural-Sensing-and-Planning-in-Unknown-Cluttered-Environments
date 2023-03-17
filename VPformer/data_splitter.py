import os
import sys
import numpy as np

if __name__ == '__main__':
    data = set()
    with open("good_data.txt", 'r') as f:
        content = f.readlines()
        for t in range(0, len(content), 2):
            f1 = content[t][:-1]
            f2 = content[t+1][:-1]
            data.add((f1, f2))
    f.close()
    data = list(data)
    m = len(data)
    vali_index = set()
    test_index = set()
    size_t = int(m*0.15)
    visited = set()
    while len(vali_index) < size_t:
        candidate = np.random.randint(0, m)
        if candidate not in visited:
            visited.add(candidate)
            vali_index.add(candidate)
    while len(test_index) < size_t:
        candidate = np.random.randint(0, m)
        if candidate not in visited:
            visited.add(candidate)
            test_index.add(candidate)
    with open('vali_data.txt', 'w') as f:
        for element in vali_index:
            f.write(data[element][0] + '\n')
            f.write(data[element][1] + '\n')
    f.close()
    with open('test_data.txt', 'w') as f:
        for element in test_index:
            f.write(data[element][0] + '\n')
            f.write(data[element][1] + '\n')
    f.close()
    with open('train_data.txt', 'w') as f:
        for t in range(m):
            if t not in visited:
                f.write(data[t][0] + '\n')
                f.write(data[t][1] + '\n')
    f.close()

