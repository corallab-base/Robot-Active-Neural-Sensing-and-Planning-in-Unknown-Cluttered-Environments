import os
import sys
import numpy as np

def write_new_file(file_name, chosen):
    prefix = file_name[:file_name.index('.')]
    container = []
    with open(file_name, 'r') as f:
        counter = 0
        for line in f:
            if counter in chosen:
                container.append(line)
            counter += 1
    f.close()
    print (len(container))
    with open(prefix + "_new.txt", 'w') as f:
        for element in container:
            f.write(element)
    f.close()



if __name__ == '__main__':
    chosen_index = set()
    with open("all_centroid_m.txt", 'r') as f:
        counter = 0
        for line in f:
            div = line[:-1].split()
            normal = float(div[-1])
            if normal < 0.05:
                print(counter)
            else:
                chosen_index.add(counter)
            counter += 1
    print (chosen_index)


    write_new_file("all_centroid_m.txt", chosen_index)
    write_new_file("object_collision.txt", chosen_index)
    write_new_file("object_offset.txt", chosen_index)
    write_new_file("object_urdf.txt", chosen_index)


