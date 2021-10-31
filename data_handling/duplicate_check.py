import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from process_data import load_plankton, load_plankton_direct
from sklearn.utils import shuffle

x, y = load_plankton_direct()

count = 0
count_dupe = 0

threshold = 0.0000001

dupe_index = []
for i in range(len(x)):
    print('current i: ' + str(i) + ' dupe count: ' + str(count_dupe))
    for j in range(i+1, len(x)):
        mse = ((x[i] - x[j])**2).mean(axis=None)
        if mse < threshold:
            dupe_index.append(i)
            count_dupe += 1
            break
dupes = np.take(x, dupe_index, axis=0)
new_x = np.delete(x, dupe_index, axis=0)
new_y = np.delete(y, dupe_index, axis=0)

np.save('plankton_x_fix', new_x, )
np.save('plankton_y_fix', new_y)
np.save('duplicates_plankton_x_fix', dupes)

def check_for_duplicates():
    fix_val = np.load('E:/Uni/Masterarbeit/Semi-supervised/fixmatch_val.npy')
    fix_val_y = np.load('E:/Uni/Masterarbeit/Semi-supervised/fixmatch_val_y.npy')
    duplicates = np.load('E:/Uni/Masterarbeit/Semi-supervised/duplicates_plankton_x_fix.npy')
    test1 = np.load('E:/Uni/Masterarbeit/Semi-supervised/no_duplicates_plankton_x_fix.npy')
    test2 = np.load('E:/Uni/Masterarbeit/Semi-supervised/no_duplicates_plankton_y_fix.npy')
    dupe_index = []
    fix_val = np.reshape(fix_val, (-1, 64, 64))
    print(str(len(fix_val)))
    for i in range(len(fix_val)):
        print(str(i))
        for j in range(len(duplicates)):
            mse = ((fix_val[i] - duplicates[j])**2).mean(axis=None)
            if mse < 0.0000001:
                dupe_index.append(i)
                break
    images = np.delete(fix_val, dupe_index, axis=0)
    labels = np.delete(fix_val_y, dupe_index, axis=0)
    print('duplicates removed: ' + str(len(dupe_index)))

    class_count = np.zeros(10, dtype=np.int32)
    images_bal = []
    labels_bal = []
    for elem in labels:
        index = np.argmax(elem)
        class_count[index] += 1
    print(str(class_count))
    class_count = np.repeat(np.amin(class_count), 10)
    for i in range(len(labels)):
        c_i = np.argmax(labels[i])
        if class_count[c_i] > 0:
            class_count[c_i] -= 1
            images_bal.append(images[i])
            labels_bal.append(labels[i])
    images = np.array(images_bal)
    labels = np.array(labels_bal)

    images = np.reshape(images, (-1, 64, 64, 1))

    np.save('no_duplicates_plankton_x_fix', images)
    np.save('no_duplicates_plankton_y_fix', labels)
