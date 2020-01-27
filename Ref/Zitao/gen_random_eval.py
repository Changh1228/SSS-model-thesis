import numpy as np
import random

random.seed(100)

mylist = []

for i in range(240):
    # 0-ground truth  1-unet  2-resnet
    mylist.append(random.randint(0, 2))

get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

print("ground truth: ", get_indexes(0, mylist))
print("U-net       : ", get_indexes(1, mylist))
print("ResNet      : ", get_indexes(2, mylist))

list_idx = list(range(240))
random.shuffle(list_idx)
# print(list_idx)

mylist_array = np.array(mylist)
# print(mylist_array[list_idx])

name_space = ['target', 'unet', 'resnet']

for i in range(240):
    print(i, "---", name_space[mylist_array[list_idx][i]], list_idx[i])