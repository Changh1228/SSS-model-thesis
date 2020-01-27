import numpy as np
import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import cv2

from tools import image_process

# import random
#
# random.seed(100)


def save_tfrecords(paths, desdir):
    cnt_file_idx = 0
    cnt_data_idx = 0
    if not os.path.exists(desdir):
        os.makedirs(desdir)

    filename = os.path.join(desdir, 'data%d.tfrecords' % cnt_file_idx)
    filename_list = [filename]

    # with tf.python_io.TFRecordWriter(filename) as writer:
    writer = tf.python_io.TFRecordWriter(filename)
    for i, path in enumerate(paths):
        data = np.load(path)
        data_shape = np.shape(data)

        width = data_shape[1]  # [height, width, channels]
        # a_image = np.array(data[:, :width // 2])
        # b_image = np.array(data[:, width // 2:])
        # a_shape = np.shape(a_image)
        # b_shape = np.shape(b_image)

        features = tf.train.Features(
            feature={
                "data": tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1))),
                "data_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=data_shape)),
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

        cnt_data_idx += 1
        if cnt_data_idx == 500:
            writer.close()
            print("file %s saved" % filename)
            cnt_file_idx += 1
            cnt_data_idx = 0
            filename = os.path.join(desdir, 'data%d.tfrecords' % cnt_file_idx)
            filename_list.append(filename)
            writer = tf.python_io.TFRecordWriter(filename)
    writer.close()
    print("file %s saved" % filename)
    print("Total %d files saved" % (cnt_file_idx+1))
    return filename_list


# def load_tfrecords(filename_list, amount):
#     filename_queue = tf.train.string_input_producer(filename_list)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read_up_to(filename_queue)
#     features = tf.parse_example(serialized_example, features={
#         "data": tf.FixedLenFeature([], tf.float32),
#         "shape": tf.FixedLenFeature([], tf.int64)
#     })
#     data = tf.cast(features["data"], tf.float32)
#     shape = tf.cast(features["shape"], tf.int64)
#     data = tf.reshape(data, shape)


# def pares_tf(example_proto):
#     features = {
#         "A": tf.VarLenFeature(dtype=tf.float32),
#         "B": tf.VarLenFeature(dtype=tf.float32),
#         "a_shape": tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
#         "b_shape": tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
#     }
#
#     parsed_example = tf.parse_single_example(serialized=example_proto, features=features)
#
#     parsed_example['A'] = tf.sparse_tensor_to_dense(parsed_example['A'])
#     parsed_example['A'] = tf.reshape(parsed_example['A'], parsed_example['a_shape'])
#     parsed_example['B'] = tf.sparse_tensor_to_dense(parsed_example['B'])
#     parsed_example['B'] = tf.reshape(parsed_example['B'], parsed_example['b_shape'])
#
#     return parsed_example
#
#
# if __name__=="__main__":
#     dir = '/home/crazybullet/Documents/MasterThesis/my_data/dataset/train/tf_data'
#     paths = glob.glob(os.path.join(dir, "*.tfrecords"))
#     rawdata = tf.data.TFRecordDataset(paths)
#     dataset = rawdata.map(pares_tf)
#     # dataset = dataset.shuffle(buffer_size=10000)
#     # dataset = dataset.batch(1).repeat()
#     iterator = dataset.make_initializable_iterator()
#     # iterator.initializer()
#     # iterator = dataset.make_one_shot_iterator()
#
#     sess = tf.InteractiveSession()
#     sess.run(iterator.initializer)
#     next_element = iterator.get_next()
#
#     i = 1
#     while True:
#         try:
#             data_a, data_b = sess.run([next_element['A'], next_element['B']])
#         except tf.errors.OutOfRangeError:
#             print("End of dataset")
#             break
#         else:
#             print('==============example %s ==============' %i)
#             print('data_a shape: %s | type: %s' %(data_a.shape, data_a.dtype))
#             print('data_b shape: %s | type: %s' %(data_b.shape, data_b.dtype))
#             plt.imshow(data_a)
#             plt.show()
#         i += 1

