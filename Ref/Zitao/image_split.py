import cv2
import os

import numpy as np
import copy
import random
import glob
import os
from tools import image_process

H = 128
stride = 32
test_stride = 64

test_full_idx = [0, 4, 18, 55]

# r-right, l-left, t-top half, b-bottom half
bad_data = {22: 'r', 15: 'rt', 17: 'r', 23: 'lb', 50: 'rb', }

# def preprocess_model(arr):
#     new_arr = copy.deepcopy(arr)
#     new_arr[arr > 0] = 0
#     return new_arr


def split_depth():
    data_dir = "../../../my_data/waterfall_npy"
    out_dir = "../../../my_data/depth"

    sss_paths = glob.glob(os.path.join(data_dir, 'sss*.npy'))
    depth_paths = glob.glob(os.path.join(data_dir, 'height*.npy'))

    assert len(sss_paths) == len(depth_paths)

    cnt = [0, 0]  # [train, test]

    for i in range(len(sss_paths)):
        print("waterfall %d" % i)

        sss_path = os.path.join(data_dir, 'sss%d.npy' % i)
        sss_waterfall = np.load(sss_path)
        depth_path = os.path.join(data_dir, 'height%d.npy' % i)
        depth_waterfall = np.load(depth_path)

        # cv2.imwrite(os.path.join(out_dir, 'full_image', 'sss_%d.png' % i), image_process.process_sss(sss_waterfall))
        cv2.imwrite(os.path.join(out_dir, 'full_image', 'depth_%d.png' % i),
                    cv2.applyColorMap(
                        np.uint8(image_process.process_norm(image_process.process_depth(depth_waterfall))),
                        cv2.COLORMAP_JET))
        #
        # print(i)

        [sss_h, sss_w] = np.shape(sss_waterfall)
        [depth_h, depth_w] = np.shape(depth_waterfall)
        #
        assert sss_h == depth_h
        assert sss_w == depth_w == 512
        #
        half_w = sss_w // 2
        sss_waterfall_l = sss_waterfall[:, :half_w]
        sss_waterfall_r = sss_waterfall[:, half_w:]
        depth_waterfall_l = depth_waterfall[:, :half_w]
        depth_waterfall_r = depth_waterfall[:, half_w:]

        data_qual = bad_data.get(i)

        if i in test_full_idx:
            data_folder_path = os.path.join(out_dir, "data")
            image_folder_path = os.path.join(out_dir, "image")
            if not os.path.exists(data_folder_path):
                os.makedirs(data_folder_path)
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)


            cnt_idx = 1

            # split l
            sss_data, depth_data = sss_waterfall_l, depth_waterfall_l
            if data_qual != "l":
                h_idx = 0 if data_qual != "lu" else sss_h // 2
                h_target = sss_h-1 if data_qual != "lb" else sss_h // 2
                cnt_tt = 0
                while h_idx + H < h_target:
                    depth_split = depth_data[h_idx:h_idx + H, :]
                    assert np.shape(depth_split) == (128, 256)
                    # depth_split = cv2.resize(depth_split, dsize=(256, 256))
                    concate_data = depth_split
                    concate_img = cv2.applyColorMap(
                        np.uint8((image_process.process_depth_seperate(depth_split))), cv2.COLORMAP_JET)

                    np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
                    cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
                    cnt[cnt_idx] += 1
                    cnt_tt += 1
                    h_idx += test_stride

                depth_split = depth_data[h_target-H:h_target, :]
                assert np.shape(depth_split) == (128, 256)
                concate_data = depth_split
                concate_img = concate_img = cv2.applyColorMap(
                    np.uint8((image_process.process_depth_seperate(depth_split))), cv2.COLORMAP_JET)

                np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
                cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
                cnt[cnt_idx] += 1


            # split r
            sss_data, depth_data = sss_waterfall_r, depth_waterfall_r
            depth_data = np.flip(depth_data, axis=1)
            if data_qual != "r":
                h_idx = 0 if data_qual != "ru" else sss_h // 2
                h_target = sss_h-1 if data_qual != "rb" else sss_h // 2
                cnt_tt = 0
                while h_idx + H < h_target:
                    depth_split = depth_data[h_idx:h_idx + H, :]
                    assert np.shape(depth_split) == (128, 256)
                    concate_data = depth_split
                    concate_img = cv2.applyColorMap(
                        np.uint8((image_process.process_depth_seperate(depth_split))), cv2.COLORMAP_JET)

                    np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
                    cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
                    cnt[cnt_idx] += 1
                    cnt_tt += 1
                    h_idx += test_stride

                depth_split = depth_data[h_target - H:h_target, :]
                assert np.shape(depth_split) == (128, 256)
                concate_data = depth_split
                concate_img = cv2.applyColorMap(
                    np.uint8((image_process.process_depth_seperate(depth_split))), cv2.COLORMAP_JET)

                np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
                cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
                cnt[cnt_idx] += 1

        else:
            pass

if __name__ == "__main__":
    split_depth()


# if __name__=="__main__":
#
#     data_dir = "../../../my_data/waterfall_npy"
#     # out_dir = "/home/crazybullet/Documents/MasterThesis/my_data"
#     out_dir = "../../../my_data"
#
#     sss_paths = glob.glob(os.path.join(data_dir, 'sss*.npy'))
#     model_paths = glob.glob(os.path.join(data_dir, 'model*.npy'))
#     # depth_paths = glob.glob(os.path.join(data_dir, 'height*.npy'))
#
#     assert len(sss_paths)==len(model_paths)
#
#     cnt = [0, 0]   # [train, test]
#
#     for i in range(len(sss_paths)):
#         print("waterfall %d" % i)
#
#         sss_path = os.path.join(data_dir, 'sss%d.npy' % i)
#         sss_waterfall = np.load(sss_path)
#         model_path = os.path.join(data_dir, 'model%d.npy' % i)
#         model_waterfall = np.load(model_path)
#
#         # cv2.imwrite(os.path.join(out_dir, 'full_image', 'model_%d.png' % i), image_process.process_model(model_waterfall))
#         # cv2.imwrite(os.path.join(out_dir, 'full_image', 'sss_%d.png' % i), image_process.process_sss(sss_waterfall))
#         #
#         # print(i)
#
#         [sss_h, sss_w] = np.shape(sss_waterfall)
#         [model_h, model_w] = np.shape(model_waterfall)
#         #
#         assert sss_h == model_h
#         assert sss_w == model_w == 512
#         #
#         half_w = sss_w // 2
#         sss_waterfall_l = sss_waterfall[:, :half_w]
#         sss_waterfall_r = sss_waterfall[:, half_w:]
#         model_waterfall_l = model_waterfall[:, :half_w]
#         model_waterfall_r = model_waterfall[:, half_w:]
#
#         data_qual = bad_data.get(i)
#
#         if i in test_full_idx:
#             data_folder_path = os.path.join(out_dir, "dataset/test")
#             image_folder_path = os.path.join(out_dir, "image_pair/test")
#             if not os.path.exists(data_folder_path):
#                 os.makedirs(data_folder_path)
#             if not os.path.exists(image_folder_path):
#                 os.makedirs(image_folder_path)
#
#
#             cnt_idx = 1
#
#             # split l
#             sss_data, model_data = sss_waterfall_l, model_waterfall_l
#             if data_qual != "l":
#                 h_idx = 0 if data_qual != "lu" else sss_h // 2
#                 h_target = sss_h-1 if data_qual != "lb" else sss_h // 2
#                 cnt_tt = 0
#                 while h_idx + H < h_target:
#                     sss_split = sss_data[h_idx:h_idx + H, :]
#                     model_split = model_data[h_idx:h_idx + H, :]
#                     # if cnt_tt % 2 == 0:
#                     #     sss_split = np.flip(sss_split, 0)
#                     #     model_split = np.flip(model_split, 0)
#                     assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                     sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                     model_split = cv2.resize(model_split, dsize=(256, 256))
#                     concate_data = np.concatenate((sss_split, model_split), axis=1)
#                     concate_img = np.concatenate(
#                         (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                         axis=1)
#
#                     assert np.shape(concate_data) == (256, 512)
#                     np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                     cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                     cnt[cnt_idx] += 1
#                     cnt_tt += 1
#                     h_idx += test_stride
#
#                 sss_split = sss_data[h_target-H:h_target, :]
#                 model_split = model_data[h_target-H:h_target, :]
#                 # if cnt_tt % 2 == 0:
#                 #     sss_split = np.flip(sss_split, 0)
#                 #     model_split = np.flip(model_split, 0)
#                 assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                 sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                 model_split = cv2.resize(model_split, dsize=(256, 256))
#                 concate_data = np.concatenate((sss_split, model_split), axis=1)
#                 concate_img = np.concatenate(
#                     (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                     axis=1)
#
#                 assert np.shape(concate_data) == (256, 512)
#                 np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                 cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                 cnt[cnt_idx] += 1
#
#
#             # split r
#             sss_data, model_data = sss_waterfall_r, model_waterfall_r
#             sss_data = np.flip(sss_data, axis=1)
#             model_data = np.flip(model_data, axis=1)
#             if data_qual != "r":
#                 h_idx = 0 if data_qual != "ru" else sss_h // 2
#                 h_target = sss_h-1 if data_qual != "rb" else sss_h // 2
#                 cnt_tt = 0
#                 while h_idx + H < h_target:
#                     sss_split = sss_data[h_idx:h_idx + H, :]
#                     model_split = model_data[h_idx:h_idx + H, :]
#                     # if cnt_tt % 2 == 0:
#                     #     sss_split = np.flip(sss_split, 0)
#                     #     model_split = np.flip(model_split, 0)
#                     assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                     sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                     model_split = cv2.resize(model_split, dsize=(256, 256))
#                     concate_data = np.concatenate((sss_split, model_split), axis=1)
#                     concate_img = np.concatenate(
#                         (
#                         image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                         axis=1)
#
#                     assert np.shape(concate_data) == (256, 512)
#                     np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                     cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                     cnt[cnt_idx] += 1
#                     cnt_tt += 1
#                     h_idx += test_stride
#
#                 sss_split = sss_data[h_target - H:h_target, :]
#                 model_split = model_data[h_target - H:h_target, :]
#                 # if cnt_tt % 2 == 0:
#                 #     sss_split = np.flip(sss_split, 0)
#                 #     model_split = np.flip(model_split, 0)
#                 assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                 sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                 model_split = cv2.resize(model_split, dsize=(256, 256))
#                 concate_data = np.concatenate((sss_split, model_split), axis=1)
#                 concate_img = np.concatenate(
#                     (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                     axis=1)
#
#                 assert np.shape(concate_data) == (256, 512)
#                 np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                 cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                 cnt[cnt_idx] += 1
#
#         else:
#             data_folder_path = os.path.join(out_dir, "dataset/train")
#             image_folder_path = os.path.join(out_dir, "image_pair/train")
#             if not os.path.exists(data_folder_path):
#                 os.makedirs(data_folder_path)
#             if not os.path.exists(image_folder_path):
#                 os.makedirs(image_folder_path)
#             cnt_idx = 0
#
#             # split l
#             sss_data, model_data = sss_waterfall_l, model_waterfall_l
#             if data_qual != "l":
#                 h_idx = 0 if data_qual != "lu" else sss_h // 2
#                 h_target = sss_h-1 if data_qual != "lb" else sss_h // 2
#                 cnt_tt = 0
#                 while h_idx + H < h_target:
#                     sss_split = sss_data[h_idx:h_idx + H, :]
#                     model_split = model_data[h_idx:h_idx + H, :]
#                     if cnt_tt % 2 == 0:
#                         sss_split = np.flip(sss_split, 0)
#                         model_split = np.flip(model_split, 0)
#                     assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                     sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                     model_split = cv2.resize(model_split, dsize=(256, 256))
#                     concate_data = np.concatenate((sss_split, model_split), axis=1)
#                     concate_img = np.concatenate(
#                         (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                         axis=1)
#
#                     assert np.shape(concate_data) == (256, 512)
#                     np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                     cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                     cnt[cnt_idx] += 1
#                     cnt_tt += 1
#                     h_idx += stride
#
#                 sss_split = sss_data[h_target - H:h_target, :]
#                 model_split = model_data[h_target - H:h_target, :]
#                 if cnt_tt % 2 == 0:
#                     sss_split = np.flip(sss_split, 0)
#                     model_split = np.flip(model_split, 0)
#                 assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                 sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                 model_split = cv2.resize(model_split, dsize=(256, 256))
#                 concate_data = np.concatenate((sss_split, model_split), axis=1)
#                 concate_img = np.concatenate(
#                     (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                     axis=1)
#
#                 assert np.shape(concate_data) == (256, 512)
#                 np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                 cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                 cnt[cnt_idx] += 1
#
#             # split r
#             sss_data, model_data = sss_waterfall_r, model_waterfall_r
#             sss_data = np.flip(sss_data, axis=1)
#             model_data = np.flip(model_data, axis=1)
#             if data_qual != "r":
#                 h_idx = 0 if data_qual != "ru" else sss_h // 2
#                 h_target = sss_h-1 if data_qual != "rb" else sss_h // 2
#                 cnt_tt = 0
#                 while h_idx + H < h_target:
#                     sss_split = sss_data[h_idx:h_idx + H, :]
#                     model_split = model_data[h_idx:h_idx + H, :]
#                     if cnt_tt % 2 == 0:
#                         sss_split = np.flip(sss_split, 0)
#                         model_split = np.flip(model_split, 0)
#                     assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                     sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                     model_split = cv2.resize(model_split, dsize=(256, 256))
#                     concate_data = np.concatenate((sss_split, model_split), axis=1)
#                     concate_img = np.concatenate(
#                         (
#                             image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                         axis=1)
#
#                     assert np.shape(concate_data) == (256, 512)
#                     np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                     cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                     cnt[cnt_idx] += 1
#                     cnt_tt += 1
#                     h_idx += stride
#
#                 sss_split = sss_data[h_target-H:h_target, :]
#                 model_split = model_data[h_target-H:h_target, :]
#                 if cnt_tt % 2 == 0:
#                     sss_split = np.flip(sss_split, 0)
#                     model_split = np.flip(model_split, 0)
#                 assert np.shape(sss_split) == np.shape(model_split) == (128, 256)
#                 sss_split = cv2.resize(sss_split, dsize=(256, 256))
#                 model_split = cv2.resize(model_split, dsize=(256, 256))
#                 concate_data = np.concatenate((sss_split, model_split), axis=1)
#                 concate_img = np.concatenate(
#                     (image_process.process_sss(sss_split), image_process.process_model(model_split)),
#                     axis=1)
#
#                 assert np.shape(concate_data) == (256, 512)
#                 np.save(os.path.join(data_folder_path, "%d.npy" % cnt[cnt_idx]), concate_data)
#                 cv2.imwrite(os.path.join(image_folder_path, "%d.png" % cnt[cnt_idx]), concate_img)
#                 cnt[cnt_idx] += 1

