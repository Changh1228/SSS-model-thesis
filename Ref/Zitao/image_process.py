import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import os


def process_sss(img):
    new_img = copy.deepcopy(img)
    new_img *= 100
    new_img[new_img < 0] = 0
    new_img[new_img >= 255] = 255
    return new_img


def process_norm(img):
    new_img = np.float64(copy.deepcopy(img))
    max_val = np.max(new_img)
    new_img *= 255.0/max_val
    return np.uint8(new_img)


def process_depth(img):
    new_img = copy.deepcopy(img)
    new_img = -(new_img + 10) * 15
    new_img[new_img < 0] = 0
    new_img[new_img >= 255] = 255
    return new_img


def process_depth_seperate(img):
    new_img = copy.deepcopy(img)
    new_img = -(new_img + 10)
    new_img[new_img < 0] = 0
    new_img[new_img >= 255] = 255
    new_img = process_norm(new_img)
    return np.uint8(new_img)


def process_model(img):
    return img * 255


def histograms_equalization(img):
    # img = np.expand_dims(img, axis=-1)
    # print(np.shape(img))
    return cv2.equalizeHist(np.uint8(img))


def color_mapping(img):
    pass


def process_sss_show(img):
    new_img = copy.deepcopy(img)
    new_img = new_img*100
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img


def process_depth_show(img):
    new_img = copy.deepcopy(img)
    new_img = -(new_img + 10) * 10
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

# def norm_sss_pair(img1, img2)

_PAIR_IN_IMAGE = 8


def save_image_pair(dir, inputs, outputs, targets, direction, name_flag=None):        #
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = 'out' if name_flag is None else name_flag

    if direction == "B2A":            # input -- depth
        depth_images = process_depth(inputs)
        sss_output_images = process_sss(outputs)
        sss_target_images = process_sss(targets)
        assert np.shape(depth_images) == np.shape(sss_output_images) == np.shape(sss_target_images)
        amount = np.shape(depth_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count ==_PAIR_IN_IMAGE:
                path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
                cv2.imwrite(path, pair_images)
                count = 0
                image_num += 1
            depth_image = depth_images[i]
            sss_output_image = sss_output_images[i]
            sss_target_image = sss_target_images[i]
            pair_image = np.concatenate((depth_image, sss_output_image, sss_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
        cv2.imwrite(path, pair_images)

    elif direction == "C2A":
        model_images = process_model(inputs)
        sss_output_images = process_sss(outputs)
        sss_target_images = process_sss(targets)
        assert np.shape(model_images) == np.shape(sss_output_images) == np.shape(sss_target_images)
        amount = np.shape(model_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count == _PAIR_IN_IMAGE:
                path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
                cv2.imwrite(path, pair_images)
                count = 0
                image_num += 1
            model_image = cv2.resize(model_images[i], dsize=(256, 128))
            sss_output_image = cv2.resize(sss_output_images[i], dsize=(256, 128))
            sss_target_image = cv2.resize(sss_target_images[i], dsize=(256, 128))

            sss_output__h_e = histograms_equalization(sss_output_image)
            sss_target__h_e = histograms_equalization(sss_target_image)

            pair_image = np.concatenate((model_image,
                                         sss_output_image, sss_target_image,
                                         sss_output__h_e, sss_target__h_e), axis=1)
            if count == 0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
        #             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
        #             plt.savefig(path)
        #             plt.clf()
        path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
        cv2.imwrite(path, pair_images)

    elif direction == "A2B":            # input -- sss
        sss_images = process_sss(inputs)
        depth_output_images = process_depth(outputs)
        depth_target_images = process_depth(targets)
        assert np.shape(sss_images) == np.shape(depth_output_images) == np.shape(depth_target_images)
        amount = np.shape(sss_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count ==_PAIR_IN_IMAGE:
                path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
                cv2.imwrite(path, pair_images)
                count = 0
                image_num += 1
            sss_image = sss_images[i]
            depth_output_image = depth_output_images[i]
            depth_target_image = depth_target_images[i]
            pair_image = np.concatenate((sss_image, depth_output_image, depth_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
        cv2.imwrite(path, pair_images)


def save_image_eval(dir, inputs, outputs, targets, direction, name_flag=None):        #
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = 'out_test' if name_flag is None else name_flag

    if direction == "B2A":            # input -- depth
#         depth_images = process_depth(inputs)
#         sss_output_images = process_sss(outputs)
#         sss_target_images = process_sss(targets)
#         assert np.shape(depth_images) == np.shape(sss_output_images) == np.shape(sss_target_images)
#         amount = np.shape(depth_images)[0]
#         count = 0
#         image_num = 0
#         for i in range(amount):
#             if count ==_PAIR_IN_IMAGE:
#                 path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
#                 cv2.imwrite(path, pair_images)
#                 count = 0
#                 image_num += 1
#             depth_image = depth_images[i]
#             sss_output_image = sss_output_images[i]
#             sss_target_image = sss_target_images[i]
#             pair_image = np.concatenate((depth_image, sss_output_image, sss_target_image), axis=1)
#             if count==0:
#                 pair_images = copy.deepcopy(pair_image)
#             else:
#                 pair_images = np.concatenate((pair_images, pair_image), axis=0)
#             count += 1
# #             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
# #             plt.savefig(path)
# #             plt.clf()
#         path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
#         cv2.imwrite(path, pair_images)
        print("Not finished code")

    elif direction == "C2A":
        cnt_odd = False

        model_images = process_model(inputs)
        sss_output_images = process_sss(outputs)
        sss_target_images = process_sss(targets)
        assert np.shape(model_images) == np.shape(sss_output_images) == np.shape(sss_target_images)
        amount = np.shape(model_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            model_image = cv2.resize(model_images[i], dsize=(256, 128))
            sss_output_image = cv2.resize(process_norm(sss_output_images[i]), dsize=(256, 128))
            sss_target_image = cv2.resize(process_norm(sss_target_images[i]), dsize=(256, 128))

            pair_image_output = np.concatenate((model_image, sss_output_image), axis=0)
            pair_image_target = np.concatenate((model_image, sss_target_image), axis=0)

            path_output = os.path.join(dir, '%s_%d_output.png' % (file_name, image_num))
            path_target = os.path.join(dir, '%s_%d_target.png' % (file_name, image_num))
            cv2.imwrite(path_output, pair_image_output)
            cv2.imwrite(path_target, pair_image_target)
            image_num += 1

            # if cnt_odd:
            #     model_image = cv2.resize(model_images[i], dsize=(256, 128))
            #     sss_output_image = cv2.resize(sss_output_images[i], dsize=(256, 128))
            #     sss_target_image = cv2.resize(sss_target_images[i], dsize=(256, 128))
            #
            #     pair_image_output = np.concatenate((model_image, sss_output_image), axis=1)
            #     pair_image_target = np.concatenate((model_image, sss_target_image), axis=1)
            #
            #     pair_images_output = process_norm(pair_image_output)
            #     pair_images_target = process_norm(pair_image_target)
            #     path_output = os.path.join(dir, '%s_%d_output.png' % (file_name, image_num))
            #     path_target = os.path.join(dir, '%s_%d_target.png' % (file_name, image_num))
            #     cv2.imwrite(path_output, pair_images_output)
            #     cv2.imwrite(path_target, pair_images_target)
            #     image_num += 1
            #
            #
            #     if count == 0:
            #         pair_images_output = copy.deepcopy(pair_image_output)
            #         pair_images_target = copy.deepcopy(pair_image_target)
            #         count += 1
            #     else:
            #         pair_images_output = process_norm(np.concatenate((pair_images_output, pair_image_output), axis=0))
            #         pair_images_target = process_norm(np.concatenate((pair_images_target, pair_image_target), axis=0))
            #
            #         count = 0
            #         path_output = os.path.join(dir, '%s_%d_output.png' % (file_name, image_num))
            #         path_target = os.path.join(dir, '%s_%d_target.png' % (file_name, image_num))
            #         cv2.imwrite(path_output, pair_images_output)
            #         cv2.imwrite(path_target, pair_images_target)
            #         image_num += 1
            # else:
            #     pass
            # cnt_odd = not cnt_odd



        #     if count == _PAIR_IN_IMAGE:
        #         path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
        #         cv2.imwrite(path, pair_images)
        #         count = 0
        #         image_num += 1
        #     model_image = cv2.resize(model_images[i], dsize=(256, 128))
        #     sss_output_image = cv2.resize(sss_output_images[i], dsize=(256, 128))
        #     sss_target_image = cv2.resize(sss_target_images[i], dsize=(256, 128))
        #
        #     sss_output__h_e = histograms_equalization(sss_output_image)
        #     sss_target__h_e = histograms_equalization(sss_target_image)
        #
        #     pair_image = np.concatenate((model_image,
        #                                  sss_output_image, sss_target_image,
        #                                  sss_output__h_e, sss_target__h_e), axis=1)
        #     if count == 0:
        #         pair_images = copy.deepcopy(pair_image)
        #     else:
        #         pair_images = np.concatenate((pair_images, pair_image), axis=0)
        #     count += 1
        # #             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
        # #             plt.savefig(path)
        # #             plt.clf()
        # path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
        # cv2.imwrite(path, pair_images)

    elif direction == "A2B":            # input -- sss
#         sss_images = process_sss(inputs)
#         depth_output_images = process_depth(outputs)
#         depth_target_images = process_depth(targets)
#         assert np.shape(sss_images) == np.shape(depth_output_images) == np.shape(depth_target_images)
#         amount = np.shape(sss_images)[0]
#         count = 0
#         image_num = 0
#         for i in range(amount):
#             if count ==_PAIR_IN_IMAGE:
#                 path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
#                 cv2.imwrite(path, pair_images)
#                 count = 0
#                 image_num += 1
#             sss_image = sss_images[i]
#             depth_output_image = depth_output_images[i]
#             depth_target_image = depth_target_images[i]
#             pair_image = np.concatenate((sss_image, depth_output_image, depth_target_image), axis=1)
#             if count==0:
#                 pair_images = copy.deepcopy(pair_image)
#             else:
#                 pair_images = np.concatenate((pair_images, pair_image), axis=0)
#             count += 1
# #             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
# #             plt.savefig(path)
# #             plt.clf()
#         path = os.path.join(dir, '%s_%d.png' % (file_name, image_num))
#         cv2.imwrite(path, pair_images)
        print("Not finished code")


def save_image_example(dir, inputs, targets, direction):        #
    if not os.path.exists(dir):
        os.makedirs(dir)

    if direction=="B2A":            # input -- depth
        depth_images = process_depth(inputs)
        sss_target_images = process_sss(targets)
        assert np.shape(depth_images) == np.shape(sss_target_images)
        amount = np.shape(depth_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                path = os.path.join(dir, 'out_%d.png' % image_num)
                cv2.imwrite(path, pair_images)
                count = 0
                image_num += 1
            depth_image = depth_images[i]
            sss_target_image = sss_target_images[i]
            pair_image = np.concatenate((sss_target_image, depth_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        path = os.path.join(dir, 'out_%d.png' % image_num)
        cv2.imwrite(path, pair_images)

    elif direction=="A2B":            # input -- sss
        sss_images = process_sss(inputs)
        depth_target_images = process_depth(targets)
        assert np.shape(sss_images) == np.shape(depth_target_images)
        amount = np.shape(sss_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                path = os.path.join(dir, 'out_%d.png' % image_num)
                cv2.imwrite(path, pair_images)
                count = 0
                image_num += 1
            sss_image = sss_images[i]
            depth_target_image = depth_target_images[i]
            pair_image = np.concatenate((sss_image, depth_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        path = os.path.join(dir, 'out_%d.png' % image_num)
        cv2.imwrite(path, pair_images)


def show_image_pair(inputs, outputs, targets, direction):        #
    if direction=="B2A":            # input -- depth
        depth_images = process_depth_show(inputs)
        sss_output_images = process_sss_show(outputs)
        sss_target_images = process_sss_show(targets)
        assert np.shape(depth_images) == np.shape(sss_output_images) == np.shape(sss_target_images)
        amount = np.shape(depth_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                cv2.imshow('pair_images', pair_images)
                cv2.waitKey(0)
                count = 0
                image_num += 1
            depth_image = depth_images[i]
            sss_output_image = sss_output_images[i]
            sss_target_image = sss_target_images[i]
            pair_image = np.concatenate((depth_image, sss_output_image, sss_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        cv2.imshow('pair_images', pair_images)
        cv2.waitKey(0)

    elif direction=="A2B":            # input -- sss
        sss_images = process_sss_show(inputs)
        depth_output_images = process_depth_show(outputs)
        depth_target_images = process_depth_show(targets)
        assert np.shape(sss_images) == np.shape(depth_output_images) == np.shape(depth_target_images)
        amount = np.shape(sss_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                cv2.imshow('pair_images', pair_images)
                cv2.waitKey(0)
                count = 0
                image_num += 1
            sss_image = sss_images[i]
            depth_output_image = depth_output_images[i]
            depth_target_image = depth_target_images[i]
            pair_image = np.concatenate((sss_image, depth_output_image, depth_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        cv2.imshow('pair_images', pair_images)
        cv2.waitKey(0)


def show_image_example(inputs, targets, direction):        #
    if direction=="B2A":            # input -- depth
        depth_images = process_depth_show(inputs)
        sss_target_images = process_sss_show(targets)
        assert np.shape(depth_images)  == np.shape(sss_target_images)
        amount = np.shape(depth_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                cv2.imshow('pair_images', pair_images)
                cv2.waitKey(0)
                count = 0
                image_num += 1
            depth_image = depth_images[i]
            sss_target_image = sss_target_images[i]
            pair_image = np.concatenate((depth_image, sss_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        cv2.imshow('pair_images', pair_images)
        cv2.waitKey(0)

    elif direction=="A2B":            # input -- sss
        sss_images = process_sss_show(inputs)
        depth_target_images = process_depth_show(targets)
        assert np.shape(sss_images) == np.shape(depth_target_images)
        amount = np.shape(sss_images)[0]
        count = 0
        image_num = 0
        for i in range(amount):
            if count==_PAIR_IN_IMAGE:
                cv2.imshow('pair_images', pair_images)
                cv2.waitKey(0)
                count = 0
                image_num += 1
            sss_image = sss_images[i]
            depth_target_image = depth_target_images[i]
            pair_image = np.concatenate((sss_image, depth_target_image), axis=1)
            if count==0:
                pair_images = copy.deepcopy(pair_image)
            else:
                pair_images = np.concatenate((pair_images, pair_image), axis=0)
            count += 1
#             plt.imshow(pair_image[:,:,0], vmin=0, vmax=255)
#             plt.savefig(path)
#             plt.clf()
        cv2.imshow('pair_images', pair_images)
        cv2.waitKey(0)


def replace_image_eval():           #  this function is only for one-time use. never call this outside this file
    dir_path = "E:/MyDocuments/Documents/master thesis/A5 Report Draft/figs/eval"
    depth_path = "E:/MyDocuments/Documents/Python/PycharmProjects/MasterThesis/my_data/depth/image"

    # eval_path = os.path.join(dir_path, 'target')
    # for i in range(240):
    #     depth_file = os.path.join(depth_path, "%d.png" % i)
    #     if os.path.exists(depth_file):
    #         print("%d.png exist     " % i, end='')
    #         eval_file = os.path.join(eval_path, "out_test_%d_target.png" % i)
    #         if os.path.exists(eval_file):
    #             print("out_test_%d_target.png   exist" % i)
    #             depth_img = cv2.imread(depth_file)
    #             eval_img = cv2.imread(eval_file)
    #             assert np.shape(depth_img) == (128, 256, 3)
    #             assert np.shape(eval_img) == (256, 256, 3)
    #             sss_img = eval_img[128:, :, :]
    #             # sss_img = cv2.applyColorMap(sss_img, cv2.COLORMAP_JET)
    #
    #             new_img = np.concatenate((depth_img, sss_img), axis=0)
    #             # cv2.imshow('img', new_pair_img)
    #             # cv2.waitKey(0)
    #             new_file_name = os.path.join(eval_path, "img_%d.png" % i)
    #             cv2.imwrite(new_file_name, new_img)
    #
    #         else:
    #             print("")
    #
    # eval_path = os.path.join(dir_path, 'resnet')
    # for i in range(240):
    #     depth_file = os.path.join(depth_path, "%d.png" % i)
    #     if os.path.exists(depth_file):
    #         print("%d.png exist     " % i, end='')
    #         eval_file = os.path.join(eval_path, "out_test_%d_output.png" % i)
    #         if os.path.exists(eval_file):
    #             print("out_test_%d_output.png   exist" % i)
    #             depth_img = cv2.imread(depth_file)
    #             eval_img = cv2.imread(eval_file)
    #             assert np.shape(depth_img) == (128, 256, 3)
    #             assert np.shape(eval_img) == (256, 256, 3)
    #             sss_img = eval_img[128:, :, :]
    #             # sss_img = cv2.applyColorMap(sss_img, cv2.COLORMAP_JET)
    #
    #             new_img = np.concatenate((depth_img, sss_img), axis=0)
    #             # cv2.imshow('img', new_pair_img)
    #             # cv2.waitKey(0)
    #             new_file_name = os.path.join(eval_path, "img_%d.png" % i)
    #             cv2.imwrite(new_file_name, new_img)
    #
    #         else:
    #             print("")
    #
    # eval_path = os.path.join(dir_path, 'unet')
    # for i in range(240):
    #     depth_file = os.path.join(depth_path, "%d.png" % i)
    #     if os.path.exists(depth_file):
    #         print("%d.png exist     " % i, end='')
    #         eval_file = os.path.join(eval_path, "out_test_%d_output.png" % i)
    #         if os.path.exists(eval_file):
    #             print("out_test_%d_output.png   exist" % i)
    #             depth_img = cv2.imread(depth_file)
    #             eval_img = cv2.imread(eval_file)
    #             assert np.shape(depth_img) == (128, 256, 3)
    #             assert np.shape(eval_img) == (256, 256, 3)
    #             sss_img = eval_img[128:, :, :]
    #             # sss_img = cv2.applyColorMap(sss_img, cv2.COLORMAP_JET)
    #
    #             new_img = np.concatenate((depth_img, sss_img), axis=0)
    #             # cv2.imshow('img', new_pair_img)
    #             # cv2.waitKey(0)
    #             new_file_name = os.path.join(eval_path, "img_%d.png" % i)
    #             cv2.imwrite(new_file_name, new_img)
    #
    #         else:
    #             print("")

    # eval_file = "E:/MyDocuments/Documents/Python/PycharmProjects/MasterThesis/server_output/report/resnet/" \
    #             "resnet7_v4-32-32-L1_10-L2_0-GAN_1/eval_images/out_test_46_output.png"

    idxs = [161, 9, 68, 212, 94, 71, 81, 63, 44, 66, 46, 40, 125, 171, 134, 35, 39, 11, 69, 140]

    for idx in idxs:
        eval_file = "E:/MyDocuments/Documents/Python/PycharmProjects/MasterThesis/server_output/report/" \
                "/resnet/resnet7_v3-32-32-L1_10-L2_0-GAN_1/eval_images/out_test_%d_output.png" % idx

        depth_file = os.path.join(depth_path, "%d.png" % idx)
        depth_img = cv2.imread(depth_file)
        eval_img = cv2.imread(eval_file)
        assert np.shape(depth_img) == (128, 256, 3)
        assert np.shape(eval_img) == (256, 256, 3)
        sss_img = eval_img[128:, :, :]
        new_img = np.concatenate((depth_img, sss_img), axis=0)
        new_file_name = os.path.join(dir_path, 'new_resnet_replace', "img_%d.png" % idx)
        cv2.imwrite(new_file_name, new_img)

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # replace_image_eval()
    dir_ = "E:\MyDocuments\Documents\master thesis\A5 Report Draft/figs"
    file_name = os.path.join(dir_, 'depth_example.png')
    file_name2 = os.path.join(dir_, 'depth_example2.png')
    img = cv2.imread(file_name)
    new_img = cv2.applyColorMap(process_norm(img), cv2.COLORMAP_JET)
    cv2.imwrite(file_name2, new_img)
