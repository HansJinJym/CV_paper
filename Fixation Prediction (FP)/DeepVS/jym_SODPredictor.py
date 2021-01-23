from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import sys
import numpy as np
from config import *
from utilities import postprocess_predictions
from jym_newASNet import sam_vgg
from scipy.misc import imread, imsave, imresize
import random


def get_test(data):
    Xims_224 = np.zeros((1, 224, 224, 3))           # （图片数量，图片宽度，图片高度，通道数）
    img = imread(data['image'])
    img_name = os.path.basename(data['image'])
    gaussian = np.zeros((1, 14, 14, nb_gaussian))
    if img.ndim == 2:
        copy = np.zeros((img.shape[0], img.shape[1], 3))
        copy[:, :, 0] = img
        copy[:, :, 1] = img
        copy[:, :, 2] = img
        img = copy
    r_img = imresize(img, (224, 224))
    r_img = r_img.astype(np.float32)
    r_img[:, :, 0] -= img_channel_mean[0]
    r_img[:, :, 1] -= img_channel_mean[1]
    r_img[:, :, 2] -= img_channel_mean[2]
    r_img = r_img[:, :, ::-1]
    Xims_224[0, :] = np.copy(r_img)
    #imsave('saliency_predictions/' + '%s_Ximg.png' % img_name[0:-4], Xims_224)
    return [Xims_224, gaussian], img, img_name


# if __name__ == '__main__':
def main_ASNet():
    if len(sys.argv) != 1:
        raise NotImplementedError
    else:
        seed = 7
        random.seed(seed)
        test_data = []

        testing_images = [datasest_path + f for f in os.listdir(datasest_path) if
                           f.endswith(('.jpg', '.jpeg', '.png'))]
        testing_images.sort()

        for image in testing_images:
            annotation_data = {'image': image}
            test_data.append(annotation_data)

        phase = 'test'
        if phase == "test":
            x = Input(batch_shape=(1, 224, 224, 3))
            x_maps = Input(batch_shape=(1, 14, 14, nb_gaussian))
            m = Model(inputs=[x, x_maps], outputs=sam_vgg([x, x_maps]))

            print("Loading weights")
            m.load_weights('ASNet.h5')
            print("Making prediction")

            saliency_output = 'saliency_predictions/'
            if not os.path.exists(saliency_output):
                os.makedirs(saliency_output)

            for data in test_data:
                Ximg, original_image, img_name = get_test(data)
                # print('Ximg: ' + str(Ximg))
                # print('original_image: ' + str(original_image))
                # print('img_name: ' + str(img_name))
                # imsave(saliency_output + '%s_Ximg.png' % img_name[0:-4], Ximg[0].astype(int))
                # imsave(saliency_output + '%s_oriimage.png' % img_name[0:-4], original_image.astype(int))
                predictions = m.predict(Ximg, batch_size=1)
                # imsave(saliency_output + '%s_pred.png' % img_name[0:-4], predictions[6][0, :, :, 0].astype(int))
                # print('pred: ' + str(predictions[6][0, :, :, 0]))       # 几乎全黑，都是0.几

                #################### 七层结果输出 ####################
                for im in range(0, 7):
                    fp = postprocess_predictions(predictions[im][0, :, :, 0], original_image.shape[0],
                                                 original_image.shape[1])
                    imsave(saliency_output + '%s_layer' % img_name[0:-4] + str(im) + '.png', fp.astype(int))

                res_saliency = postprocess_predictions(predictions[6][0, :, :, 0], original_image.shape[0],
                                                       original_image.shape[1])
                imsave(saliency_output + '%s.png' % img_name[0:-4], res_saliency.astype(int))

                # for image in range(0, 7):
                #     print("predictions[" + str(image) +"] shape: " + str(predictions[image].shape))
                #     print("final_output[" + str(image) +"] shape: " + str(predictions[image][0, :, :, 0].shape))
                # fp = predictions[0]
                # print(str(fp.shape) + str(type(fp)))

                m.reset_states()
        else:
            raise NotImplementedError