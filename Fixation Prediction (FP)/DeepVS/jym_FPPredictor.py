import numpy as np
import tensorflow as tf
import cv2
import time
import h5py
import OMCNN_2CLSTM as Network  # define the CNN
import random

import sys
import os
from tqdm import tqdm

batch_size = 1
framesnum = 16
inputDim = 448
input_size = (inputDim, inputDim)
resize_shape = input_size
outputDim = 112
output_size = (outputDim, outputDim)
epoch_num = 15
overlapframe = 5  # 0~framesnum+frame_skip

random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5
dp_in = 1
dp_h = 1

targetname = 'LSTMconv_prefinal_loss05_dp075_075MC100-200000'
VideoName = 'test20.mp4'
CheckpointFile = './model/pretrain/' + targetname


def _BatchExtraction(VideoCap, batchsize=batch_size, last_input=None, video_start=True):
    if video_start:
        _, frame = VideoCap.read()
        frame = cv2.resize(frame, resize_shape)
        frame = frame.astype(np.float32)
        frame = frame / 255.0 * 2 - 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Input_Batch = frame[np.newaxis, ...]
        for i in range(batchsize - 1):
            _, frame = VideoCap.read()
            frame = cv2.resize(frame, resize_shape)
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=0)
        Input_Batch = Input_Batch[np.newaxis, ...]
    else:
        Input_Batch = last_input[:, -overlapframe:, ...]
        for i in range(batchsize - overlapframe):
            _, frame = VideoCap.read()
            frame = cv2.resize(frame, resize_shape)
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[np.newaxis, np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=1)
    return Input_Batch


def main_DeepVS():
    net = Network.Net()
    net.is_training = False
    input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
    RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    net.inference(input, RNNmask_in, RNNmask_h)
    net.dp_in = dp_in
    net.dp_h = dp_h
    predicts = net.out

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, CheckpointFile)
    VideoName_short = VideoName[:-4]
    VideoCap = cv2.VideoCapture(VideoName)

    VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        'New video: %s with %d frames and size of (%d, %d)' % (VideoName_short, VideoFrame, VideoSize[1], VideoSize[0]))
    fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
    videoWriter = cv2.VideoWriter(VideoName_short + '_result.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                                  output_size, isColor=False)
    start_time = time.time()
    videostart = True

    print("--------------------------------------------------------")
    print("Starting predicting...")
    print("predicts shape: " + str(predicts.shape))
    print("predicts type: " + str(type(predicts)))
    while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - framesnum - frame_skip + overlapframe:
        if videostart:
            Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, video_start=videostart)
            videostart = False
            Input_last = Input_Batch
        else:
            Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, last_input=Input_last,
                                           video_start=videostart)
            Input_last = Input_Batch

        mask_in = np.ones((1, 28, 28, 128, 4 * 2))
        mask_h = np.ones((1, 28, 28, 128, 4 * 2))
        np_predict = sess.run(predicts, feed_dict={input: Input_Batch, RNNmask_in: mask_in, RNNmask_h: mask_h})
        print("\n--------------------------------")
        # print("np_predict: " + str(np_predict))
        # print("np_predict shape: " + str(np_predict.shape))
        # print("np_predict type: " + str(type(np_predict)))
        for index in range(framesnum):
            Out_frame = np_predict[0, index, :, :, 0]
            # print("Out_frame shape 1: " + str(Out_frame.shape))
            # print("Out_frame type 1: " + str(type(Out_frame)))		# (112, 112) numpy.ndarray
            Out_frame = Out_frame * 255
            # print("Out_frame shape 2: " + str(Out_frame.shape))
            # print("Out_frame type 2: " + str(type(Out_frame)))		# (112, 112) numpy.ndarray
            Out_frame = np.uint8(Out_frame)
            # print("Out_frame shape 3: " + str(Out_frame.shape))
            # print("Out_frame type 3: " + str(type(Out_frame)))		# (112, 112) numpy.ndarray
            videoWriter.write(Out_frame)

        print("Testing space, for parameter: predicts, np_predicts.")
        test, np_test, first_frame = predicts, np_predict, np_predict[0, 0, :, :, 0]
        print("predicts shape: " + str(test.shape))
        print("predicts type: " + str(type(test)))  # (1, 16, 112, 112, 1), tensor
        print("np_predicts shape: " + str(np_test.shape))
        print("np_predicts type: " + str(type(np_test)))  # (1, 16, 112, 112, 1), numpy.ndarray
        print("first frame shape: " + str(first_frame.shape))
        print("first frame type: " + str(type(first_frame)))  # (112, 112), numpy.ndarray
        tensor_test = tf.convert_to_tensor(np_test[:, 0, :, :, :])  # [:, 0, :, :, :]
        print("tensor_test: " + str(tensor_test))
        print("tensor_test shape: " + str(tensor_test.shape))  # (1, 112, 112, 1)
        print("tensor_test type: " + str(type(tensor_test)))
        tensor_test_resize = tf.image.resize_images(tensor_test, [14, 14])
        print("tensor_test_resize: " + str(tensor_test_resize))
        print("tensor_test_resize shape: " + str(tensor_test_resize.shape))  # (1, 14, 14, 1)
        print("tensor_test_resize type: " + str(type(tensor_test_resize)))
        # with tf.Session():
        # 	print(tensor_test_resize.eval())
        print("--------------------------------\n")
    # restFrame = VideoFrame - VideoCap.get(cv2.CAP_PROP_POS_FRAMES)
    # overlap = framesnum - restFrame + overlapframe
    # Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, last_input=Input_last,video_start=False)
    # mask_in = np.ones((1, 28, 28, 128, 4 * 2))
    # mask_h = np.ones((1, 28, 28, 128, 4 * 2))
    # np_predict = sess.run(predicts, feed_dict={input: Input_Batch, RNNmask_in: mask_in, RNNmask_h: mask_h})
    # for index in range(restFrame):
    #	Out_frame = np_predict[0, framesnum - restFrame + index, :, :, 0]
    #	Out_frame = Out_frame * 255
    #	Out_frame = np.uint8(Out_frame)
    #	videoWriter.write(Out_frame)

    duration = float(time.time() - start_time)
    print('Total time for this video %f' % (duration))
    # print(duration)
    print("Prediction finished.")
    print("--------------------------------------------------------")
    VideoCap.release()

    # global global_fp
    # global_fp = tensor_test_resize
    # # print(global_fp)

    return tensor_test_resize


if __name__ == '__main__':
    print("###########################################")
    print("In DeepVS, predicting FP...")
    fp_from_DeepVS = main_DeepVS()
    print("FP done, tensor returned.")
