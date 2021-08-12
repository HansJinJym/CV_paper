# LDF

## Info

- Paper: Label Decoupling Framework for Salient Object Detection
- IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020
- Project: https://github.com/weijun88/LDF

## My Research

Used to change background of a given video.

Giving the original video and a background video, 
aiming to cut the foreground of the original video by LDF and merge it into the background video.

In ./train-fine,

- LDFbased-merger.py uses opencv to cut and merge frames

- LDFbased_merger_ffmpeg.py uses FFMPEG.

Result:
For a simpler background of the original video, the result is better, 
due to the essential problem of SOD.
