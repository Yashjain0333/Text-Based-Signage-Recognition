# -*- coding: utf-8 -*-
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import plane_detector
import file_utils
import file_utils1
import json
import zipfile
from realsensecv import RealsenseCapture
from craft import CRAFT
import pyrealsense2 as rs

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()
#for output
pathOut = 'video_out_large.avi'
fps = 24
frame_array = []
texts = []

""" For test images in a folder """
image_list, _, _ = file_utils1.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys, char_boxes = craft_utils.getWordAndCharBoxes(image, score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    char_boxes = craft_utils.adjustResultCoordinates(char_boxes, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, char_boxes

def get_image():
    # Define some constants
    resolution_width = 1280  # pixels
    resolution_height = 720  # pixels
    frame_rate = 15  # fps

    # Create a pipeline
    pipeline = rs.pipeline()

    # Enable depth, infrared, color streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    config.enable_stream(rs.stream.infrared)  # , 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    pc = rs.pointcloud()
    pipeline.start(config)

    colorizer = rs.colorizer()
    align_to = rs.stream.color
    align = rs.align(align_to)
    align1 = rs.align(align_to)

    colorizer = rs.colorizer()

    align = rs.align(rs.stream.depth)
    align1 = rs.align(rs.stream.color)

    # skip the fist Frames
    for i in range(10):
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

    aligned = align.process(frames)
    aligned1 = align1.process(frames)
    color_aligned_to_depth = aligned.first(rs.stream.color)
    color_aligned_to_depth1 = aligned1.first(rs.stream.color)

    depth_frame = frames.first(rs.stream.depth)
    color_frame = aligned.get_color_frame()
    color_frame1 = aligned1.get_color_frame()

    # Apply filters
    filters = [rs.disparity_transform(),
               rs.spatial_filter(),
               rs.temporal_filter(),
               rs.disparity_transform(False)]
    for f in filters:
        depth_frame = f.process(depth_frame)

    color = np.asanyarray(color_frame.get_data())
    color1 = np.asanyarray(color_frame1.get_data())
    points = pc.calculate(depth_frame)
    w = rs.video_frame(depth_frame).width
    h = rs.video_frame(depth_frame).height
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)

    # Stop pipeline
    pipeline.stop()
    return color1, vertices

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    k = 0
    while (1):
        k = k + 1
        image, verts = get_image()

        # cv2.imshow('OP',image)
        # cv2.waitKey(20)
        image = np.asarray(image)

        try:
            height, width, layers = image.shape
            bboxes, polys, score_text, char_boxes = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        except:
            break
        print("Done with CRAFT!")
        img = file_utils1.drawboxes(k,image, polys, texts, verts)

    texts = np.array(texts)
    texts = set(texts)
    # print(texts)
    cv2.destroyAllWindows()

    print("elapsed time : {}s".format(time.time() - t))