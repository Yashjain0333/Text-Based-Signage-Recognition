# -*- coding: utf-8 -*-
import os
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import cv2
import imgproc
from bisect import bisect_right as upper_bound
from PIL import Image
import pytesseract
import pyrealsense2 as rs
import statistics
import numpy as np
from ransac import *
from plane_fitting import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def ocr(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
    except:
        text = ""

    return text

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def binaryMedian(m, r, d):
    for i in range(0,r):
        m[i].sort()
    mi = m[0][0]
    mx = 0
    for i in range(r):
        if m[i][0] < mi:
            mi = m[i][0]
        if m[i][d-1] > mx :
            mx =  m[i][d-1]
     
    desired = (r * d + 1) // 2
     
    while (mi < mx):
        mid = mi + (mx - mi) // 2
        place = [0];
         
        # Find count of elements smaller than mid
        for i in range(r):
             j = upper_bound(m[i], mid)
             place[0] = place[0] + j
        if place[0] < desired:
            mi = mid + 1
        else:
            mx = mid
    return mi

def drawboxes(frame, img, boxes, texts, verts, verticals=None, dirname='./result/'):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        f = 0
        pc = rs.pointcloud()
        for i, box in enumerate(boxes):
            f = f+1
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)

            cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
            res = cv2.bitwise_and(img,img,mask = mask)
            rect = cv2.boundingRect(poly) # returns (x,y,w,h) of the rect
            cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

            # print("C1")
            try:
                # texts.append(ocr(cropped))  # Tesseract OCR
                batch_max_length = 25
                character = string.printable[:-6]
                converter = CTCLabelConverter(character)
                num_class = len(converter.character)
                batch_size = 1
                text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)
                print(preds_str)
            except:
                continue
            # print("C2")
            xmin, xmax, ymin, ymax = rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3]  # Bounding Box

            roi = verts[ymin:ymax, xmin:xmax, :]

            # Pass xyz to Open3D.o3d.geometry.PointCloud
            xyzs = np.reshape(roi, (-1, 3))  # xyzs = np.reshape(verts, (-1, 3))
            arr = []
            for v in xyzs:
                if v.all() != 0:
                    arr.append(v)

            del xyzs
            xyzs = np.array(arr)
            # print(xyzs)
            # print("Done Printing points.")
            fig = plt.figure()
            ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)

            def plot_plane(a, b, c, d):
                xx, yy = np.mgrid[:10, :10]
                return xx, yy, (-d - a * xx - b * yy) / c

            max_iterations = 500
            print(xyzs.shape[0])
            goal_inliers = xyzs.shape[0]*0.9
            xyzs[:50, 2:] = xyzs[:50, :1]
            ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

            # RANSAC
            m, best_inliers = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.001), 3, goal_inliers, max_iterations)
            a, b, c, d = m
            xx, yy, zz = plot_plane(a, b, c, d)
            ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

            plt.show()

            cv2.imshow("output", cropped)
            cv2.waitKey(5000)

        return img

def saveResult(char_boxes, img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        
        if(char_boxes==True):
            res_img_file = dirname + "cboxes_" + filename + '.jpg'
        else:
            res_img_file = dirname + "wboxes_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)