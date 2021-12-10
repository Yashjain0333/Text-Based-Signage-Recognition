# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
from bisect import bisect_right as upper_bound
from PIL import Image
import pytesseract
import statistics

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

def drawboxes(frame, img, boxes, texts, depth_img, verticals=None, dirname='./result/'):
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

        for i, box in enumerate(boxes):
            f = f+1
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)

            cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
            res = cv2.bitwise_and(img,img,mask = mask)
            rect = cv2.boundingRect(poly) # returns (x,y,w,h) of the rect
            cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

            depth_mask = np.zeros(depth_img.shape[0:2], dtype=np.uint8)

            cv2.drawContours(depth_mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
            depth_res = cv2.bitwise_and(depth_img, depth_img, mask = depth_mask)
            depth_cropped = depth_res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            
            try:
                texts.append(ocr(cropped))
            except:
              continue

            avg_depth = 0
            # cv2.imshow("output", cropped)
            # cv2.waitKey(20)
            res_file = dirname + str(frame) + "_" + "textno" + str(f) + '.jpg'
            cv2.imwrite(res_file, cropped)
            res_file = dirname + str(frame) + "_" + "textno" + str(f) + '.txt'
            grid = np.asarray(depth_cropped)
            r = grid.shape[0]
            c = grid.shape[1]
            median = binaryMedian(grid, r, c)
            print(median)
            res = 0
            cnt = 0
            for i in range(0,r):
                for j in range(0,c):
                    if abs(median-grid[i][j]) <= 50:
                        res += grid[i][j]
                        cnt = cnt+1
            res /= cnt
            file1 = open(res_file,"w")
            file1.write(str(res))
            print(res)

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

