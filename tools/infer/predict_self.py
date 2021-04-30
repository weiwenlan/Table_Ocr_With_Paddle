# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

from numpy.core.numeric import True_

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt
from PIL import ImageGrab
import csv


logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def judge_width(x0,x1,d):
    if abs(x0-x1)<d:
        return True
    else:
        return False
    


def caculate_center(box):

    x=(box[0][0]+box[2][0])/2
    y=(box[0][1]+box[2][1])/2
    return [x,y]




def out_table(boxes,recs):
    txts = [recs[i][0] for i in range(len(recs))]
    scores = [recs[i][1] for i in range(len(recs))]
    drop_score =0.5

    dots=[]
    all_width=0
    all_height=0
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
                continue
        #print(box)
        tmp=caculate_center(box)
        tmp.append(txt)
        tmp.append('unflag')
        # box got 4 (x,y) [[up left],[up right],[down right],[down left]]
        tmp.append(box[3][0])
        tmp.append(box[2][0])
        # Dots BOX INSIDE [CenterX, CenterY, TEXT. Flag, DownLeftX, DownRightX]
        dots.append(tmp)
        all_width+=int(abs(box[3][0]-box[2][0])/len(txt))
        all_height+=int(abs(box[2][1]-box[1][1]))

    spqce_w = int(all_width/len(dots))
    space_h = all_height/len(dots)
    space_h = 0.5*space_h

    rows={dots[0][1]:[dots[0]]}
    xrows=[dots[0][1]]
    for i in range(1,len(dots)):
        if dots[i][3]!='flaged':
            xflag=None
            for j in range(len(xrows)):
                if judge_width(dots[i][1],xrows[j],space_h):
                    xflag=j
            if xflag !=None:    
                rows[xrows[j]].append(dots[i])
                dots[i][3]='flaged'
            
            else:
                xrows.append(dots[i][1])
                rows[dots[i][1]]=[dots[i]]
                dots[i][3]='flaged'


    



    rows_sort={}
    for key in rows:
        rows_sort[key]=sorted(rows[key],key=lambda x:x[0])

    rows_align={}
    for key in rows_sort:
        rows_align[key]=[rows_sort[key][0]]
        for i in range(1,len(rows_sort[key])):
            if abs(rows_sort[key][i][4]- rows_align[key][-1][5]) < 2 * spqce_w:
                text_pre = rows_align[key][-1][2]
                text_nex = rows_sort[key][i][2]
                text_pre = text_pre + " " + text_nex
                rows_align[key][-1][2] = text_pre 
            else:
                rows_align[key].append(rows_sort[key][i])


    
    with open('example.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        for key in rows_align:
            row_text=[]
            for i in rows_sort[key]:
                row_text.append(i[2])
            logger.info(row_text)
            writer.writerow(row_text)
            #print(row_text)
    
        
    
   

def call_model(args):
    img = ImageGrab.grabclipboard()
    img.save('pic.png')
    img = cv2.imread('pic.png')
    text_sys = TextSystem(args)
    dt_boxes, rec_res = text_sys(img)
    out_table(dt_boxes,rec_res)


def main(args):
    # image_file_list = get_image_file_list(args.image_dir)
    # text_sys = TextSystem(args)
    # is_visualize = True
    # font_path = args.vis_font_path
    # drop_score = args.drop_score

    while True:
        instructions = input('Extract Table From Image ("?"/"h" for help,"x" for exit).')
        ins = instructions.strip().lower()
        if ins == 'x':
            break
        try:
            call_model(args)
        except KeyboardInterrupt:
            pass

'''
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        out_table(dt_boxes,rec_res)
'''


if __name__ == "__main__":
    main(utility.parse_args())
