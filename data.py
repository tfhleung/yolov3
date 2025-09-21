#%%
import torch
import torch.nn as nn

from coco.cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab

import itertools

#%%
# You can format your JSON document using Shift+Alt+F or Format Document from the context menu.
#person_keypoints_val2017
#person_keypoints_train2017

#each tuple corresponds to the width/height of an anchor box relative to the image size
#we need to determine the anchor box the has the highest IOU with the bounding box of the image
ANCHOR_BOXES = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
])

#%%
class data_COCO():
    def __init__(self, datadir, datatype, anchors, num_anchors_per_scale = 3, scale_size = [13, 26, 52], iou_thresh = 0.5):
        self.datadir = datadir
        self.datatype = datatype

        self.coco = COCO(f'{datadir}/annotations/instances_{datatype}.json')
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds(self.cats)
        self.imgIds = self.coco.getImgIds(catIds = self.catIds)
        self.labels = {cat['id'] : cat['name'] for cat in self.cats}

        self.anchors = anchors
        self.num_anchors_per_scale = num_anchors_per_scale
        self.scale_size = scale_size
        self.iou_thresh = iou_thresh

    def __len__(self):
        return len(self.imgIds)

    @staticmethod
    def coco_to_yolo(anns, img):
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype = torch.long)
        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype = torch.float32)

        bbox[:, 0] = bbox[:, 0] + 0.5*bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] + 0.5*bbox[:, 3]
        bbox[:, [0,2]] /= img['width']
        bbox[:, [1,3]] /= img['height']

        return labels, bbox

    @staticmethod
    def yolo_to_coco(anns, img):
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype = torch.long)
        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype = torch.float32)

        bbox[:, 0] = bbox[:, 0] - 0.5*bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] - 0.5*bbox[:, 3]
        bbox[:, [0,2]] *= img['width']
        bbox[:, [1,3]] *= img['height']

        return labels, bbox

    def __getitem__(self, idx):
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        I = io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}')

        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)

        output = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.scale_size]
        labels, bbox = self.coco_to_yolo(anns, img)

        ious = self.compute_iou_wh(bbox[:,2:4], self.anchors)

        # Loop through the ious for all bounding boxes.
        # Determine the best anchor box size based on iou (for each scale)
        # If the iou is greather than threshold but not the highest, set flag = -1

        # Looping over bounding boxes
        for bbox_idx, iou in enumerate(ious):
            anchor_iou = iou.argsort(descending=True, dim=1)

            #Looping over scales
            for idx, anchor_idx in enumerate(anchor_iou.view(-1)):
                scale_idx = idx // self.num_anchors_per_scale
                if idx % self.num_anchors_per_scale == 0:
                    anchor_set = False

                i, j = self.scale_size[scale_idx] * bbox[bbox_idx, 0], self.scale_size[scale_idx] * bbox[bbox_idx, 1]  # which cell

                if output[scale_idx][anchor_idx, int(i), int(j), 0] != 1 and anchor_set == False:
                    x_cell, y_cell = i - int(i), j - int(j)  # both between [0,1]
                    width_cell, height_cell = (
                        bbox[bbox_idx, 2] * self.scale_size[scale_idx],
                        bbox[bbox_idx, 3] * self.scale_size[scale_idx],
                    )  # can be greater than 1 since it's relative to cell
                    abox = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    output[scale_idx][anchor_idx, int(i), int(j), 0] = 1
                    output[scale_idx][anchor_idx, int(i), int(j), 1:5] = abox
                    output[scale_idx][anchor_idx, int(i), int(j), 5] = labels[bbox_idx]

                    anchor_set = True
                elif anchor_set == True and iou[scale_idx][anchor_idx] > self.iou_thresh:
                    output[scale_idx][anchor_idx, int(i), int(j), 0] = -1
 
        #image needs to be of type float32 and label needs to be torch long (the label is an index, not name)
        return I, tuple(output)
        # return I, torch.tensor(output, dtype=torch.long)
    
    # calculate iou between abox and bbox where iou is defined as: area of overlap / area of union
    # bbox format is x, y, w, h, where the +x is to the right and +y is down
    @staticmethod
    def compute_iou(bbox1, bbox2):
        if abs(bbox1[...,0] - bbox2[...,0]) > 1.e-6:
            w = max(0., 0.5*(bbox1[...,2] + bbox2[...,2]) - abs(bbox1[...,0] - bbox2[...,0])) #if there is no intersection, set negative result to 0
        else:
            w = torch.min(bbox1[...,0], bbox2[...,0])
    
        if abs(bbox1[...,1] - bbox2[...,1]) > 1.e-6:
            h = max(0., 0.5*(bbox1[...,3] + bbox2[...,3]) - abs(bbox1[...,1] - bbox2[...,1]))
        else:
            h = torch.min(bbox1[...,1], bbox2[...,1])

        aint = w*h
        a1 = bbox1[...,2]*bbox1[...,3]
        a2 = bbox2[...,2]*bbox2[...,3]
        return aint/(a1 + a2 - aint)

    @staticmethod
    def compute_iou_wh(bbox1, bbox2):
        """
        Compute IoU between bboxes and anchor boxes (width & height only)
        
        Args:
            bboxes: Tensor of shape (N, 2) → width & height of N boxes
            anchors: Tensor of shape (S, A, 2) → anchor grid (S scales, A anchors, w,h)
            
        Returns:
            Tensor of shape (N, S, A) → IoU of each bbox with each anchor
        """
        bbox1 = bbox1[:, None, None, :]   # (N,1,1,2), None keyword adds a singleton dimension
        bbox2 = bbox2[None, :, :, :]    # (1,S,A,2), for broadcasting to work, extra dimensions have to be added for the dimensions that do not match

        w = torch.min(bbox1[..., 0], bbox2[..., 0])
        h = torch.min(bbox1[..., 1], bbox2[..., 1])

        area_intersect = w*h
        area_bbox = bbox1[..., 0]*bbox1[..., 1]
        area_anchor = bbox2[..., 0]*bbox2[..., 1]
        # print(area_intersect.size(), area_bbox.size(), area_anchor.size())
        return area_intersect/(area_bbox + area_anchor - area_intersect)

#%%

#%%
if __name__ == "__main__":
    data = data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)
    print(data.__len__())
    #%%
    img, output = data.__getitem__(16)
    img, output = data.__getitem__(22)
    # plt.imshow(img)

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)

    print(output[0].size())
    print(output[0].size()[1])
    print(output[0].size()[2])
    print(output[1].size())
    print(output[2].size())

    #bounding boxes are defined as x_topleft, y_topleft, w, h - where x, y are the top left corner of the image (for the coco format)
    #YOLO format, x_center, y_center, w, h
    def convert_ij_to_xy(bbox, i, j, width, height, scale_size = 13):
        x = (bbox[0] + i) * width / scale_size
        y = (bbox[1] + j) * height / scale_size
        w = bbox[2] * width / scale_size
        h = bbox[3] * height / scale_size

        return torch.tensor([x, y, w, h], dtype=torch.float32)

    def plot_boxes(coord, fig_name):
        x = coord[0] - 0.5*coord[2]
        y = coord[1] - 0.5*coord[3]

        rect = patches.Rectangle(xy = (x, y), width = coord[2], height = coord[3], linewidth=1, linestyle = '--', edgecolor='r', facecolor='none')
        fig_name.add_patch(rect)

    # print(output[0][0])
    def check_outputs(output):
        print(f'size = {output.size()}')

        count = 0 
        for i in range(output.size()[1]):
            for j in range(output.size()[2]):
                for k in range(output.size()[0]):
                    if output[k,i,j,0] == 1:
                        # bbox0 = convert_ij_to_xy(output[k,i,j,1:5], i, j, 640, 480)
                        # bbox0 = convert_ij_to_xy(output[k,i,j,1:5], i, j, 399, 640)
                        bbox0 = convert_ij_to_xy(output[k,i,j,1:5], i, j, 640, 427)
                        print(f'i={i}, j={j}, anchor_idx={k}, label={data.labels[int(output[k,i,j,5])]}, bbox={output[k,i,j,1:5]}, bbox0={bbox0}')

                        plot_boxes(bbox0, ax)
                        count += 1

        print(f'count = {count}')

    check_outputs(output[0])
    plt.show()

    #%%
    # obj = output[...,0] == 1
    obj = output[0][...,0] == 1
    noobj = output[0][...,0] == 0
    # print(obj)

    print(obj.size())
    print(output[0].size())
    print(output[0][...,0].size())
    print(output[0][...,0:1].size()) # we specify 0:1 instead of 0 in order to preserve the extra dimension (for broadcasting)
    print(output[0][...,0:1][obj].size())
    print(output[0][...,0:1][noobj].size())

    # print(output[0][...,0:1][obj])
    # print(output[0][...,0:1])
# %%
