#%%
import torch
import torch.nn as nn
from torch import Tensor

from coco.cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%% original coco class index
# {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
#remapped index 
# remapped_labels={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
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

class data_COCO():
    def __init__(self, datadir: str, datatype: str, anchors: Tensor, num_anchors_per_scale: int=3, scale_size: list[int]=[13, 26, 52], iou_thresh: float=0.5, target_size: list[int]=[416,416]):
        self.datadir = datadir
        self.datatype = datatype

        self.coco = COCO(f'{datadir}/annotations/instances_{datatype}.json')
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds(self.cats)
        self.imgIds = self.coco.getImgIds(catIds = self.catIds)
        self.target_size = target_size

        self.anchors = anchors
        self.num_anchors_per_scale = num_anchors_per_scale
        self.scale_size = scale_size
        self.iou_thresh = iou_thresh

        self.coco_to_yolo_dict = {cat['id']: i for i, cat in enumerate(self.cats)}
        self.yolo_to_coco_dict = {i: cat['id'] for i, cat in enumerate(self.cats)}
        self.coco_to_yolo_dict = {cat['id']: i for i, cat in enumerate(self.cats)}

        self.cocolabels = {cat['id'] : cat['name'] for cat in self.cats}
        self.yololabels = {i: self.cocolabels[self.yolo_to_coco_dict[i]] for i in range(len(self.cocolabels))}
        self.num_classes = len(self.cocolabels)

    def __len__(self):
        return len(self.imgIds)
    
    def label(self, idx: int) -> str:
        return self.yololabels[idx]
    
    def size(self, idx: int) -> tuple[int, int]:
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        return img['width'], img['height']

    def coco_to_yolo(self, anns, img):
        labels = torch.tensor([self.coco_to_yolo_dict[ann['category_id']] for ann in anns], dtype = torch.long)
        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype = torch.float32)

        #some images may not have any annotations, check and discard
        if bbox.numel() == 0:
            # print(f'bbox.size() = {bbox.size()}')
            # print(f'labels.size() = {labels.size()}')
            # print('No annotations')
            return None, None

        # coco bbox format is xmin, ymin, w, h, where the +x is to the right and +y is down
        # we convert to yolo bbox format, xcenter, ycenter, w, h
        bbox[:, 0] = bbox[:, 0] + 0.5*bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] + 0.5*bbox[:, 3]
        bbox[:, [0,2]] /= img['width']
        bbox[:, [1,3]] /= img['height']

        return labels, bbox

    @staticmethod
    def yolo_to_coco(anns: list, img: np.array) -> tuple[Tensor, Tensor]:
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype = torch.long)
        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype = torch.float32)

        bbox[:, 0] = bbox[:, 0] - 0.5*bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] - 0.5*bbox[:, 3]
        bbox[:, [0,2]] *= img['width']
        bbox[:, [1,3]] *= img['height']

        return labels, bbox

    def labels_to_bbox(self, labels: Tensor) -> list[Tensor]:
        """
        Return all bounding boxes for multiple images 

        Args: 
            labels (tensor): train_labels is [BS, A, I, J, O] where:
                BS is batch_size,
                A is anchor index (three sizes per scale),
                I is anchor position index x-position,
                J is anchor position index y-position,
                O is the label output [obj, x, y, w, h, class index]

        Return:
            bbox (list[Tensor]): bbox is [num_bboxes, O] where O is (xmid, ymid, w, h, class_idx, img_idx)
        """

        obj = labels[...,0] == 1
        indices = obj.nonzero()
        bbox = torch.zeros((indices.size(0),6))

        # print(f'label.shape={label.shape}')
        # width, height = self.size(idx) #width and height of the original image
        width, height = 1., 1.

        bbox[:,0] = (labels[...,1][obj] + indices[:,1]) * width / self.scale_size[0]
        bbox[:,1] = (labels[...,2][obj] + indices[:,2]) * height / self.scale_size[0]
        bbox[:,2] = labels[...,3][obj] * width / self.scale_size[0]
        bbox[:,3] = labels[...,4][obj] * height / self.scale_size[0]
        bbox[:,4] = labels[...,5][obj] #label index
        bbox[:,5] = idx #img index

        return bbox

    def label_to_bbox(self, idx: int, label: Tensor = None) -> list[Tensor]:
        """
        Return all bounding boxes for a single image in the YOLO format (xcenter, ycenter, w, h).  If label is not None, convert the label
        data to YOLO format bounding boxes.  Otherwise, get the label data based on the idx input.  Note that the bounding boxes are computed for 
        the original image (not normalized) using the width and height from the original image.

        Args: 
            idx (int): image index
            label (tensor): label is [A, I, J, O] where:
                A is anchor index (three sizes per scale),
                I is anchor position index x-position,
                J is anchor position index y-position,
                O is the label output [obj, x, y, w, h, class index]           

        Return:
            bbox (list[Tensor]): list of bounding box tensors [xmid, ymid, w, h, class_index]
        """
        if label == None:
            _, label = self.__getitem__(idx)
            label = label[0]

        print(f'label.shape={label.shape}')
        width, height = self.size(idx) #width and height of the original image

        obj = label[...,0] == 1
        indices = obj.nonzero()
        bbox = torch.zeros((indices.size(0),5))

        bbox[:,0] = (label[...,1][obj] + indices[:,1]) * width / self.scale_size[0]
        bbox[:,1] = (label[...,2][obj] + indices[:,2]) * height / self.scale_size[0]
        bbox[:,2] = label[...,3][obj] * width / self.scale_size[0]
        bbox[:,3] = label[...,4][obj] * height / self.scale_size[0]
        bbox[:,4] = label[...,5][obj] #label index

        return bbox
    
    def plot_bboxes(self, idx: int, labels: Tensor = None, figsize: tuple = (30,30)) -> None:
        """
        Plot the bounding boxes from the COCO dataset.  The bounding box data in the data_COCO class is stored in YOLO format (xcenter, ycenter) but 
        is transformed to (xmin, ymin) for plotting with matplotlib.  If labels is None, then the label data is obtained based on the image index.

        Args: 
            idx (int): image index
            labels (Tensor): Bounding box label data from COCO dataset formatted for the YOLO model.  Tensor is formatted as the following: 
            batch_size, obj, x, y, w, h, class index
            figsize (tuple[float]): image index
        Return:
            None
        """
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        if labels == None:
            img, labels = self.__getitem__(idx)

        bbox = self.label_to_bbox(idx, labels[0])
        patch = [Rectangle((b[0]-0.5*b[2],b[1]-0.5*b[3]), b[2], b[3]) for b in bbox] #yolo format is (xcenter, ycenter) but matplotlib is (xmin, ymin)
        collection = PatchCollection(patch, linewidth=3, linestyle='--', edgecolor='b', facecolor='none')

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.add_collection(collection) #add all rectangles at once

        for idx, b in enumerate(bbox):
            ax.text(b[0]-0.5*b[2], b[1]-0.5*b[3], self.label(int(b[4])), ha='center', va='center', fontsize=24, color='red')

        ax.imshow(img)
        # plt.show()

    @staticmethod
    def letterbox(img, target_size = [640,640]):
        #img format is h, w, c
        import cv2
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        # print(f'w={w}, h={h}, scale={scale}')

        top, bottom = (target_size[1]-int(scale*h))//2, target_size[1] - (target_size[1]-int(scale*h))//2 - int(scale*h)
        left, right = (target_size[0]-int(scale*w))//2, target_size[0] - (target_size[0]-int(scale*w))//2 - int(scale*w)
        # top, bottom = (target_size[1]-int(scale*h))//2, target_size[1] - (target_size[1]-int(scale*h))//2 - h
        # left, right = (target_size[0]-int(scale*w))//2, target_size[0] - (target_size[0]-int(scale*w))//2 - w
        # print(f'int(scale*h)={int(scale*h)}, h={h}')
        # print(f'int(scale*w)={int(scale*w)}, w={w}')
        # print(f'c={img.shape[2]}, h={h}, w={w}')

        img_resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
        # print(f'top={top}, side={left}')
        # return torch.tensor(img_resized.reshape(3, target_size[0], target_size[1])), (scale, scale), (left, top)
        # return torch.tensor(img_resized), scale, (left, top)
        return torch.tensor(img_resized), scale, (left, bottom)

    def rescaled_bbox(self, bbox, scale, padding, img):
        # print(f'padding = {padding}')
        bbox = bbox.clone()
        bbox *= scale

        #DOUBLE-CHECK THE PADDING IS DONE CORRECTLY!! FOR NOW IT IS CLAMPED!!!
        bbox[:, 0] += padding[0]/img['width']
        bbox[:, 1] += padding[1]/img['height']
        bbox[:, 0:4] = torch.clamp(bbox[:, 0:4], max=(1.0-1.e-6))
        # print(f'padding[1]={padding[1]}, img[height]={img['height']}, padding[1]//img[height]={padding[1]//img['height']}')
        return bbox

    def convert_to_RGB(self, I):
        from skimage import color
        if I.ndim == 2:  # grayscale
            return color.gray2rgb(I)  # expand to (H, W, 3)
        elif I.shape[2] == 4:  # RGBA
            return I[..., :3]
        else:
            return I

    def __getitem__(self, img_idx: int, img_resize: bool = True) -> tuple[Tensor, tuple[Tensor]]:
        """
        Post-process the COCO dataset into training examples for the YOLO model.  The bounding boxes are transformed into anchor indices for each
        scale.

        Args: 
            img_idx (int): image index
            img_resize (bool): Bounding box label data from COCO dataset formatted for the YOLO model.  Tensor is formatted as the following: 
            batch_size, obj, x, y, w, h, class index

        Return:
            I, output, img_idx (tuple[Tensor, tuple(Tensor), int]): Returns the image, label data as tuple output for the three scales and image index
            The train_labels format is is [S, BS, A, I, J, O] where:
                S is the scale index,
                BS is batch_size,
                A is anchor index (three sizes per scale),
                I is anchor position index x-position,
                J is anchor position index y-position,
                O is the label output [obj, x, y, w, h, class index]
        """

        img = self.coco.loadImgs(self.imgIds[img_idx])[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)
        train_label = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.scale_size]

        if img_resize == True:
            I, scale, padding = self.letterbox(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}')), self.target_size)
        else:
            I, scale, padding = torch.tensor(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}'))), 1., (0, 0)

        labels, bbox_normalized = self.coco_to_yolo(anns, img)
        if labels is None:
            return I.detach().clone(), tuple(train_label), int(img_idx)

        bbox_normalized = self.rescaled_bbox(bbox_normalized, scale, padding, img)
        ious = self.compute_iou_wh(bbox_normalized[:,2:4], self.anchors)

        # Loop through the ious for all bounding boxes.
        # Determine the best anchor box size based on iou (for each scale)
        # If the iou is greather than threshold but not the highest, set flag = -1

        # Looping over bounding boxes
        for bbox_idx, iou in enumerate(ious):
            anchor_iou = iou.argsort(descending=True, dim=1)

            #Looping over scales
            for idx, anchor_idx in enumerate(anchor_iou.view(-1)):
                scale_idx = idx // self.num_anchors_per_scale
                if idx % self.num_anchors_per_scale == 0: #reset anchor_set = False for each new scale
                    anchor_set = False

                i, j = self.scale_size[scale_idx] * bbox_normalized[bbox_idx, 0], self.scale_size[scale_idx] * bbox_normalized[bbox_idx, 1]  # which cell
                # print(f'idx={idx}, i={i}, j={j}, scale_size[scale_idx]={self.scale_size[scale_idx]}, xbox={bbox_normalized[bbox_idx, 0]}, ybox={bbox_normalized[bbox_idx, 1]}')

                if train_label[scale_idx][anchor_idx, int(i), int(j), 0] != 1 and anchor_set == False:
                    x_cell, y_cell = i - int(i), j - int(j)  # return fraction parts of i and j
                    width_cell, height_cell = (
                        bbox_normalized[bbox_idx, 2] * self.scale_size[scale_idx],
                        bbox_normalized[bbox_idx, 3] * self.scale_size[scale_idx],
                    )  # can be greater than 1 since it's relative to cell
                    abox = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # train_label[scale_idx][anchor_idx, int(i), int(j), 0] = 1
                    train_label[scale_idx][anchor_idx, int(i), int(j), 0] = int(1)
                    train_label[scale_idx][anchor_idx, int(i), int(j), 1:5] = abox
                    train_label[scale_idx][anchor_idx, int(i), int(j), 5] = labels[bbox_idx]

                    # print(f'idx={idx}, abox={abox}')
                    # print(f'train_label[scale_idx][anchor_idx, int(i), int(j), 1:5]={train_label[scale_idx][anchor_idx, int(i), int(j), 1:5]}')
                    # print(f'labels[bbox_idx]={labels[bbox_idx]}')

                    anchor_set = True
                elif anchor_set == True and iou[scale_idx][anchor_idx] > self.iou_thresh:
                    train_label[scale_idx][anchor_idx, int(i), int(j), 0] = -1
 
        #image needs to be of type float32 and label needs to be torch long (the label is an index, not name)
        return I.detach().clone(), tuple(train_label) # O is obj, x, y, w, h, class index
        # return img, tuple(train_label) # O is obj, x, y, w, h, class index

    
    # calculate iou between abox and bbox where iou is defined as: area of overlap / area of union
    # bbox format is x, y, w, h, where the +x is to the right and +y is down
    @staticmethod
    def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
        """
        Compute pairwise IoU between two sets of boxes.

        Args:
            box1: [M, 4]
            box2: [N, 4]

        Returns:
            IoU: [M, N] where the IoU is computed between bbox1 from the i-th row and bbox2 from the j-th column 
        """
        # Coordinates of intersection
        inter_x1 = torch.max(box1[:, None, 0], box2[:, 0]) #box1[:,None,0].shape = [M,1], box2[:,0].shape = [N,] => output [M,N]
        inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) #area1.shape = [M]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) #area2.shape = [N]

        return inter_area / (area1[:, None] + area2 - inter_area + 1e-6)

    # @staticmethod
    # def compute_iou(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    #     """
    #     Compute pairwise IoU between bbox1 and bbox2.
        
    #     Args:
    #         bbox1 (tensor): [M, 4]
    #         bbox2 (tensor): [N, 4]
    #     Returns:
    #         IoU (tensor): [M, N]
    #     """

    #     diff = bbox1[...,0] - bbox2[...,0] # [M,N]
    #     w = diff > 1.e-6 #[M,N]

    #     w[diff>1.e-6] = max(0., 0.5*(bbox1[...,2] + bbox2[...,2]) - abs(bbox1[...,0] - bbox2[...,0]))
    #     w[diff<=1.e-6] = torch.min(bbox1[...,0], bbox2[...,0])

    #     # if abs(bbox1[...,0] - bbox2[...,0]) > 1.e-6:
    #     #     w = max(0., 0.5*(bbox1[...,2] + bbox2[...,2]) - abs(bbox1[...,0] - bbox2[...,0])) #if there is no intersection, set negative result to 0
    #     # else:
    #     #     w = torch.min(bbox1[...,0], bbox2[...,0])
    
    #     if abs(bbox1[...,1] - bbox2[...,1]) > 1.e-6:
    #         h = max(0., 0.5*(bbox1[...,3] + bbox2[...,3]) - abs(bbox1[...,1] - bbox2[...,1]))
    #     else:
    #         h = torch.min(bbox1[...,1], bbox2[...,1])

    #     # w = (bbox1[..., 2]/2 + bbox2[..., 2]/2 - torch.abs(bbox1[..., 0] - bbox2[..., 0])).clamp(min=0)
    #     # h = (bbox1[..., 3]/2 + bbox2[..., 3]/2 - torch.abs(bbox1[..., 1] - bbox2[..., 1])).clamp(min=0)

    #     area_intersect = w*h
    #     area1 = bbox1[..., 0]*bbox1[..., 1]
    #     area2 = bbox2[..., 0]*bbox2[..., 1]
    #     # print(area_intersect.size(), area_bbox.size(), area_anchor.size())
    #     return area_intersect/(area1 + area2 - area_intersect + 1.e-6)

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


class data_COCOeval(data_COCO):
    def __init__(self, datadir: str, datatype: str, anchors: Tensor, num_anchors_per_scale: int=3, scale_size: list[int]=[13, 26, 52], iou_thresh: float=0.5, target_size: list[int]=[416,416]):
        super.__init__(datadir, datatype, anchors, num_anchors_per_scale, scale_size, iou_thresh, target_size)

    def __getitem__(self, img_idx, img_resize = True):
        """
        Post-process the COCO dataset into training examples for the YOLO model.  The bounding boxes are transformed into anchor indices for each
        scale.

        Args: 
            img_idx (int): image index
            img_resize (bool): Bounding box label data from COCO dataset formatted for the YOLO model.  Tensor is formatted as the following: 
            batch_size, obj, x, y, w, h, class index

        Return:
            I, output, img_idx (tuple[Tensor, tuple(Tensor), int]): Returns the image, label data as tuple output for the three scales and image index
            The train_labels format is is [S, BS, A, I, J, O] where:
                S is the scale index,
                BS is batch_size,
                A is anchor index (three sizes per scale),
                I is anchor position index x-position,
                J is anchor position index y-position,
                O is the label output [obj, x, y, w, h, class index]
        """

        img = self.coco.loadImgs(self.imgIds[img_idx])[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)
        train_label = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.scale_size]

        if img_resize == True:
            I, scale, padding = self.letterbox(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}')), self.target_size)
        else:
            I, scale, padding = torch.tensor(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}'))), 1., (0, 0)

        labels, bbox_normalized = self.coco_to_yolo(anns, img)
        if labels is None:
            return I.detach().clone(), tuple(train_label), int(img_idx)

        bbox_normalized = self.rescaled_bbox(bbox_normalized, scale, padding, img)

        return I.detach().clone(), labels, bbox_normalized, img_idx # O is obj, x, y, w, h, class index

#%% test
if __name__ == "__main__":
    data = data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)
    print(data.__len__())

#%%
    # print(data.__dict__.keys())
    dir(data_COCO)

#%%
    # data.plot_bboxes(15)
    # data.plot_bboxes(96) #chair is hard to see
    data.plot_bboxes(25) #no labels


    # data.plot_bboxes(39) #no labels

    #%%
    dataloader = torch.utils.data.DataLoader(data, batch_size = 3, shuffle = False)
    imgs, train_labels, idx = next(iter(dataloader)) #returns a list of tensors
    
    print(len(imgs))
    print(len(train_labels[0]))
    print(idx)
 
    print(f"Image batch shape: {imgs[0].size()}")
    print(f"Labels batch shape: {train_labels[0].size()}")
    # print(f"Labels batch shape: {train_labels[1].size()}")
    # print(f"Labels batch shape: {train_labels[2].size()}")

    obj = train_labels[0][...,0] == 1
    print(train_labels[0][obj].size)
    # print([obj])

    # bbox = data.label_to_bbox(0, label = train_labels[0][0])
    # print(len(bbox), bbox)
    # data.plot_bboxes(idx[0], figsize=(30,18))

    bbox = data.labels_to_bbox(train_labels[0], idx)


    # img = imgs[0]
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.imshow(img)
    # plt.show()

    # rows, cols = 2, 3
    # figsize=(30,18)
    # fig, ax = plt.subplots(rows, cols, figsize=figsize)
    # for i, img in enumerate(imgs):
    #     plt.subplot(rows, cols, i+1)
    #     # data.plot_bboxes(idx[i], figsize=(10,6)) #no labels
    #     plt.imshow(img)
    # plt.show()


    #%%
    print(dataloader.device)

    #%%
    no_annot = []
    for i, data in enumerate(dataloader):
        # print(data[0].device)
        print(f'data[0].size() = {data[0].size()}')
        print(f'data[1][0].size() = {data[1][0].size()}')
        print(f'data[1][1].size() = {data[1][1].size()}')
        print(f'data[1][2].size() = {data[1][2].size()}')

        data[1][0] = data[1][0].to('cuda')
        data[1][1] = data[1][1].to('cuda')
        data[1][2] = data[1][2].to('cuda')
        # print(data[0].device)
        train_features, train_labels = data[0].to('cuda'), data[1]
        print(i)
    #%%
    test = torch.tensor([])
    print(test.size())

    if test.numel() == 0:
        print('no elements')


    #%%
    idx = 23
    img, labels = data.__getitem__(idx)
    width, height = data.size(idx)
    print(f'width={width}, height={height}')

    print(labels[0].size())

    #idx = 16, 22, 1680, 160, 54, 999






# %%
input = 'I ended up calling their Honeywell customer support and they asked me for the serial code'

n = 3
k = 0
str_dict = {}
temp = ''
for char in input:
    if char == ' ' or char == '\t':
        continue

    temp += char
    k += 1
    # print(f'temp={temp}')

    if (k) % n == 0:
        print(k, temp)
        if temp in str_dict.keys():
            str_dict[temp] += 1
        else:
            str_dict[temp] = 1

        temp = ''
    # print(char)

print(str_dict)
# %%
def parser(input: str, n: int) -> dict[str]:
    str_dict = {}
    temp = ''

    for char in input:
        if char == ' ' or char == '\t':
            continue

        temp += char

        if len(temp) == n:
            str_dict[temp] = str_dict.get(temp, 0) + 1
            temp = ''
    
    return str_dict

def parser2(input: str, n: int) -> dict[str]:
    str_dict = {}
    temp = ''

    for i, char in enumerate(input):
        if char == ' ' or char == '\t':
            continue

        idx = i
        while len(temp) != n and idx < len(input):
            if input[idx] == ' ' or input[idx] == '\t':
                pass
            else:
                temp += input[idx]
            
            print(f'idx={idx}, temp={temp}')
            idx += 1

        if len(temp) == n:
            str_dict[temp] = str_dict.get(temp, 0) + 1
        temp = ''
        # print(f'i={i}')

    return str_dict

#%%
test = 'Im a humidifier snob. i want warm mist only (warm/cold mist humidifiers ALWAYS leave residue even when on the warm mist setting) and i want it to turn on and off automatically to maintain the set humidity levels. shockingly few humidifiers fit these needs. this one does, and for under $100. its nearly silent, though i did have to tape a notecard over the display because i need 100 darkness to sleep. plus the components are dishwasher safe. this will be the only humidifier for me for the rest of my life'
test = 'aaa aaa aaa aaa what'
test = 'hello world'

output = parser2(test, 3)
print(output)


# %%
len(test)
test[506]
# %%

import numpy as np
test = [(0., 0.50196961, 0.5073157), (1., 0.45216657, 0.47749015)]
error = np.array(test).T

print(error)

error = error.T
error[0]
error[0]
error[1]
# %%

phi = np.array([[10, 0], [30, 0], [10, 20], [20, 20]])
# mu = np.array([[20,30], [20, -10]])
mu = np.array([[0,10], [30, 20]])

def get_labels(phi, mu):
    assigments = np.zeros( [phi.shape[0], mu.shape[0]] )
    for i, pt in enumerate(phi):
        assigments[i] = [np.linalg.norm(pt - mean)**2 for mean in mu]

    # return labels
    return np.argmin(assigments, axis=1) + 1 #returns the argmin for each element on the chosen axis

def update_centroids(phi, mu):
    labels = get_labels(phi, mu)
    labelled_data = np.column_stack((phi, labels))

    mu_new = np.zeros( [mu.shape[0],2] )

    for i in range(mu_new.shape[0]):
        mask = labelled_data[:,2] == i+1
        # test = np.sum(labelled_data[:,0:2][mask], axis=0)/mask.sum() #perform the summation along the specified axis (axis=0 are the rows)
        mu_new[i] = np.sum(labelled_data[:,0:2][mask], axis=0)/mask.sum()

        # print(f'masked_data={labelled_data[:,0:2][mask]}')
        # print(f'masked_datasum={np.sum(labelled_data[:,0:2][mask], axis=0)}')
        # print(f'test={test}')

    print(labelled_data)
    return mu_new

# print(get_labels(phi, mu))
# print(phi)

# print(update_centroids(phi, mu))


mu1 = update_centroids(phi, mu)
print(f'mu1={mu1}')
mu2 = update_centroids(phi, mu1)
print(f'mu2={mu2}')

# new = np.column_stack((phi, get_labels(phi,mu)))
# print(new)
# print(new[:,0])

# mask0 = new[:,2] == 1
# print(mask0)
# print(mask0.sum())
# print(new[:,0:2][mask0])
# print(np.sum(new[:,0:2][mask0],axis=1))

# test = get_labels(phi, mu)
# print(np.argmin(test, axis=1))

# phi.shape[0]
# mu.shape[0]

#%%
test = [1, 2, 3, 1, -1, 5, -5, 2]

def argmin(x: list) -> int:
    check = x[0]
    for i, item in enumerate(x):
        if item < check:
            idx = i
            check = x[idx]

    return idx

print(min(test))
print(range(len(test)))
print(min(range(len(test))))
print(min(range(len(test)), key=lambda k: test[k]))

print(argmin(test))

#%%
test = {'a': 1, 'b': 2}
test2 = {'a': 5, 'c': 4}

c = test.keys() + test2.keys()

# %%
from numpy import linalg
import math

test = np.array([1, 2, 3])
print(np.linalg.norm(test))
print(math.sqrt(1**2 + 2**2 + 3**2))

print(np.argmin(test))

# %%
test = [1, 2, 3]
sum(test)
# %%
import numpy as np

featureDim = 36
actions = [1, 2, 3]

a = np.random.standard_normal(size=(featureDim, len(actions)))
b = np.random.standard_normal(size=(featureDim))
# c = np.dot(a, b)

print(f'a.shape = {a.shape}')
print(f'b.shape = {b.shape}')
# print(f'bprime.shape = {bprime.shape}')

c = np.sum(a*b[:, None], axis = 0)
print(c.shape)
# %%
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
# %%
print(factorial(5))
# %%

def minimax(game, state):
    def recurse(state):
        if game.isEnd(state):
            return (game.Utility(state), None) #(value, action)
        
        actions = [ (recurse(game.succ(state,action))[0], action) for action in game.actions(state)]
        if game.player(state) == 1:
            return max(actions)
        elif game.player(state) == -1:
            return min(actions)
        
    utility, action = recurse(state)
    return action
