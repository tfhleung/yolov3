#%%
import torch
import torch.nn as nn

from coco.cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
# {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

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
    def __init__(self, datadir, datatype, anchors, num_anchors_per_scale = 3, scale_size = [13, 26, 52], iou_thresh = 0.5, target_size = [416,416]):
        self.datadir = datadir
        self.datatype = datatype

        self.coco = COCO(f'{datadir}/annotations/instances_{datatype}.json')
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds(self.cats)
        self.imgIds = self.coco.getImgIds(catIds = self.catIds)
        self.labels = {cat['id'] : cat['name'] for cat in self.cats}
        self.target_size = target_size

        self.anchors = anchors
        self.num_anchors_per_scale = num_anchors_per_scale
        self.scale_size = scale_size
        self.iou_thresh = iou_thresh

        self.coco_to_yolo_dict = {cat['id']: i for i, cat in enumerate(self.cats)}

    def __len__(self):
        return len(self.imgIds)

    def coco_to_yolo(self, anns, img):
        # labels = torch.tensor([ann['category_id'] for ann in anns], dtype = torch.long)
        labels = torch.tensor([self.coco_to_yolo_dict[ann['category_id']] for ann in anns], dtype = torch.long)
        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype = torch.float32)

        #some images may not have any annotations, check and discard
        if bbox.numel() == 0:
            print(f'bbox.size() = {bbox.size()}')
            print(f'labels.size() = {labels.size()}')
            print('No annotations')
            return None, None

        # coco bbox format is xmin, ymin, w, h, where the +x is to the right and +y is down
        # we convert to yolo bbox format, xcenter, ycenter, w, h
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

    def __getitem__(self, idx, img_resize = True):
        #need to map idx so that it is continous from 0 to 5000
        print(f'idx = {idx}')
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)
        # output = [torch.zeros((self.num_anchors_per_scale, S, S, 6), dtype=torch.long) for S in self.scale_size]
        output = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.scale_size]

        if img_resize == True:
            I, scale, padding = self.letterbox(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}')), self.target_size)
        else:
            I, scale, padding = torch.tensor(self.convert_to_RGB(io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}'))), 1., (0, 0)

        labels, bbox_normalized = self.coco_to_yolo(anns, img)
        if labels is None:
            return I.detach().clone(), tuple(output)

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

                if output[scale_idx][anchor_idx, int(i), int(j), 0] != 1 and anchor_set == False:
                    x_cell, y_cell = i - int(i), j - int(j)  # return fraction parts of i and j
                    width_cell, height_cell = (
                        bbox_normalized[bbox_idx, 2] * self.scale_size[scale_idx],
                        bbox_normalized[bbox_idx, 3] * self.scale_size[scale_idx],
                    )  # can be greater than 1 since it's relative to cell
                    abox = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # output[scale_idx][anchor_idx, int(i), int(j), 0] = 1
                    output[scale_idx][anchor_idx, int(i), int(j), 0] = int(1)
                    output[scale_idx][anchor_idx, int(i), int(j), 1:5] = abox
                    output[scale_idx][anchor_idx, int(i), int(j), 5] = labels[bbox_idx]

                    # print(f'idx={idx}, abox={abox}')
                    # print(f'output[scale_idx][anchor_idx, int(i), int(j), 1:5]={output[scale_idx][anchor_idx, int(i), int(j), 1:5]}')
                    # print(f'labels[bbox_idx]={labels[bbox_idx]}')

                    anchor_set = True
                elif anchor_set == True and iou[scale_idx][anchor_idx] > self.iou_thresh:
                    output[scale_idx][anchor_idx, int(i), int(j), 0] = -1
 
        #image needs to be of type float32 and label needs to be torch long (the label is an index, not name)
        return I.detach().clone(), tuple(output)
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

        # w = (bbox1[..., 2]/2 + bbox2[..., 2]/2 - torch.abs(bbox1[..., 0] - bbox2[..., 0])).clamp(min=0)
        # h = (bbox1[..., 3]/2 + bbox2[..., 3]/2 - torch.abs(bbox1[..., 1] - bbox2[..., 1])).clamp(min=0)

        area_intersect = w*h
        area_bbox = bbox1[..., 0]*bbox1[..., 1]
        area_anchor = bbox2[..., 0]*bbox2[..., 1]
        # print(area_intersect.size(), area_bbox.size(), area_anchor.size())
        return area_intersect/(area_bbox + area_anchor - area_intersect)

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
    print(int(1.-1e-16))
#%% test
if __name__ == "__main__":
    data = data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)
    print(data.__len__())

    #%%
    dataloader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle = True)
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Feature batch shape: {train_features[0].size()}")
    # print(f"Feature batch shape: {train_features[0].squeeze().size()}")
    print(f"Labels batch shape: {train_labels[0].size()}")
    print(f"Labels batch output: {train_labels[0][0,:,:,:,:].size()}")
    print(f"Labels batch output: {train_labels[1][0,:,:,:,:].size()}")
    print(f"Labels batch output: {train_labels[2][0,:,:,:,:].size()}")
    print(f"Labels batch output: {train_labels[2][0,0,0,0,:]}")
    # print(f"Labels batch output: {train_labels[2][0][0][0][0]}")


    img = train_features[0]
    # img = train_features[0].squeeze()
    label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    plt.imshow(img)
    plt.show()

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
    # img, output = data.__getitem__(16, img_resize=True) #w = 640, h = 480
    img, output = data.__getitem__(22, img_resize=True) #height = 427, width = 640
    # img, output = data.__getitem__(1680)
    # img, output = data.__getitem__(160)
    # img, output = data.__getitem__(54) #height = 427, width = 640
    img, output = data.__getitem__(999) #height = 429, width = 500
    # plt.imshow(img)

    # print(f'img.size() = {img.size()}')


    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(30,30))
    ax.imshow(img)
    # trainerval.plot_imgs(8, font_size = 3, axis_size = 2, shuffle = True, dpi = 224*8)

    print(output[0].size())
    print(output[0].size()[1])
    print(output[0].size()[2])
    print(output[1].size())
    print(output[2].size())

    #bounding boxes output is xmin, ymin, w, h - where x, y are the top left corner of the image (for the coco format)
    #YOLO format, x_center, y_center, w, h
    def convert_ij_to_xy(bbox, i, j, width, height, scale_size = 13):
        xmin = (bbox[0] + i) * width / scale_size
        ymin = (bbox[1] + j) * height / scale_size
        w = bbox[2] * width / scale_size
        h = bbox[3] * height / scale_size

        return torch.tensor([xmin, ymin, w, h], dtype=torch.float32)

    def plot_boxes(coord, fig_name):
        x = coord[0] - 0.5*coord[2]
        y = coord[1] - 0.5*coord[3]

        rect = patches.Rectangle(xy = (x, y), width = coord[2], height = coord[3], linewidth=3, linestyle = '--', edgecolor='r', facecolor='none')
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
                        bbox0 = convert_ij_to_xy(output[k,i,j,1:5], i, j, 500, 429)
                        # bbox0 = convert_ij_to_xy(output[k,i,j,1:5], i, j, 640, 427)
                        print(f'i={i}, j={j}, anchor_idx={k}, bbox={output[k,i,j,1:5]}, bbox0={bbox0}')
                        # print(f'i={i}, j={j}, anchor_idx={k}, label={data.labels[int(output[k,i,j,5])]}, bbox={output[k,i,j,1:5]}, bbox0={bbox0}')

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

    #%%
    print(ANCHOR_BOXES.size())
    scale0 = ANCHOR_BOXES[0]
    print(scale0.size())



# %%
