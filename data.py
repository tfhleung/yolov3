#%%
import torch
import torch.nn as nn

from coco.cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

#%%
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='./coco/'
dataType='val2017'
# dataType='train2017'

# initialize COCO api for instance annotations
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

#get all classes
cats = coco.loadCats(coco.getCatIds())
#filter the labels ids based on label names
catIds = coco.getCatIds(cats)
#filter the imgs based on classes
imgIds = coco.getImgIds(catIds = catIds)

labels = {}
for i, cat in enumerate(cats):
    labels[cat['id']] = cat['name']

#%%
img = coco.loadImgs(imgIds[16])[0]
I = io.imread(f'{dataDir}/images/{dataType}/{img['file_name']}')

# print(img)
# print(I)

plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns) #shows bounding boxes

# print(anns[0]['area'])
print(img.keys())

print(anns[0].keys())
print(f'num_of_imgs = {len(imgIds)}')
print(f'num_of_objects = {len(anns)}')
print(f'height = {img['height']}, width = {img['width']}')
for obj in anns:
    # print(f'image_id = {obj['image_id']},', f'category_id = {obj['category_id']},', f'label_name = {labels[obj['category_id']]},', f'bbox = {obj['bbox']},', f'id = {obj['id']}')
    bbox = obj['bbox'] #[x,y,width,height]
    bbox_normalized = [bbox[0]/img['width'], bbox[1]/img['height'], bbox[2]/img['width'], bbox[3]/img['height']]

    print(f'category_id = {obj['category_id']},', f'label_name = {labels[obj['category_id']]},', f'bbox = {obj['bbox']}', f'bbox_normalized = {bbox_normalized}')

#%%
# You can format your JSON document using Shift+Alt+F or Format Document from the context menu.
#person_keypoints_val2017
#person_keypoints_train2017

#each tuple corresponds to the width/height of an anchor box relative to the image size
#we need to determine the anchor box the has the highest IOU with the bounding box of the image
ABOX = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

class data_COCO():
    def __init__(self, datadir, datatype):
        self.datadir = datadir
        self.datatype = datatype

        self.coco = COCO(f'{datadir}/annotations/instances_{datatype}.json')
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds(self.cats)
        self.imgIds = self.coco.getImgIds(catIds = self.catIds)
   
        self.labels = {}
        for cat in self.cats:
            self.labels[cat['id']] = cat['name']

        self.abox = torch.tensor(ABOX[0])
        # self.abox = torch.tensor(ABOX[0] + ABOX[1] + ABOX[2])

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        I = io.imread(f'{self.datadir}/images/{self.datatype}/{img['file_name']}')

        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)

        labels = []
        bbox = []
        for obj in anns:
            labels.append(obj['category_id'])
            bbox.append(obj['bbox'])

        #image needs to be of type float32 and label needs to be torch long (the label is an index, not name)
        return I, torch.tensor(labels, dtype=torch.long), torch.tensor(bbox)
    
    # calculate iou between abox and bbox where iou is defined as: area of overlap / area of union
    def _iou(self):
        pass

#%%  
data = data_COCO(datadir = './coco/', datatype = 'val2017')
print(data.__len__())
print(data.abox)

#%%
img, labels, bbox = data.__getitem__(16)

# print(data.__getitem__(32))  
print(labels)
print(bbox)
print(bbox)

#%%
if __name__ == 'main':
    data = data_COCO(datadir = './coco/', datatype = 'val2017')
    print(coco.__len__())
    print('test')
# %%
