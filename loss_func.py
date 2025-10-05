#%%
import torch
import torch.nn as nn

class YOLO_LOSSV3(nn.Module):
    def __init__(self, anchor_boxes,
                  lambda_box = 10., lambda_cls = 1., lambda_noobj = 10., lambda_obj = 1.,
                  sigmoid = nn.Sigmoid()):
        super().__init__()
        # self.anchor_boxes = anchor_boxes
        # self.anchors = [scale.reshape(3,1,1,2) for scale in self.anchor_boxes]
        # for i, scale in enumerate(anchor_boxes):
        #     self.register_buffer(f'anchors_{i}', scale.reshape(3, 1, 1, 2))
        # self.anchor_buffers = [getattr(self, f'anchors_{i}') for i in range(len(anchor_boxes))]

        # anchors = [scale.reshape(3,1,1,2) for scale in anchor_boxes]
        # anchors = torch.tensor(anchor_boxes)
        # self.register_buffer(f'anchors', anchors)

        self.register_buffer(f'anchor_boxes', anchor_boxes)

        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj

        self.sigmoid = sigmoid
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    # train_labels is S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor position index, J is anchor position index and O is the label output
    # O is obj, x, y, w, h, class index
    def forward(self, preds, target): # B, C, S, S, O where B is batch size, C is the number of channels, S is the anchor box index and O is the output
        anchors = [scale.reshape(3,1,1,2) for scale in self.anchor_boxes]
        total_loss, box_loss, noobj_loss, obj_loss, cls_loss = [torch.zeros(3) for _ in range(5)]

        for i in range(len(preds)):
            #boolean masks
            obj = target[i][..., 0] == 1
            noobj = target[i][..., 0] == 0

            #box loss
            preds[i][..., 1:3] = self.sigmoid(preds[i][..., 1:3]) # calcuate sigma(x) and sigma(y)
            target[i][..., 3:5] = torch.log(target[i][..., 3:5]/anchors[i] + 1.e-15) # calculate log(w) and log(h)
            box_loss[i] = self.mse(preds[i][..., 1:5][obj], target[i][..., 1:5][obj])

            #no object loss
            noobj_loss[i] = self.bce((preds[i][..., 0][noobj]), (target[i][..., 0][noobj]))

            #object loss
            box_preds = torch.cat([self.sigmoid(preds[i][..., 1:3]), torch.exp(preds[i][..., 3:5]) * anchors[i]], dim=-1) #convert preds to ground-truth format (offsets)
            ious = self.iou(box_preds[obj], target[i][..., 1:5][obj]).detach()
            obj_loss[i] = self.bce(self.sigmoid(preds[i][..., 0][obj]), ious*target[i][..., 0][obj])

            #class loss
            cls_loss[i] = self.ce(preds[i][..., 5:][obj], target[i][..., 5][obj].long())

            total_loss[i] = (
                self.lambda_box*box_loss[i] + 
                self.lambda_noobj*noobj_loss[i] + 
                self.lambda_obj*obj_loss[i] + 
                self.lambda_cls*cls_loss[i]
            )

        return total_loss, box_loss, obj_loss, cls_loss
    
    def iou(self, bbox1, bbox2):
        w = (bbox1[..., 2]/2 + bbox2[..., 2]/2 - torch.abs(bbox1[..., 0] - bbox2[..., 0])).clamp(min=0)
        h = (bbox1[..., 3]/2 + bbox2[..., 3]/2 - torch.abs(bbox1[..., 1] - bbox2[..., 1])).clamp(min=0)

        aint = w*h
        a1 = bbox1[...,2]*bbox1[...,3]
        a2 = bbox2[...,2]*bbox2[...,3]
        return aint/(a1 + a2 - aint)
    
#%%
if __name__ == '__main__':
    ANCHOR_BOXES = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ])

#%%
    S = [13, 26, 52]
    preds = [torch.rand(3, S[i], S[i], 80) for i in range(3)]
    target = [torch.rand(3, S[i], S[i], 80) for i in range(3)]

    anchors = [scale.reshape(3,1,1,2) for scale in ANCHOR_BOXES]
    print(anchors[0].size())
    target[0][..., 3:5] = torch.log(target[0][..., 3:5]/anchors[0] + 1.e-15)
    target[1][..., 3:5] = torch.log(target[1][..., 3:5]/anchors[1] + 1.e-15)
    target[2][..., 3:5] = torch.log(target[2][..., 3:5]/anchors[2] + 1.e-15)

    # print(f'len = {len(target)}')
    # for i in range(len(preds)):
    #     preds[i][..., 1:3] = self.sigmoid(preds[i][..., 1:3])
    #     target[i][..., 3:5] = torch.log(target[i][..., 3:5]/anchors[i] + 1.e-15)

    #%%
    from data import data_COCO
    import matplotlib.pyplot as plt
    from yolo import YOLO
    
    # data = data_COCO(datadir = './coco/', datatype = 'train2017', anchors = ANCHOR_BOXES)
    data = data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)
    print(data.__len__())
    model = YOLO(in_channels = 3, num_classes = 80)

    #%%
    bs = 4
    dataloader = torch.utils.data.DataLoader(data, batch_size = bs, shuffle = False)
    train_features, train_labels = next(iter(dataloader)) 
    # train_feature B, W, L, C, where B is batch_size, W is width, L is length is C is channels
    # train_labels S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor index, J is anchor index and O is the label output

    #%%
    # fig, ax = plt.subplots(1,bs, figsize=(150,150))
    # for i in range(bs):
    #     ax[i].imshow(train_features[i])
    
    #%%
    print(f'num_of_images = {train_features.size(0)}')
    print(f'images = {train_features.size()}')
    print(f'labels = {train_labels[0].size()}')
    # print(f'labels = {train_labels[0]}')

    preds = model(train_features.view(bs,3,416,416).float())
    loss_func = YOLO_LOSSV3(ANCHOR_BOXES)
    print(f'preds[0] = {preds[0].size()}')
    print(f'preds[1] = {preds[1].size()}\n')

    loss, _, _, _ = loss_func(preds, train_labels)
    total_loss = loss[0] + loss[1] + loss[2]
    print(f'loss={loss}, total_loss={total_loss}')

    
    # print(loss_func)
    # obj = train_labels[0][...,0] == 1
    # indices = torch.nonzero(obj)
    # # print(indices)
    # # print(torch.nonzero(obj))
    # print(indices.size())
    # {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
#%%
    loss_func = YOLO_LOSSV3(ANCHOR_BOXES)
    criterion_ce = torch.nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction='sum')

    # img, labels = data['val'].__getitem__(160)
    img, labels = data.__getitem__(55, img_resize=False) #labels format, Obj, X, Y, W, H, C
    print(img.size())
    img_resized, _, _ = data_COCO.letterbox(img.numpy(), target_size = [416, 416])
    print(img_resized.size())

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(img_resized)
    preds = model(img_resized.view(1,3,416,416).float())
    obj = labels[0][...,0] == 1

    print(f'preds.size() = {preds[0].size() }')
    print(f'labels[0].size() = {labels[0].size() }')
    print(f'obj.size() = {obj.size() }')

    print(torch.nonzero(obj))
    # print(preds[0][...,5:][obj])
    print(f'preds[0][...,5:].squeeze()[obj].size() = {preds[0][...,5:].squeeze()[obj].size()}')
    print(f'labels[0][...,5:][obj].size() = {labels[0][...,5:][obj].squeeze().size()}')

    loss_ce = criterion_ce(preds[0][...,5:].squeeze()[obj], labels[0][...,5:].squeeze()[obj].long())
    # loss_ce = criterion_ce(target[0][...,5].long(), target[0][...,5].long())

    # preds[i][..., 1:3] = self.sigmoid(preds[i][..., 1:3]) # calcuate sigma(x) and sigma(y)
    # target[i][..., 3:5] = torch.log(target[i][..., 3:5]/anchors[i] + 1.e-15) # calculate log(w) and log(h)
    box_loss = mse(preds[0][..., 1:5].squeeze()[obj], labels[0][..., 1:5][obj])
    print(f'box_loss = {box_loss}')

    # dummy = torch.rand(1, 3, 416, 416)
    # preds = model(dummy)

#%%
    loss_func = YOLO_LOSSV3(ANCHOR_BOXES)
    criterion_ce = torch.nn.CrossEntropyLoss()

    # img, labels = data['val'].__getitem__(160)
    img, labels = data.__getitem__(55, img_resize=False) #labels format, Obj, X, Y, W, H, C
    print(img.size())
    img_resized, _, _ = data_COCO.letterbox(img.numpy(), target_size = [416, 416])
    print(img_resized.size())

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(img_resized)
    preds = model(img_resized.view(1,3,416,416).float())

    print(f'preds.size() = {preds[0].size() }')
    # print(labels)
    print(f'labels[0].size() = {labels[0].size() }')

    target = [labels[i].reshape(1,3,13*2**i,13*2**i,6) for i in range(3)]
    obj = labels[0][...,0] == 1
    print(torch.nonzero(obj))
    # print(preds[0][...,5:][obj])
    print(labels[0][...,5:][obj])
    print(labels[0].size())


    target0 = torch.zeros(1, 3, 13, 13)
    target0 = target[0][..., 5]
    # for i in range(1):
    #     for j in range(3):
    #         for k in range(13):
    #             for l in range(13):
    #                 target0[i,j,k,l] = target[0][i,j,k,l,5]

    print(f'target0.size() = {target0.size()}')
    print(f'target0.size() = {target0.unsqueeze(0).size()}')


    print(f'labels[0].size() = {target[0].size() }')
    # print(f'labels[1].size() = {target[1].size() }')
    # print(f'labels[2].size() = {target[2].size() }')

    print(f'preds[0][...,5:].size() = {preds[0][...,5:].size() }')
    print(f'preds[0][...,5:].permute().size() = {preds[0].permute(0,4,1,2,3).size() }')
    print(f'target[0][...,5:].size() = {target[0][...,5:].size() }')
    print(f'target[0][...,5:].size() = {target[0][...,5:].squeeze(-1).size() }')

    # print(f'preds[0].size() = {preds[0].size() }')
    # print(f'target[0].size() = {target[0].size() }')

    # print(preds[0][0,2,12,12,:])
    # print(target[0][0,2,12,12,5])
    # print(f'preds[0][3,13,13,5:].size() = {preds[0][1,3,13,13,5].size() }')
    
    loss_ce = criterion_ce(preds[0][0,2,12,12,5:], target[0][0,2,12,12,5].long())
    loss_ce = criterion_ce(preds[0][...,5:].permute(0,4,1,2,3), target[0][...,5:].squeeze(-1).long())
    loss_ce = criterion_ce(preds[0][...,5:].permute(0,4,1,2,3), target0.long())
    loss_ce = criterion_ce(preds[0][...,5:][obj], target[0][...,5:][obj])
    # loss_ce = criterion_ce(target[0][...,5].long(), target[0][...,5].long())

    # dummy = torch.rand(1, 3, 416, 416)
    # preds = model(dummy)

#%%
