#%%
import torch
import torch.nn as nn

class YOLO_LOSSV3(nn.Module):
    def __init__(self, anchors, scales = [13, 26, 52],
                  lambda_box = 10., lambda_cls = 1., lambda_obj = 1.,
                  sigmoid = nn.Sigmoid(), box = nn.MSELoss(), cls = nn.BCEWithLogitsLoss(reduction="mean"), obj = nn.BCEWithLogitsLoss(reduction='sum')):
        super().__init__()
        self.anchors = anchors
        self.scales = scales

        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj

        self.sigmoid = sigmoid
        self.box = box
        self.cls = cls
        self.obj = obj

        self.mse = nn.MSELoss()


    def forward(self, preds, target):
        anchors = [scale.reshape(3,1,1,2) for scale in ANCHOR_BOXES]
        total_loss, box_loss, obj_loss, cls_loss = [[0.]*3 for _ in range(4)]

        for i in range(len(preds)):
            obj = target[i][..., 1] == 1
            noobj = target[i][..., 0] == 0

            #box loss
            preds[i][..., 1:3] = self.sigmoid(preds[i][..., 1:3])
            target[i][..., 3:5] = torch.log(target[i][..., 3:5]/anchors[i] + 1.e-15)
            box_loss[i] = self.mse(preds[i][..., 1:5][obj], target[i][..., 1:5][obj])

            #object loss
            box_preds = torch.cat([self.sigmoid(preds[i][..., 1:3]), torch.exp(preds[i][..., 3:5]) * anchors[i]], dim=-1) #convert preds to ground-truth format (offsets)
            # print(f'box_preds[i][obj].size() = {box_preds[obj].size()}')
            # print(f'target[i][..., 1:5][obj] = {target[i][..., 1:5][obj]}')
            ious = self.iou(box_preds[obj], target[i][..., 1:5][obj]).detach()
            obj_loss[i] = self.obj(self.sigmoid(preds[i][..., 0][obj]), ious*target[i][..., 0][obj])
            # obj_loss[i] = self.obj(self.sigmoid(preds[i][..., 0:1][obj]), ious*target[i][..., 0:1][obj])
            # obj_loss[i] = bce(sigmoid(preds[i][..., 0][obj]), ious*target[i][..., 0][obj])

            #class loss
            cls_loss[i] = self.cls(preds[i][..., 5:][obj], target[i][..., 5:][obj])
        # print(f'preds[..., 5:][obj] = {preds[..., 5:][obj]}')
        # print(f'target[..., 5:][obj] = { target[..., 5:][obj]}')
        # print(f'box_loss = {box_loss}')
        # print(f'cls_loss = {cls_loss}')

            total_loss[i] = (
                self.lambda_box*box_loss[i] + 
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
#%%
    i = 0
    S = [13, 26, 52]
    loss_func = YOLO_LOSSV3(ANCHOR_BOXES[i])

    preds = torch.rand(3,S[i],S[i],25)
    target = torch.rand(3,S[i],S[i],25)
    print(target.size())
    print(loss_func(preds,target))

#%%
    ANCHOR_BOXES = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ])

    S = [13, 26, 52]
    preds = [torch.rand(3, S[i], S[i], 25) for i in range(3)]
    target = [torch.rand(3, S[i], S[i], 25) for i in range(3)]

    anchors = [scale.reshape(3,1,1,2) for scale in ANCHOR_BOXES]
    print(anchors[0].size())
    target[0][..., 3:5] = torch.log(target[0][..., 3:5]/anchors[0] + 1.e-15)
    target[1][..., 3:5] = torch.log(target[1][..., 3:5]/anchors[1] + 1.e-15)
    target[2][..., 3:5] = torch.log(target[2][..., 3:5]/anchors[2] + 1.e-15)

    # print(f'len = {len(target)}')
    # for i in range(len(preds)):
    #     preds[i][..., 1:3] = self.sigmoid(preds[i][..., 1:3])
    #     target[i][..., 3:5] = torch.log(target[i][..., 3:5]/anchors[i] + 1.e-15)

    loss_func = YOLO_LOSSV3(ANCHOR_BOXES)
    print(loss_func(preds,target))
#%%
    box_loss, obj_loss, cls_loss = [[0.]*3 for _ in range(3)]
    print(box_loss)

#%%
    def iou(bbox1, bbox2):
        overlap_w = (bbox1[..., 2]/2 + bbox2[..., 2]/2 - torch.abs(bbox1[..., 0] - bbox2[..., 0])).clamp(min=0)
        overlap_h = (bbox1[..., 3]/2 + bbox2[..., 3]/2 - torch.abs(bbox1[..., 1] - bbox2[..., 1])).clamp(min=0)

        inter_area = overlap_w * overlap_h
        area1 = bbox1[..., 2] * bbox1[..., 3]
        area2 = bbox2[..., 2] * bbox2[..., 3]

        return inter_area / (area1 + area2 - inter_area + 1e-7)


    preds = torch.rand(3,13,13,5)
    target = torch.ones(3,13,13,5)
    sigmoid = nn.Sigmoid()
    obj = target[..., 1] == 1
    bce = nn.BCEWithLogitsLoss(reduction='sum')

    anchors = ANCHOR_BOXES[0].reshape(3, 1, 1, 2)
    # print(ANCHOR_BOXES[0].size())
    # print(anchors.size())

    # print(sigmoid(preds[..., 1:3]).size())
    # print(torch.exp(preds[..., 3:5]).size())
    # print((torch.exp(preds[..., 3:5]) * anchors).size())

    box_preds = torch.cat([sigmoid(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1)
    print((box_preds[obj]).size())
    ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()

    # print((sigmoid(preds[..., 0:1][obj])).size())
    # print((target[..., 0:1][obj]).size())
    print((ious).size())
    # print(ious)
    # print(obj.size())
    print(sigmoid(preds[..., 0][obj]).size())
    print((target[..., 0][obj]).size())
    print((ious*target[..., 0][obj]).size())
    # print(target[..., 0:1][obj])


    obj_loss = bce(sigmoid(preds[..., 0][obj]), ious*target[..., 0][obj])
    # obj_loss = bce(sigmoid(preds[..., 0:1][obj]), target[..., 0:1][obj])



#%%
