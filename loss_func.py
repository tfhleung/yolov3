#%%
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        anchors = anchors.reshape(1, 3, 1, 1, 2) # reshaping for broadcasting 
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

class YOLO_LOSS(nn.Module):
    def __init__(self, anchors,
                  lambda_box = 10., lambda_cls = 1., lambda_obj = 1.,
                  sigmoid = nn.Sigmoid(), box = nn.MSELoss(), cls = nn.BCEWithLogitsLoss(reduction="mean"), obj = nn.BCEWithLogitsLoss(reduction='sum')):
        super().__init__()
        self.anchors = anchors

        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj

        self.sigmoid = sigmoid
        self.box = box
        self.cls = cls
        self.obj = obj

    def forward(self, preds, target):
        obj = target[..., 1] == 1
        noobj = target[..., 0] == 0
        # anchors = self.anchors.reshape(1, 3, 1, 1, 2)
        anchors = self.anchors.reshape(3, 1, 1, 2)

        #box loss
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        target[..., 3:5] = torch.log(target[..., 3:5]/anchors + 1.e-15)
        # print(f'target = {target}')
        box_loss = self.box(preds[..., 1:5][obj], target[..., 1:5][obj])
        # print(f'box_loss = {box_loss}')

        #object loss
        box_preds = torch.cat([self.sigmoid(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1) #convert preds to ground-truth format (offsets)
        ious = self.iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # obj_loss = self.obj(self.sigmoid(preds[..., 0:1][obj]), ious*target[..., 0:1][obj])
        obj_loss = bce(sigmoid(preds[..., 0][obj]), ious*target[..., 0][obj])

        #class loss
        cls_loss = self.cls(preds[..., 5:][obj], target[..., 5:][obj])
        print(f'box_loss = {cls_loss}')

        total_loss = (
            self.lambda_box*box_loss + 
            self.lambda_obj*obj_loss + 
            self.lambda_cls*cls_loss
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
    loss_func = YOLO_LOSS(ANCHOR_BOXES[0])

    preds = torch.ones(3,13,13,5)
    target = torch.ones(3,13,13,5)
    print(target.size())
    print(loss_func(preds,target))

#%%
    loss_func2 = YoloLoss()
    print(loss_func2(preds,target,ANCHOR_BOXES[1]))

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
