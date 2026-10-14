#%%
import torch
import torch.nn as nn
from torch import device, dtype, Tensor

import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from data import data_COCO
from yolo import YOLO
from loss_func import YOLO_LOSSV3


#%%
class Trainer():
    def __init__(self, dataset, model, loss_func, epochs = 5, lr = 0.001, momentum = 0.9, device = 0, batch_size = 8, num_workers = 0, shuffle = True):
        self.dataset = dataset
        self.device = device
        self.model = model.to(device)
        print(f'model is on {next(self.model.parameters()).device}')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataloader = {'train': torch.utils.data.DataLoader(self.dataset['train'], self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers),
                        'val': torch.utils.data.DataLoader(self.dataset['val'], self.batch_size, shuffle = False, num_workers = self.num_workers),
                        'test': torch.utils.data.DataLoader(self.dataset['test'], self.batch_size, shuffle = False, num_workers = self.num_workers)}

        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

        self.loss_func = loss_func.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

        self.accuracy = {'train': [], 'val': [], 'test': []}
        self.accuracyperclass = {'train': [], 'val': [], 'test': []}

        torch.cuda.reset_peak_memory_stats(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters = {total_params}, Number of Trainable Parameters = {total_trainable_params}")
        print(f'Memory allocated={torch.cuda.memory_allocated()/ (1024**2):.2f} MB')
        if torch.cuda.is_available():
            max_allocated_memory = torch.cuda.max_memory_allocated()
            print(f"Peak allocated GPU memory={max_allocated_memory / (1024**2):.2f} MB")
        else:
            print("CUDA is not available.")

        for name, param in self.model.named_parameters():
            print(f"{name}, device = {param.device}")
            

    def train(self):
        runningloss = 0.0
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(self.epochs):
            totalrunningloss = 0.0
            for i, data in enumerate(tqdm(self.dataloader['train'])):
                imgs, train_labels = data[0].to(self.device), [d.to(self.device) for d in data[1]]

                self.optimizer.zero_grad()
                self.model.train(True)
                preds = self.model(imgs.view(imgs.size(0),3,416,416).float())

                # preds = [p.to(self.device) for p in preds]
                # train_labels = [t.to(self.device) for t in train_labels]

                loss, box_loss, obj_loss, cls_loss = self.loss_func(preds, train_labels)
                total_loss = loss[0] + loss[1] + loss[2]

                total_loss.backward()
                self.optimizer.step()

                # print statistics
                runningloss += total_loss.item()  #Returns the value of this tensor as a standard Python number
                totalrunningloss += total_loss.item()

                if i % (int(len(self.dataloader['train'])/6)) == (int(len(self.dataloader['train'])/6)-1):  # print every 2000 mini-batches (mini-batch is the number of data points used to compute one Newton step)
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningloss / int(len(self.dataloader['train'])/6):.3f}')  # loss averaged over 2000 iterations
                    runningloss = 0.0

            torch.cuda.reset_peak_memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(0)
            print(f"Model Usage = {torch.cuda.memory_allocated(self.device)/1024**3:2.2f} GB, Free Memory = {free/1024**3:2.2f} GB, Total Memory = {total/1024**3:2.2f} GB.")

            self.accuracy['val'].append(self.compute_class_accuracy('val'))
            self.accuracy['train'].append(self.compute_class_accuracy('train'))
            # self.accuracyperclass['train'].append(self.compute_perclass_accuracy('train'))
            # self.accuracyperclass['val'].append(self.compute_perclass_accuracy('val'))
            print(f'Epoch {epoch + 1}/{self.epochs} complete. train_accuracy = {self.accuracy['train'][-1][0]:2.3f}%, val_accuracy = {self.accuracy['val'][-1][0]:2.3f}%, train_loss = {self.accuracy['train'][-1][1]:.3f}, val_loss = {self.accuracy['val'][-1][1]:.3f}\n')

        print('Finished Training')
        print("--- %.3f seconds ---" % (time.time() - start_time))
        print("--- %.3f minutes ---" % ((time.time() - start_time)/60.))
        print("--- %.3f hours ---" % ((time.time() - start_time)/3600.))

    def __call__(self, *args, **kwds):
        self.train()

    def compute_class_accuracy(self, set: str):
        correct, total = 0, 0
        totalrunningloss = 0.
        self.model.train(False) #disables dropout layers and clears normalizing batch stats

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.dataloader[set])):
                imgs, train_labels = data[0].to(self.device), [d.to(self.device) for d in data[1]]
                output = self.model(imgs.view(imgs.size(0),3,416,416).float())
        
                for scaleidx in range(len(output)):
                    obj = train_labels[scaleidx][..., 0] == 1
                    # preds is S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor position index, J is anchor position index and O is the label output

                    max_val, max_idx = torch.max(output[scaleidx][...,5:][obj], dim=1)
                    correct += ( max_idx == train_labels[scaleidx][...,5][obj] ).sum().item()
                    total += int(train_labels[scaleidx][...,5][obj][0])

                loss, _, _, _ = self.loss_func(output, train_labels)
                totalrunningloss += sum(loss).item()  # Returns the value of this tensor as a standard Python number

        print(f'Dataset: {set}, Num of Images = {total}, Num of Iterations = {i+1}')
        print(f'Correct = {correct}, Total = {total}, Accuracy = {100. * correct/total:.3f}%')
        return (100. * correct / total), totalrunningloss / (i+1.)
    
    def compute_perclass_accuracy(self, set: str) -> tuple[float]:
        correct, total = [torch.zeros(self.dataset['val'].num_classes) for _ in range(2)]
        totalrunningloss = 0.
        self.model.train(False) #disables dropout layers and clears normalizing batch stats

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.dataloader[set])):
                imgs, train_labels = data[0].to(self.device), [d.to(self.device) for d in data[1]]
                output = self.model(imgs.view(imgs.size(0),3,416,416).float())
        
                for scaleidx in range(len(output)):
                    obj = train_labels[scaleidx][..., 0] == 1
                    # preds is S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor position index, J is anchor position index and O is the label output
                    # O is obj, x, y, w, h, class index

                    max_val, max_idx = torch.max(output[scaleidx][...,5:][obj], dim=1)

                    for cls in range(self.dataset['val'].num_classes):
                        clsmax_idx = train_labels[scaleidx][...,5][obj] == cls
                        correct[cls] += int(( max_idx[clsmax_idx] == train_labels[scaleidx][...,5][obj][clsmax_idx] ).sum().item())
                        total[cls] += int(len(train_labels[scaleidx][...,5][obj][clsmax_idx]))

                loss, _, _, _ = self.loss_func(output, train_labels)
                totalrunningloss += sum(loss).item()  # Returns the value of this tensor as a standard Python number

        print(f'Dataset: {set}, Num of Images = {int(torch.sum(total))}, Num of Iterations = {i+1}')

        for cls in range(self.dataset['val'].num_classes):
            print(f'Class={self.dataset['val'].label(cls)}, Correct={int(correct[cls])}, Total={int(total[cls])}, Accuracy={100. * correct[cls]/total[cls]:.3f}%')

        return correct, total

    def compute_mAP(self, bbox_preds, bbox_gt):
        """
        Calculates bbox based on YOLO model predictions.
        precision = (true positive)/(true positive + false positive)
        recall = (true positive)/(true positive + false negative)
        
        Args: 
            bbox_preds (list[Tensor]): [img_idx, M, x, y, w, h, obj_confidence, class_idx]
            bbox_gt (list[Tensor]): [img_idx, N, x, y, w, h, obj_confidence, class_idx]
        Return:
            bbox (list): list of tensors corresponding to bounding boxes and class index (xmid, ymid, w, h, class_idx)
        """

        mAP = 0.
        return mAP
    
    def get_bboxes(self, dataloader) -> tuple[tuple[Tensor], Tensor]:
        """
        Get the bbox
        
        Args: 
            dataloader (list[Tensor]): list of three tensors with sizes ([1, 3, 13, 13, num_classes+5])
                                                    ([1, 3, 26, 26, num_classes+5])
                                                    ([1, 3, 52, 52, num_classes+5])
            preds (tensor): preds is [S, BS, A, I, J, O], where O is the label output [obj, x, y, w, h, class probabilites for all classes]
            labels (tensor): train_labels is [S, BS, A, I, J, O] where:
                S is the scale index,
                BS is batch_size,
                A is anchor index (three sizes per scale),
                I is anchor position index x-position,
                J is anchor position index y-position,
                O is the label output [obj, x, y, w, h, class index]
                                                    
        Return:
            bboxes (tuple[tuple[Tensor], Tensor]): tuple of tensors corresponding to bounding boxes and class index (xmid, ymid, w, h, class_idx)

        """

        bbox_preds, bbox_preds = [], []

        self.model.train(False) #disables dropout layers and clears normalizing batch stats
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                imgs, train_labels = data[0].to(self.device), [d.to(self.device) for d in data[1]]
                preds = self.model(imgs.view(imgs.size(0),3,416,416).float()) # [3][BS, ANCHOR_BOX, scale_idx, scale_idx, num_classes+5]

                bbox_gt = self.dataset['val'].labels_to_bbox(train_labels) # [N][5]: xmid, ymid, w, h, class_index
                bbox_preds = self.preds_to_bboxes(preds)# [3][M][5]: xmid, ymid, w, h, class_index

                # for pred in preds:
                for scaleidx in range(len(preds)):
                    pass

        return tuple(bbox_preds, bbox_gt)

    def compute_mAPold(self, set: str):
        """
        Calculates bbox based on YOLO model predictions.
        precision = (true positive)/(true positive + false positive)
        recall = (true positive)/(true positive + false negative)
        
        Args: 
            preds (tensor): preds is [S, BS, A, I, J, O], where O is the label output [obj, x, y, w, h, class probabilites for all classes]
            preds (list[Tensor]): list of three tensors with sizes ([1, 3, 13, 13, num_classes+5])
                                                    ([1, 3, 26, 26, num_classes+5])
                                                    ([1, 3, 52, 52, num_classes+5])
        Return:
            bbox (list): list of tensors corresponding to bounding boxes and class index (xmid, ymid, w, h, class_idx)
        """

        self.model.train(False) #disables dropout layers and clears normalizing batch stats
        with torch.no_grad():
            precision, TP, FP = [torch.zeros(self.dataset['val'].num_classes) for _ in range(3)]

            for i, data in enumerate(tqdm(self.dataloader[set])):
                imgs, train_labels = data[0].to(self.device), [d.to(self.device) for d in data[1]]
                preds = self.model(imgs.view(imgs.size(0),3,416,416).float()) # [3][BS, ANCHOR_BOX, scale_idx, scale_idx, num_classes+5]

                bbox_gt = self.dataset['val'].labels_to_bbox(train_labels) # [N][5]: xmid, ymid, w, h, class_index
                bbox_preds = self.preds_to_bboxes(preds)# [3][M][5]: xmid, ymid, w, h, class_index

                # for pred in preds:
                for scaleidx in range(len(preds)):

                    #sort bbox_preds based on object confidence
                    bbox_preds = bbox_preds[scaleidx][[...,0].argsort(descending=True)]

                    #calculate the precision/recall for each class
                    for cls in range(self.dataset['val'].num_classes):
                        #filter bboxes based on class
                        obj_pred = bbox_preds[scaleidx][...,4] == cls
                        obj_gt = bbox_gt[...,4] == cls
                        ious = data_COCO.compute_iou(bbox_preds[scaleidx][..., 0:4][obj_pred], bbox_gt[..., 0:4][obj_gt]) #[M,N], where each row corr

                        gt_check = torch.zeros(len(bbox_gt[..., 0:4][obj_gt]), dtype=torch.bool)

                        #loop though predicted bboxs
                        for iou in ious:
                            val, idx = iou.max(0)

                            if val > self.dataset[set].iou_thresh and gt_check[idx] is False:
                                TP[cls] += 1
                                gt_check[idx] = True
                            else:
                                FP[cls] += 1

            precision = TP/(TP + FP)
            # recall = TP/(TP + FP)
            return precision


    def preds_to_bboxes(self, preds: tuple[Tensor]) -> list[Tensor]:
        """
        Calculates bbox based on YOLO model predictions.  The class index is determined based on the class with the highest confidence score.

        Args: 
            preds (tensor): preds is [S, BS, A, I, J, O], where O is the label output [obj, x, y, w, h, class probabilites for all classes]
            preds (tuple[tensor]): tuple of tensors for each scale with sizes ([1, 3, 13, 13, num_classes+5])
                                                    ([1, 3, 26, 26, num_classes+5])
                                                    ([1, 3, 52, 52, num_classes+5])
        Return:
            bbox (list): list of tensors corresponding to bounding boxes and class index (xmid, ymid, w, h, obj_confidence, class_idx) for all scales
        """
        bbox = []
        for scaleidx in range(len(preds)):
            obj = preds[scaleidx][...,0] > self.dataset['val'].iou_thresh
            cell_indices = obj.nonzero()
            bbox.append(torch.zeros((cell_indices.size(0),6)))

            bbox[scaleidx][:,0] = torch.sigmoid(preds[scaleidx][...,1][obj]) + cell_indices[:,1]/self.dataset['val'].scale_size[scaleidx]
            bbox[scaleidx][:,1] = torch.sigmoid(preds[scaleidx][...,2][obj]) + cell_indices[:,2]/self.dataset['val'].scale_size[scaleidx]
            bbox[scaleidx][:,2] = 1./self.dataset['val'].scale_size[scaleidx] * torch.exp(preds[scaleidx][...,3][obj])
            bbox[scaleidx][:,3] = 1./self.dataset['val'].scale_size[scaleidx] * torch.exp(preds[scaleidx][...,4][obj])
            bbox[scaleidx][:,4] = preds[scaleidx][...,0][obj]
            _, bbox[scaleidx][:,5] = torch.max(preds[scaleidx][...,5:][obj], dim=1)
            # max_val, max_idx = torch.max(preds[scaleidx][...,5:][obj], dim=1)

        return bbox


    def plot_bbox(self, input):
        #resize image
        self.model.train(False)
        preds = self.model(input.view(1,3,416,416).float())

        bbox = []
        height, width, channels = img.shape

        for scaleidx in range(3):
            obj = preds[scaleidx][...,0] > self.dataset['train'].iou_thresh
            indices = obj.nonzero()
            bbox.append(torch.zeros((indices.size(0),4)))
            # print(f'bbox[0].shape={bbox[0].shape}')

            bbox[scaleidx][:,0] = (input[scaleidx][...,1][obj] + indices[:,1]) * width / self.scale_size[scaleidx]
            bbox[scaleidx][:,1] = (input[scaleidx][...,2][obj] + indices[:,2]) * height / self.scale_size[scaleidx]
            bbox[scaleidx][:,2] = input[scaleidx][...,3][obj] * width / self.scale_size[scaleidx]
            bbox[scaleidx][:,3] = input[scaleidx][...,4][obj] * height / self.scale_size[scaleidx]

        # bbox = self.convert_ij_to_xy(idx, input)
        # patch = [Rectangle((b[0]-0.5*b[2],b[1]-0.5*b[3]), b[2], b[3]) for b in bbox[0]] #yolo format is (xcenter, ycenter) but matplotlib is (xmin, ymin)
        # collection = PatchCollection(patch, linewidth=3, linestyle='--', edgecolor='b', facecolor='none')

        # fig, ax = plt.subplots(1, figsize=(30,30))
        # ax.add_collection(collection)
        # ax.imshow(img)

    def plot_results(self):
        import numpy as np
        acc_np = {'train': np.array(self.accuracy['train']),
              'val': np.array(self.accuracy['val'])}
        epoch = np.linspace(1, 1+len(acc_np['train']), len(acc_np['train']))

        plt.subplot(1, 2, 1)
        plt.title("Model Accuracy")
        plt.plot(epoch ,acc_np['train'][:,0], label = 'Training Accuracy')
        plt.plot(epoch ,acc_np['val'][:,0], label = 'Validation Accuracy')
        plt.ylim([0.,100.])
        plt.ylabel('Percentage Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.title("Loss")
        plt.plot(epoch ,acc_np['train'][:,1], label = 'Training Loss')
        plt.plot(epoch ,acc_np['val'][:,1], label = 'Validation Loss')
        plt.ylim([0.,10.])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

    def plot_imgs(self, num_imgs, font_size = 6, axis_size = 6, shuffle = False, dpi = 100):
        import torch.nn.functional as F
        newline = '\n'

        dataloader = torch.utils.data.DataLoader(self.dataset['test'], num_imgs, shuffle = shuffle, num_workers = self.num_workers)
        test_imgbatch, test_labelsbatch = next(iter(dataloader))

        outputs = self.model(test_imgbatch.to('cuda'))
        _, predicted = torch.max(outputs, 1)
        probability = F.softmax(outputs, dim=1)
        probability_max, _ = torch.max(probability, dim=1)

        print('Predicted: ', ' '.join(f'{self.dataset['test'].labels[predicted[j]]:5s}' for j in range(num_imgs)))
        print('Predicted: ', ' '.join(f'{probability_max[j]*100.:2.2f}' for j in range(num_imgs)))

        fig = plt.subplots(1,num_imgs)
        plt.rcParams.update({'font.size': axis_size, 'figure.dpi': dpi})
        for i in range(num_imgs):
            plt.subplot(1, num_imgs, i+1)
            plt.imshow(test_imgbatch[i].permute(1, 2, 0)) #image format is channel x width x height
            plt.title(f'Ground Truth = {self.dataset['test'].labels[test_labelsbatch[i]]} {newline}'
                    f'Predicted = {self.dataset['test'].labels[predicted[i]]} {newline}', fontsize=font_size)
                    # f'Probability = {probability_max[i] * 100.:2.2f}%', fontsize=font_size)

        plt.show()

#%%
# def output_to_bbox(preds, dataset):
#     """
#     Input: 
#     preds: tensor of size ([1, 3, 52, 52, num_classes+5])
#     torch.Size([1, 3, 13, 13, num_classes+5])
#     torch.Size([1, 3, 26, 26, num_classes+5])
#     torch.Size([1, 3, 52, 52, num_classes+5])

#     """
#     bbox = []
#     for scaleidx in range(len(preds)):
#         obj = preds[scaleidx][...,0] > dataset.iou_thresh
#         cell_indices = obj.nonzero()
#         bbox.append(torch.zeros((cell_indices.size(0),5)))
#         # print(cell_indices)

#         bbox[scaleidx][:,0] = torch.sigmoid(preds[scaleidx][...,1][obj]) + cell_indices[:,1]/dataset.scale_size[scaleidx]
#         bbox[scaleidx][:,1] = torch.sigmoid(preds[scaleidx][...,2][obj]) + cell_indices[:,2]/dataset.scale_size[scaleidx]
#         bbox[scaleidx][:,2] = 1./dataset.scale_size[scaleidx] * torch.exp(preds[scaleidx][...,3][obj])
#         bbox[scaleidx][:,3] = 1./dataset.scale_size[scaleidx] * torch.exp(preds[scaleidx][...,4][obj])
#         _, bbox[scaleidx][:,4] = torch.max(preds[scaleidx][...,5:][obj], dim=1)

#     return bbox

# ANCHOR_BOXES = torch.tensor([
# [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
# [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
# [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ])
    
# dataset = data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)
# dummy_pred = torch.rand([1, 3, 13, 13, 25])
# # torch.Size([1, 3, 26, 26, 25])
# # torch.Size([1, 3, 52, 52, 25])

# bbox = output_to_bbox(dummy_pred, dataset)
# print(bbox)


#%%
if __name__ == '__main__':
    ANCHOR_BOXES = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ])
    
    data = {'train': data_COCO(datadir = './coco/', datatype = 'train2017', anchors = ANCHOR_BOXES),
            'val': data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES),
            'test': data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)}

#%%
    model = YOLO(in_channels = 3, num_classes = 80)
    yolov3_loss = YOLO_LOSSV3(ANCHOR_BOXES)

#%%
    # yolo = Trainer(data, model, loss_func=YOLO_LOSSV3(ANCHOR_BOXES), device='cuda', batch_size=16, num_workers=4, epochs=1)
    yolo = Trainer(data, model, loss_func=yolov3_loss, device='cuda', epochs=1, batch_size=16, lr=1.e-8, num_workers=8, shuffle=True)

#%%
    # correct, total = yolo.compute_perclass_accuracy('val')

#%%
    yolo()

    #%%
    yolo.compute_mAP(set='val')

#%%
    dataloader = {'train': torch.utils.data.DataLoader(data['train'], batch_size=16, shuffle = True, num_workers = 8),
                        'val': torch.utils.data.DataLoader(data['val'], batch_size=16, shuffle = False, num_workers = 8),
                        'test': torch.utils.data.DataLoader(data['test'], batch_size=16, shuffle = False, num_workers = 8)}


    def 

    def compute_mAP(set: str):
        """
        Calculates bbox based on YOLO model predictions.
        # precision = (true positive)/(true positive + false positive)
        # recall = (true positive)/(true positive + false negative)
        
        Args: 
            preds (list): list of three tensors with sizes ([1, 3, 13, 13, num_classes+5])
                                                    ([1, 3, 26, 26, num_classes+5])
                                                    ([1, 3, 52, 52, num_classes+5])
        Return:
            bbox (list): list of tensors corresponding to bounding boxes and class index (xmid, ymid, w, h, class_idx)
        """

        model.train(False) #disables dropout layers and clears normalizing batch stats
        with torch.no_grad():
            precision, TP, FP = [torch.zeros(data[set].num_classes) for _ in range(3)]
            print('test')

            for i, d in enumerate(tqdm(dataloader['val'])):
                imgs, train_labels = d[0].to('cuda'), [d.to('cuda') for d in d[1]]
                preds = model(imgs.view(imgs.size(0),3,416,416).float()) # [3][BS, ANCHOR_BOX, scale_idx, scale_idx, num_classes+5]

                bbox_gt = data['val'].labels_to_bbox(train_labels) # [BS][N][O], O = [xmid, ymid, w, h, class_index]
                bbox_preds = yolo.preds_to_bboxes(preds)# [BS][3][M][O]: O = [xmid, ymid, w, h, class_index]

                # for pred in preds:
                for scaleidx in range(len(preds)):
                    # print(f'bbox_preds[scaleidx].shape={bbox_preds[scaleidx].shape}')
                    #sort bbox_preds based on object confidence
                    bbox_preds[scaleidx] = bbox_preds[scaleidx][bbox_preds[scaleidx][...,0].argsort(descending=True)]

                    #calculate the precision/recall for each class
                    for cls in range(data['val'].num_classes):
                        #filter bboxes based on class
                        obj_pred = bbox_preds[scaleidx][...,4] == cls
                        obj_gt = bbox_gt[...,4] == cls
                        ious = data_COCO.compute_iou(bbox_preds[scaleidx][..., 0:4][obj_pred], bbox_gt[..., 0:4][obj_gt]) #[M,N], where each row corr

                        gt_check = torch.zeros(len(bbox_gt[..., 0:4][obj_gt]), dtype=torch.bool)

                        #loop though predicted bboxs
                        for iou in ious:
                            val, idx = iou.max(0)

                            if val > data['val'].iou_thresh and gt_check[idx] is False:
                                TP[cls] += 1
                                gt_check[idx] = True
                            else:
                                FP[cls] += 1

            precision = TP/(TP + FP)
            return precision
#%%
    data['val'].iou_thresh = -1.e-6
    print(compute_mAP(set='val'))

#%%
    bs = 1
    dataloader = torch.utils.data.DataLoader(data['val'], batch_size = bs, shuffle = True)
    train_features, train_labels = next(iter(dataloader))
    preds = model(train_features.view(bs,3,416,416).float().to('cuda'))


    bbox_gt = data['val'].labels_to_bbox(train_labels) # [N][5]: xmid, ymid, w, h, class_index
    bbox_preds = yolo.preds_to_bboxes(preds)# [3][M][5]: xmid, ymid, w, h, class_index

    print(bbox_preds[0])
    bbox_preds[0] = bbox_preds[0][bbox_preds[0][...,0].argsort(descending=True)]

#%%
    bs = 1

    dataloader = torch.utils.data.DataLoader(data['val'], batch_size = bs, shuffle = True)
    train_features, train_labels = next(iter(dataloader))
    train_features.to('cuda')
    model.to('cuda')
    print(f'model is on {next(model.parameters()).device}')

    train_labels = [t.to('cuda') for t in train_labels]


    print(f"Feature batch shape: {train_features.size()}")
    print(f"Feature batch shape: {train_features[0].size()}")
    # print(f"Feature batch shape: {train_features[0]}")
    print(f"Labels batch shape: {train_labels[0].size()}")
    print(f"Labels batch output: {train_labels[2][0,0,0,0,:]}")


    img = train_features[0]
    # img = train_features[0].squeeze()
    print(img.size())
    label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    plt.imshow(img)
    plt.show()

    # img_cuda = img.reshape(bs,3,416,416).float()
    # img_cuda.to('cuda')
    # preds = model(img_cuda)

    preds = model(train_features.view(bs,3,416,416).float().to('cuda'))
    print(preds[0].size())
    print(preds[1].size())
    print(preds[2].size(4))

    obj = [label[...,0] == 1 for label in train_labels]

    # preds is S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor position index, J is anchor position index and O is the label output
    scaleidx = 0
    cls = 0
    print(preds[scaleidx][...].size())
    print(preds[scaleidx][...,5:][obj[scaleidx]].size())

    max_val, max_idx = torch.max(preds[scaleidx][...,5:][obj[scaleidx]], dim=1)
    clsmax_idx = train_labels[scaleidx][...,5][obj[scaleidx]] == cls
    print(f'model_obj = {preds[scaleidx][...,0][obj[scaleidx]]}')
    print(f'max_idx = {max_idx}')
    print(f'output size = {max_idx.shape}')

    # label is S, B, A, I, J, O where S is the scale index, B is batch_size, A is anchor index, I is anchor position index, J is anchor position index and O is the label output
    # print(f'labels = {train_labels[scaleidx].size()}')
    print(f'all labels = {train_labels[scaleidx].shape}')
    print(f'label_obj.shape[0] = {train_labels[scaleidx][...,5][obj[scaleidx]].shape[0]}')
    print(f'label_obj = {train_labels[scaleidx][...,0][obj[scaleidx]]}')
    print(f'labels = {train_labels[scaleidx][...,5][obj[scaleidx]]}')
    # print(f'clsmax_idx = {clsmax_idx}')

    correct = ( max_idx == train_labels[scaleidx][...,5][obj[scaleidx]] ).sum().item()
    print(f'correct={correct}')

    for cls in range(80):
        clsmax_idx = train_labels[scaleidx][...,5][obj[scaleidx]] == cls
        correct_class = ( max_idx[clsmax_idx] == train_labels[scaleidx][...,5][obj[scaleidx]][clsmax_idx] ).sum().item()
        # total = ( cls == train_labels[scaleidx][...,5][obj[scaleidx]][clsmax_idx] ).sum().item()
        total = len(train_labels[scaleidx][...,5][obj[scaleidx]][clsmax_idx])
        print(f'class={cls}, correct={correct_class}, total={total}')

    print(f'obj.shape={obj[0].shape}')

    # output = model(train_features.view(bs,3,416,416).float())

    # preds = model(img.reshape(bs,3,416,416).float())

#%%
    # img, labels = data['val'].__getitem__(160)
    img, labels = data['val'].__getitem__(55, img_resize=False)

    yolo.plot_bbox(img)


    # print(img)
    print(img.size())
    # img2 = img.view(3, 375, 500)
    # print(img2.size())
    

    # img2 = torch.tensor(img)
    # print(img2)

    img_resized, _, _ = data_COCO.letterbox(img.numpy(), target_size = [416, 416])
    print(img_resized.size())

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(img_resized)
    preds0 = model(img_resized.view(1,3,416,416).float())
    print(f'preds0.size() = {preds0[0].size() }')

    # img.reshape(3,416,416)
    # print(img.size())

    # dummy = torch.rand(1, 3, 416, 416)
    # preds = model(dummy)


# %%
