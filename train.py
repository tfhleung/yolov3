#%%
import torch
import torch.nn as nn
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from data import data_COCO
from yolo import YOLO
from loss_func import YOLO_LOSS


#%%
class Trainer():
    def __init__(self, dataset, model, epochs = 5, lr = 0.001, momentum = 0.9, device = 0, batch_size = 8, num_workers = 0, shuffle = True, loss_func = torch.nn.CrossEntropyLoss(reduction = 'sum')):
        self.dataset = dataset
        self.device = device
        self.model = model.to(device)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataloader = {'train': torch.utils.data.DataLoader(self.dataset['train'], self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers),
                        'val': torch.utils.data.DataLoader(self.dataset['val'], self.batch_size, shuffle = False, num_workers = self.num_workers),
                        'test': torch.utils.data.DataLoader(self.dataset['test'], self.batch_size, shuffle = False, num_workers = self.num_workers)}

        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

        self.loss_func = loss_func
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

        self.accuracy = {'train': [], 'val': [], 'test': []}

        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters = {total_params}, Number of Trainable Parameters = {total_trainable_params}")

    def train(self):
        running_loss = 0.0
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for i, data in enumerate(tqdm(self.dataloader['train'])):
                imgs, output = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                self.model.train(True)

                preds = self.model(imgs)
                loss = self.loss_func(preds[0], output[0]) + self.loss_func(preds[1], output[1]) + self.loss_func(preds[2], output[2])
                loss.backward()
                self.optimizer.step()

                # print statistics
                # running_loss += loss.item()  #Returns the value of this tensor as a standard Python number
                # total_loss += loss.item()

                # if i % (int(len(self.dataloader['train'])/6)) == (int(len(self.dataloader['train'])/6)-1):  # print every 2000 mini-batches (mini-batch is the number of data points used to compute one Newton step)
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / int(len(self.dataloader['train'])/6):.3f}')  # loss averaged over 2000 iterations
                #     running_loss = 0.0

            # self.accuracy['val'].append(self.compute_accuracy('val'))
            # self.accuracy['train'].append(self.compute_accuracy('train'))
            # print(f'Epoch {epoch + 1}/{self.epochs} complete. train_accuracy = {self.accuracy['train'][-1][0]:2.3f}%, val_accuracy = {self.accuracy['val'][-1][0]:2.3f}%, train_loss = {self.accuracy['train'][-1][1]:.3f}, val_loss = {self.accuracy['val'][-1][1]:.3f}\n')

        print('Finished Training')
        print("--- %.3f seconds ---" % (time.time() - start_time))
        print("--- %.3f minutes ---" % ((time.time() - start_time)/60.))
        print("--- %.3f hours ---" % ((time.time() - start_time)/3600.))

    def __call__(self, *args, **kwds):
        self.train()

    def compute_accuracy(self, set):
        correct = 0
        total = 0
        total_loss = 0.
        self.model.train(False) #disables dropout layers and clears normalizing batch stats

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.dataloader[set])):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)

                output = self.model(imgs)
                _, index = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (index == labels).sum().item()

                loss = self.loss_func(output, labels)
                total_loss += loss.item()  # Returns the value of this tensor as a standard Python number

        print(f'Dataset: {set}, Num of Images = {total}, Num of Iterations = {i+1}')
        print(f'Correct = {correct}, Total = {total}, Accuracy = {100. * correct/total:.3f}%')
        return (100. * correct / total), total_loss / (i+1.)

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
if __name__ == '__main__':
    ANCHOR_BOXES = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ])
    
    data = {'train': data_COCO(datadir = './coco/', datatype = 'train2017', anchors = ANCHOR_BOXES),
            'val': data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES),
            'test': data_COCO(datadir = './coco/', datatype = 'val2017', anchors = ANCHOR_BOXES)}

    model = YOLO(in_channels = 3, num_classes = 20)

    yolo = Trainer(data, model)
    yolo.train()
# %%
