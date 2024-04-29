import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
import random
import SimpleITK as sitk
import torch.optim as optim
import torch.nn as nn
import timeit
from tqdm import tqdm
from torch.autograd import Variable
import gc
from tensorboardX import SummaryWriter
import vggmodel.densenet as densenet
import random


class imageDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.imagelist, self.labellist = self.load_image()
        self.info = list(zip(self.imagelist, self.labellist))


    def __getitem__(self, item):
        path, label = self.info[item]
        img = self.read_data(path)
        return img, label

    def __len__(self):
        return len(self.info)

    def load_image(self):
        image_path = pathlib.Path(self.path)
        all_image_paths = list(image_path.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)
        label_names = sorted(item.name for item in image_path.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        return all_image_paths, all_image_labels

    def read_data(self, path):
        image = sitk.ReadImage(path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = image_array[np.newaxis, :]
        return image_array


def train(depth):
    data_path = r'F:\all_users\gaohy\data\ddm\training_sets\train_120_enhanced'

    dataset = imageDataset(data_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_data, batch_size=7, shuffle=True)
    validation_loader = DataLoader(dataset=val_data, batch_size=7, shuffle=True)
    trainval_loaders = {'train': train_loader, 'val': validation_loader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    log_dir = 'F:/all_users/gaohy/data/ddm/newresult/eventsout/DenseNet'
    writer = SummaryWriter(log_dir=log_dir)

    net = densenet.generate_model(depth)
    net = net.double()
    print(net)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device being using', device)
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
    #
    #                                     gamma=0.1)
    gc.collect()
    torch.cuda.empty_cache()
    nEpochs = 100  # train epochs
    snapshot = 10  # store model every snapshot epochs
    is_val = 0
    val_acc = 0
    for epoch in range(100):

        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                net.train()
            else:
                net.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                # inputs = Variable(inputs, requires_grad=True).to(device, dtype = torch.float)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = net(inputs)
                else:
                    with torch.no_grad():
                        outputs = net(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step() is to be called once every epoch during training
                    # scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]


            if phase == 'train':
                is_val = 0
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                is_val = 1
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print(" [{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            # if epoch % snapshot == (snapshot - 1):
            if is_val == 1 and epoch_acc > val_acc:
                val_acc = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join('F:/all_users/gaohy/data/ddm/newresult/modelsave/Densenet/', 'all_aug' + str(depth)  + '_epoch-' + str(epoch) + '_val_loss-' + '%.3f'%(epoch_loss) + '_val_acc-' + '%.3f'%(epoch_acc) + '.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join('F:/all_users/gaohy/data/ddm/newresult/modelsave/Densenet/','all_aug' + str(depth)  + '_epoch-' + str(epoch) + '_val_loss-' + '%.3f'%(epoch_loss) + '_val_acc-' + '%.3f'%(epoch_acc) + '.pth.tar')))

    print('Finished Training')

if __name__ == '__main__':
    depths = [121]
    for depth in depths:
        train(depth)
    print('finish')