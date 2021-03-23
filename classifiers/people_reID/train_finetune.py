# -*- coding: utf-8 -*-
#pip install "pillow<7"
#Adapted from:
#Beyond Part Models: Person Retrieval with Refined Part Pooling and A Strong Convolutional Baseline
#Authors: Yifan Suny and Liang Zhengz and Yi Yangz and Qi Tianx and Shengjin Wang


from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import numpy as np
from people_reID.model.model_finetuning import ft_net, ft_net_dense, ft_net_NAS, PCB
from people_reID.random_erasing import RandomErasing
import yaml
from PIL import Image, ImageDraw
from shutil import copyfile

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


class MyDataset(Dataset):
	def __init__(self,base,dataframe, x_col, y_col,transform=None):
		self.input_images = dataframe[x_col]
		if y_col!=None:
			self.target_images = dataframe[y_col]
		else: 
			self.target_images = []
		self.transform = transform
		self.base = base

	def __getitem__(self, idx):
		if self.base != None:
			image = Image.open(self.base+self.input_images[idx])
		else:
			image = Image.open(self.input_images[idx])

		if len(self.target_images)>0:
			label  = self.target_images[idx]
		else:
			label = 0
		#print(len(image.getbands()))
		#if len(image.getbands()) == 1:
			#print(image.shape)
		image = image.convert("RGB") 
			#print(image.shape)
		
		if self.transform:
			image = self.transform(image)
		#print(image.size())
		#print(label)
		return image,torch.tensor(int(label))

	def __len__(self):
		return len(self.input_images)




def finetune_reID_network(weights_ini,dataframe_train, dataframe_val,x_col,y_col, dir_out_config, gpu_ids):
    ######################################################################
    # Options
    # --------
    
    color_jitter = True
    erasing_p = 0
    opt_lr = 0.05
    batchsize = 3
    fp16 = False

    opt_PCB = True

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids)>0:
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    #

    transform_train_list = [
            #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256,128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    if opt_PCB:
        transform_train_list = [
            transforms.Resize((384,192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        transform_val_list = [
            transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    if erasing_p>0:
        transform_train_list = transform_train_list +  [RandomErasing(probability = erasing_p, mean=[0.0, 0.0, 0.0])]

    if color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose( transform_train_list ),
        'val': transforms.Compose(transform_val_list),
    }


    image_datasets = {}

    image_datasets['train'] = MyDataset(None,dataframe_train,x_col,y_col, transform=data_transforms['train'])
    image_datasets['val'] = MyDataset(None,dataframe_val,x_col,y_col, transform=data_transforms['val'])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True) for x in ['train', 'val']}
    class_names = dataframe_train[y_col]


    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()

    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    print(len(inputs[0]))
    print(time.time()-since)
    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        #best_model_wts = model.state_dict()
        #best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data
                    now_batch_size,c,h,w = inputs.shape
                    if now_batch_size<batchsize: # skip the last batch
                        continue
                    #print(inputs.shape)
                    # wrap them in Variable
                    if use_gpu:
                        #print(inputs)
                        #print(labels)
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # if we use low precision, input also need to be fp16
                    #if fp16:
                    #    inputs = inputs.half()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                    else:
                        #print(inputs.size())
                        outputs = model(inputs)

                    if not opt_PCB:
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                    else:
                        part = {}
                        sm = nn.Softmax(dim=1)
                        num_part = 6
                        for i in range(num_part):
                            part[i] = outputs[i]

                        score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                        _, preds = torch.max(score.data, 1)

                        loss = criterion(part[0], labels)
                        for i in range(num_part-1):
                            loss += criterion(part[i+1], labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if fp16: # we use optimier to backward loss
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                    # statistics
                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0-epoch_acc)
                # deep copy the model
                if phase == 'val':
                    last_model_wts = model.state_dict()
                    if epoch%10 == 9:
                        save_network(model, epoch)
                    draw_curve(epoch)

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(last_model_wts)
        save_network(model, 'last')
        return model


    ######################################################################
    # Draw Curve
    #---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig( os.path.join(dir_out_config,'train_reID.jpg'))

    ######################################################################
    # Save model
    #---------------------------
    def save_network(network, epoch_label):
        save_filename = 'net_%s_reID.pth'% epoch_label
        save_path = os.path.join(dir_out_config,save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])


    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #
    def train(model):
        nclasses = len(class_names)

        print(model)

        if not opt_PCB:
            ignored_params = list(map(id, model.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer_ft = optim.SGD([
                    {'params': base_params, 'lr': 0.1*opt_lr},
                    {'params': model.classifier.parameters(), 'lr': opt_lr}
                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.model.fc.parameters() ))
            ignored_params += (list(map(id, model.classifier0.parameters() ))
                            +list(map(id, model.classifier1.parameters() ))
                            +list(map(id, model.classifier2.parameters() ))
                            +list(map(id, model.classifier3.parameters() ))
                            +list(map(id, model.classifier4.parameters() ))
                            +list(map(id, model.classifier5.parameters() ))
                            #+list(map(id, model.classifier6.parameters() ))
                            #+list(map(id, model.classifier7.parameters() ))
                            )
            #base_params = filter(lambda p: id(p) not in ignored_params)
            optimizer_ft = optim.Adam([
                    #{'params': base_params, 'lr': 0.001},
                    {'params': model.model.fc.parameters(), 'lr': 0.00001},
                    {'params': model.classifier0.parameters(), 'lr': 0.00001},
                    {'params': model.classifier1.parameters(), 'lr': 0.00001},
                    {'params': model.classifier2.parameters(), 'lr': 0.00001},
                    {'params': model.classifier3.parameters(), 'lr': 0.00001},
                    {'params': model.classifier4.parameters(), 'lr': 0.00001},
                    {'params': model.classifier5.parameters(), 'lr': 0.00001},
                    #{'params': model.classifier6.parameters(), 'lr': 0.01},
                    #{'params': model.classifier7.parameters(), 'lr': 0.01}
                ], weight_decay=1e-5)


        # Decay LR by a factor of 0.1 every 40 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

        ######################################################################
        # Train and evaluate
        # ^^^^^^^^^^^^^^^^^^
        #
        # It should take around 1-2 hours on GPU.

    
        # save opts
        #with open('%s/opts_reID.yaml'%dir_out_config,'w') as fp:
        #    yaml.dump(vars(opt), fp, default_flow_style=False)

        # model to gpu
        model = model.cuda()
        if fp16:
            model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

        criterion = nn.CrossEntropyLoss()

        model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=50)

    net = PCB(751)
    net.load_state_dict(torch.load(weights_ini))
    net.change_last(len(class_names))
    #net = PCB(len(class_names))
    print(train(net))

