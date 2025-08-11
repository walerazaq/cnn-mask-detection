import torch
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import glob as glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

from utils import *
from prepare_dataset import *
from visualisation import *
from model import *
from training import *

BATCH_SIZE = 4 # batches to load data in
RESIZE_TO = 400 # resize the image for training and transforms

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = ['background', 'without_mask', 'with_mask', 'mask_weared_incorrect']
NUM_CLASSES = 4

IMG_DIR = '/input/face-mask-detection/images'
XML_DIR = '/input/face-mask-detection/annotations'

torch.cuda.is_available()
    
train_set, valid_set, test_set = train_valid_test_split(img_dir = IMG_DIR)

train_dataset = maskSet(train_set, RESIZE_TO, RESIZE_TO, CLASSES, IMG_DIR, XML_DIR)
tran_trainDataset = maskSet(train_set, RESIZE_TO, RESIZE_TO, CLASSES, IMG_DIR, XML_DIR, get_train_transform())
valid_dataset = maskSet(valid_set, RESIZE_TO, RESIZE_TO, CLASSES, IMG_DIR, XML_DIR)

new_trainSet = ConcatDataset([train_dataset, tran_trainDataset])  

train_loader = DataLoader(
    new_trainSet,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

for i in random.sample(range(len(new_trainSet)), 5):
    image, target = new_trainSet[i]
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    visualize_sample(image, target)

plt.style.use('ggplot')

SAVE_MODEL_EPOCH = 2 # save model after these many epochs
NUM_EPOCHS = 20 # number of epochs to train for

model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
train_loss_list = []
val_loss_list = []
MODEL_NAME = 'model'

for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    
    train_loss_hist.reset()
    val_loss_hist.reset()
    
    start = time.time()
    train_loss = train(train_loader, model)
    val_loss = validate(valid_loader, model)
    
    print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
    print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
        
    if (epoch+1) == NUM_EPOCHS: # save model once at the end
        torch.save(model.state_dict(), f"/kaggle/working/model{epoch+1}.pth")
        
    elif (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        torch.save(model.state_dict(), f"/working/model{epoch+1}.pth")
        print('SAVING MODEL COMPLETE...\n')

window_size2 = 50 # window size for moving average

moving_average2 = np.convolve(train_loss, np.ones(window_size2) / window_size2, mode='valid')

plt.plot(moving_average2, color="r", label="Moving Average")
plt.title("Training Loss (Moving Average)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_loss_plot.png")
plt.show()

window_size1 = 25 # window size for  moving average

moving_average1 = np.convolve(val_loss, np.ones(window_size1) / window_size1, mode='valid')

plt.plot(moving_average1, color="b", label="Moving Average")
plt.title("Validation Loss (Moving Average)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("validation_loss_plot.png")
plt.show()

model.load_state_dict(torch.load('/working/model18.pth', map_location=DEVICE))
model.eval()

detection_threshold = 0.8

test_images = []

for i in test_set:
    test_images.append(os.path.join(IMG_DIR, i))

for i in random.sample(range(len(test_images)), 5):
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(float)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        labels = labels[scores >= detection_threshold]
        
        draw_boxes = boxes.copy()
        
        for box, label in zip(draw_boxes, labels):
            label = CLASSES[label]
            
            if label == "without_mask":
                color = (0, 0, 255)
                
            elif label == "with_mask":
                color = (0, 255, 0)
                
            elif label == "mask_weared_incorrect":
                color = (255, 0, 0)
                
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 1)
            cv2.putText(orig_image, label,
                        (int(box[0]), int(box[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()
    print(f"Image {i+1} done...")
    print('-'*50)
#test