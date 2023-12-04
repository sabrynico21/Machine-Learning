import torch
import numpy as np
np.random.seed(1328)
torch.random.manual_seed(1328)
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import warnings
import matplotlib.pyplot as plt
import PIL
import torch.nn.functional as F
import tensorflow as tf
import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam
import torchmetrics
from sklearn.metrics import f1_score
import argparse
from PIL import Image

class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, value, num):
        self.sum += value * num
        self.num += num

    def compute(self):
        try:
            return self.sum/self.num
        except:
            return None

def map_color_to_classid(labels): 
    rgb_view = labels.reshape(-1, 3)
    rgb_tuples = [tuple(rgb) for rgb in rgb_view]
    mapped_values_list = [color_to_class.get(rgb_tuple, -1) for rgb_tuple in rgb_tuples]
    mapped_values = np.array(mapped_values_list, dtype=int).reshape(labels.shape[:-1])
    return mapped_values

color_to_class={(102, 102, 156): 2,
                    (107, 142, 35): 4, 
                    (0, 0, 142): 7, 
                    (220, 220, 0): 3, 
                    (70, 130, 180):5 , 
                    (0, 0, 90): 7,
                    (150, 100, 100): 2, 
                    (255, 0, 0): 6, 
                    (0, 0, 230): 7, 
                    (220, 20, 60): 6,
                    (244, 35, 232): 1, 
                    (128, 64, 128): 1, 
                    (111, 74, 0): 0,
                    (150, 120, 90): 2,
                    (70, 70, 70): 2, 
                    (119, 11, 32): 7, 
                    (0, 80, 100): 7, 
                    (250, 170, 30): 3, 
                    (152, 251, 152): 4,
                    (0, 0, 110): 7,
                    (0, 60, 100): 7, 
                    (0, 0, 0): 0, 
                    (190, 153, 153): 2, 
                    (0, 0, 70): 7, 
                    (81, 0, 81): 0,
                    (153, 153, 153): 3,
                    (230, 150, 140): 1,
                    (250, 170, 160): 1,
                    (180, 165, 180): 2}

class_to_color={ 0:(0,0,0), 
                    1:(128, 64,128), 
                    2:( 70, 70, 70), 
                    3:(153,153,153), 
                    4:(107,142, 35), 
                    5:(70,130,180),
                    6:(220, 20, 60),  
                    7:(0, 0,142)
    }  

class_to_name={ 0: "Void",
                1: "Flat",
                2: "Construction",
                3: "Object",
                4: "Nature",
                5: "Sky",
                6: "Human",
                7: "Vehicle"
}

def get_one_hot_encoding(image,num_classes):
    image=np.array(image)
    image=image[:,:,:3]
    class_id = map_color_to_classid(image)
    class_id_tensor = torch.tensor(class_id).long()
    one_hot_encoding = F.one_hot(class_id_tensor, num_classes)
    one_hot_encoding = one_hot_encoding.permute(2, 0, 1)
    return one_hot_encoding

def map_classid_to_rgb(labels_batch, batch_size):
    rgb_batch = torch.zeros(batch_size, 128, 256, 3, dtype=torch.uint8)
    for class_id, color in class_to_color.items():
        mask = labels_batch == class_id
        rgb_batch[mask] = torch.tensor(color, dtype=torch.uint8)
    return rgb_batch

def calculate_iou(pred, label, num_classes):
    iou_scores = []
    mean_iou=[]
    for batch_index in range(pred.shape[0]):
        for class_id in range(num_classes):
            intersection = torch.logical_and(pred[batch_index] == class_id, label[batch_index] == class_id).sum().item()
            union = torch.logical_or(pred[batch_index] == class_id, label[batch_index] == class_id).sum().item()
            iou = intersection / union if union > 0 else 0.0
            iou_scores.append(iou)
        mean_iou.append(sum(iou_scores) / len(iou_scores))
    return sum(mean_iou) / pred.shape[0]

def calculate_iou_for_classes(pred, label, num_classes):
    mean_iou={}
    for class_id in range(num_classes):
        mean_iou[class_id]=0
    n=pred.shape[0]
    for batch_index in range(n):
        for class_id in range(num_classes):
            intersection = torch.logical_and(pred[batch_index] == class_id, label[batch_index] == class_id).sum().item()
            union = torch.logical_or(pred[batch_index] == class_id, label[batch_index] == class_id).sum().item()
            iou = intersection / union if union > 0 else 0.0 
            mean_iou[class_id]+=iou
    for class_id in range(num_classes):
        mean_iou[class_id]=mean_iou[class_id]/n
    return mean_iou

def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, num_classes, model_name):
    loader = {
    'train' : train_loader,
    'val' : val_loader
    }
    metrics_dict = {
        'train': {
            'iou': AverageValueMeter(),
            'loss': AverageValueMeter(),
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
            'precision': torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass'),
            'recall': torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
        },
        'val': {
            'iou': AverageValueMeter(),
            'loss': AverageValueMeter(),
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
            'precision': torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass'),
            'recall': torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
        }
    }
    avg_metrics_dict = {
        'train': {
            'iou': AverageValueMeter(),
            'loss': AverageValueMeter(),
            'accuracy': AverageValueMeter(),
            'precision': AverageValueMeter(),
            'recall': AverageValueMeter()
        },
        'val': {
            'iou': AverageValueMeter(),
            'loss': AverageValueMeter(),
            'accuracy': AverageValueMeter(),
            'precision': AverageValueMeter(),
            'recall': AverageValueMeter()
        }
    }
    for epoch in range(num_epochs):
        for mode in ['train', 'val']:
            model.train() if mode == 'train' else model.eval()
            metrics = metrics_dict[mode]
            avg_metrics=avg_metrics_dict[mode]
            metrics['iou'].reset()
            metrics['loss'].reset()
            metrics['accuracy'].reset()
            metrics['precision'].reset()
            metrics['recall'].reset()
            for param in model.parameters():
                param.requires_grad = (mode == 'train')
            with torch.set_grad_enabled(mode=='train'):
                for inputs, labels in loader[mode]:
                    outputs = model(inputs)['out']
                    labels = torch.argmax(labels, dim=1).long()
                    loss = criterion(outputs, labels)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    n=inputs.shape[0]
                    _, predicted = torch.max(outputs, 1)
                    metrics['iou'].add(calculate_iou(predicted, labels, num_classes),n)
                    metrics['loss'].add(loss.item(),n)
                    metrics['accuracy'](predicted, labels)
                    metrics['precision'](predicted, labels)
                    metrics['recall'](predicted, labels)
                    
            print(
                f'{mode.capitalize()} Mode - '
                f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Loss: {metrics["loss"].compute():.4f}, '
                f'Accuracy: {metrics["accuracy"].compute():.4f}, '
                f'Precision: {metrics["precision"].compute():.4f}, '
                f'Recall: {metrics["recall"].compute():.4f}, '
                f'IoU: {metrics["iou"].compute():.4f}'
            )
            avg_metrics['iou'].add(metrics["iou"].compute(),1)
            avg_metrics['loss'].add(metrics["loss"].compute(),1)
            avg_metrics['accuracy'].add(metrics["accuracy"].compute(),1)
            avg_metrics['precision'].add(metrics["precision"].compute(),1)
            avg_metrics['recall'].add(metrics["recall"].compute(),1)

        torch.save(model.state_dict(), f'{model_name}_epoch_{epoch+1}.pth')
    for mode in ['train', 'val']:
        avg_metrics = avg_metrics_dict[mode]
        print(f'Avg {mode.capitalize()} Metrics - '
              f'Loss: {avg_metrics["loss"].compute():.4f}, '
              f'Accuracy: {avg_metrics["accuracy"].compute():.4f}, '
              f'Precision: {avg_metrics["precision"].compute():.4f}, '
              f'Recall: {avg_metrics["recall"].compute():.4f}, '
              f'IoU: {avg_metrics["iou"].compute():.4f}'
             )
        
def test_model(model, test, num_classes):
    model.eval()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    precision = torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass')
    recall = torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
    avg_loss=AverageValueMeter()
    avg_iou={0:AverageValueMeter(),
             1:AverageValueMeter(),
             2:AverageValueMeter(),
             3:AverageValueMeter(),
             4:AverageValueMeter(),
             5:AverageValueMeter(),
             6:AverageValueMeter(),
             7:AverageValueMeter()} 
    accuracy.reset()
    precision.reset()
    recall.reset()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test:   
            outputs = model(inputs)['out']
            labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)  
            n=inputs.shape[0] 
            
            avg_loss.add(loss.item(),n)
            iou=calculate_iou_for_classes(predictions, labels, num_classes)
            for class_id in range(num_classes):
                avg_iou[class_id].add(iou[class_id],n)
            accuracy(predictions, labels)
            precision(predictions, labels)
            recall(predictions, labels)
            
    loss_value=avg_loss.compute()
    for class_id in range(num_classes):
        avg_iou[class_id]=avg_iou[class_id].compute()
    accuracy_value = accuracy.compute()
    precision_value = precision.compute()
    recall_value = recall.compute()
    print( f'Loss: {loss_value:.4f}, '
              f'Accuracy: {accuracy_value:.4f}, '
              f'Precision: {precision_value:.4f}, '
              f'Recall: {recall_value:.4f}, ', '\n'
              )
    for class_id in range(num_classes):
        print(f'IoU - Class {class_to_name[class_id]}: {avg_iou[class_id]:.4f} ')

def predict(model, image,norm_image, label, model_name,n, criterion, num_classes):
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    precision = torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass')
    recall = torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
    model[n].load_state_dict(torch.load(f'{model_name[n]}_epoch_5.pth'))
    model[n].eval()
    with torch.no_grad():  
        outputs = model[n](norm_image)['out']
        prediction = torch.argmax(outputs, dim=1)  
        label = torch.argmax(label, dim=1)
        loss = criterion(outputs, label)
    accuracy(prediction, label)
    precision(prediction, label)
    recall(prediction, label)
    iou=calculate_iou_for_classes(prediction,label,num_classes)
    print( f'Loss: {loss:.4f}, '
              f'Accuracy: {accuracy.compute():.4f}, '
              f'Precision: {precision.compute():.4f}, '
              f'Recall: {recall.compute():.4f}, ', '\n'
              )
    for class_id in range(num_classes):
        print(f'IoU - Class {class_to_name[class_id]}: {iou[class_id]:.4f} ')
    prediction = map_classid_to_rgb(prediction,1)
    label = map_classid_to_rgb(label,1)
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    images={
        0: image,
        1: label.squeeze().numpy(),
        2: prediction.squeeze().numpy()
    }
    titles=["original image", "label image", f'{model_name[n]} prediciction']
    for i in range (0,3):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    
    plt.show()

def main():

    parser = argparse.ArgumentParser(description='Train or test a model')
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], help='Specify whether to train or test the models or predict model results')
    parser.add_argument('--image', help='Path to the image file (required for the predict mode)')
    parser.add_argument('--label', help='Path to the label file (required for the predict mode)')
    args = parser.parse_args()

    image_size=(128, 256)
    num_classes=8
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.Normalize(mean=mean, std=std)
        ])
    label_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
        transforms.Lambda(lambda x: get_one_hot_encoding(x, num_classes))
        ])

    fcn_model_1 = models.segmentation.fcn_resnet50(pretrained=True)
    fcn_model_1.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)

    fcn_model_2 = models.segmentation.fcn_resnet101(pretrained=True)
    fcn_model_2.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)

    deeplab_model_1 = models.segmentation.deeplabv3_resnet50(pretrained=True)
    deeplab_model_1.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    deeplab_model_2 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    deeplab_model_2.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    model_type={
        1: fcn_model_1,
        2: fcn_model_2,
        3: deeplab_model_1,
        4: deeplab_model_2
    }
    model_name={
        1: "fcn_resnet50",
        2: "fcn_resnet101",
        3: "deeplab_resnet50",
        4: "deeplab_resnet101"
    }
    criterion = nn.CrossEntropyLoss() 
    if args.mode == 'train':
        dataset =Cityscapes(root="C:/Users/sabry/Downloads/cityscapes", split='train', mode='fine',
                     target_type='color',target_transform=label_transform, transform= transforms.Lambda(lambda x: transform(x)))
        dataset_size =len(dataset)
        train_size = int(0.85 * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
      
        num_epochs = 5
        for i in range(1,5):
            optimizer = Adam(model_type[i].parameters(), lr=0.001)
            train_model(num_epochs, train_loader,val_loader, model_type[i], criterion, optimizer, num_classes, model_name[i])
    elif args.mode == 'test':
        test_dataset =Cityscapes(root="C:/Users/sabry/Downloads/cityscapes", split='val', mode='fine',
                     target_type='color',target_transform=label_transform, transform= transforms.Lambda(lambda x: transform(x)))
    
        test=DataLoader(test_dataset,batch_size=32,shuffle=False)
        m=4
        model_type[m].load_state_dict(torch.load(f'{model_name[m]}_epoch_5.pth'))
        test_model(model_type[m],test,num_classes)

    elif args.mode == 'predict':
        if args.image is None or args.label is None:
            print('Error: --image and --label are required for predict mode.')
        else:    
            image = Image.open(args.image).convert("RGB")
            tensor_image = transform(image)
            tensor_image = tensor_image.unsqueeze(0)

            label = Image.open(args.label).convert("RGB")
            tensor_label = label_transform(label) 
            tensor_label = tensor_label.unsqueeze(0)
            best_model=4
            predict(model_type,image,tensor_image, tensor_label, model_name, best_model, criterion, num_classes)
    else:
        print('Invalid mode. Please use "train", "test" or "predict".')

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
    main()