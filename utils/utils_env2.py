import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2, cla: list = ['angii','control_selected','iso']):
    """Split dataset into training and validation sets.
    
    Args:
        root (str): Root directory of the dataset
        val_rate (float): Proportion of data to use for validation (default: 0.2)
        cla (list): List of class names (default: ['angii','control_selected','iso'])
    
    Returns:
        Four lists containing training/validation paths and labels
    """
    random.seed(0)  # Ensure reproducible random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traverse folders, each folder corresponds to one class
    flower_class = cla
    # Sort to ensure consistency across platforms
    # For 2-class classification, ensure control corresponds to 0 and model group corresponds to 1
#     if len(flower_class) == 2:
#         if "angii" in flower_class:
#             flower_class.sort(reverse=True)
#         else:
#             flower_class.sort()
#     else:
#         flower_class.sort()
    # Generate class names and corresponding numeric indices
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []       # Store all image paths for training set
    train_images_label = []      # Store label indices for training set images
    val_images_path = []         # Store all image paths for validation set
    val_images_label = []        # Store label indices for validation set images
    every_class_num = []         # Store total sample count for each class
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tiff"]  # Supported file extensions
    # Traverse files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        
        # Collect all supported image files in class directory
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()  # Sort for cross-platform consistency
        
        image_class = class_indices[cla]  # Get class index
        every_class_num.append(len(images))  # Record class sample count
        
        # Randomly select validation samples
        val_path = random.sample(images, k=int(len(images) * val_rate))

        # Split into training and validation sets
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    # Print dataset statistics
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    # Validate dataset size
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    """Visualize a batch of images from the data loader.
    
    Args:
        data_loader: DataLoader containing images and labels
    """
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    
    # Load class indices mapping
    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # Convert from [C, H, W] to [H, W, C] for visualization
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()

            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    """Save list to pickle file.
    
    Args:
        list_info: List to save
        file_name: Output pickle file path
    """
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    """Read list from pickle file.
    
    Args:
        file_name: Pickle file path
        
    Returns:
        List loaded from pickle file
    """
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train model for one epoch.
    
    Args:
        model: Neural network model
        optimizer: Optimizer for training
        data_loader: Training data loader
        device: Device to run training on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average loss, accuracy) for the epoch
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Cumulative loss
    accu_num = torch.zeros(1).to(device)   # Cumulative correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """Evaluate model performance.
    
    Args:
        model: Neural network model
        data_loader: Validation data loader
        device: Device to run evaluation on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average loss, accuracy) for evaluation
    """
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Cumulative correct predictions
    accu_loss = torch.zeros(1).to(device)  # Cumulative loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
