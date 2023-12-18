import torch
import clip
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def save_embeddings(images_path, embeddings_path):
    '''
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, _ = clip.load('ViT-B/32', device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float16)
    ])
    
    dataset = datasets.ImageFolder(images_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    image_folders = [label_name for label_name in os.listdir(images_path)]
    image_folders = sorted(image_folders)
    print(len(image_folders))
    
    assert len(image_folders) == 101

    for i, data in enumerate(dataloader):
        img, label = data
        embedding = model.visual(img.to(device))

        file_path, _ = dataset.samples[i]
        file_basename = os.path.basename(file_path)
        file_name = os.path.splitext(file_basename)[0]

        label_folder = embeddings_path + image_folders[label] + "/"
        new_path = label_folder + file_name + ".pt"

        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        torch.save(embedding, new_path)

chopped_path = "/projectnb/ec523kb/projects/chopped_data/"

'''datapath_test = chopped_path + "food101_20percent/" #corrupted file
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")

print("done with 20")'''

'''datapath_test = chopped_path + "food101_40percent/"
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")'''

'''print("done with 40") #only has 99

datapath_test = chopped_path + "food101_60percent/"
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")

print("done with 60")'''

'''datapath_test = chopped_path + "food101_70percent/"
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")

print("done with 70")'''

datapath_test = chopped_path + "food101_80percent/" # has 102
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")

print("done with 80")

'''datapath_test = chopped_path + "food101_90percent/"
save_embeddings(datapath_test + "train/", datapath_test+"embeddings/")

print("done with 90")'''