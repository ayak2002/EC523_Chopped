import requests
from PIL import Image
from io import BytesIO

def download_images(url_list, target_folder):
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(f"{target_folder}/image_{i}.jpg")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

download_images(image_urls, 'path_to_save_images')


from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root='path_to_save_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
