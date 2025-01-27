import webdataset as wds
import importlib
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import CLIPModel

wds = importlib.reload(wds)

import os
url = 'pipe:aws s3 cp s3://laion-super-resolution/{00000..16000}.tar - --endpoint-url=https://s3.eu-central-1.wasabisys.com'


def identity(x):
    return x

class ResizeIfSmaller(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, sample):
        H, W = sample.size

        if H < self.image_size or W < self.image_size:
            return transforms.Resize(self.image_size)(sample)
        else:
            return sample

# generating embeddings does not work on 256
# works on 224, does not work on 64
image_size=24

preproc = transforms.Compose([
    ResizeIfSmaller(image_size),
    #transforms.RandomChoice([
    transforms.RandomCrop(image_size),
        #transforms.Resize((image_size, image_size)),
    #]),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = (
    wds.WebDataset(url)
    .shuffle(32)
    .decode("pil")
    .to_tuple("jpg", "json")
    .map_tuple(preproc, identity)
    #.batched(32) # we specify the batching in DataLoader instance
)

def collate_fn(batch):
    images = torch.cat([x[0].unsqueeze(0) for x in batch], dim=0)
    js = [x[1] for x in batch]
    return images, js

dloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=16, prefetch_factor=4, pin_memory=True, collate_fn = collate_fn)

for img, js in tqdm(dloader):
    print("Image has shape: " + str(img))
    print("type: " + str(type(img)))
    print("Shape: " + str(img.shape))
    print("Initializing model: ")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    outputs = model.get_image_features(img)
    print("Outputs/embeddings for the batch: " + str(outputs))
    print("shape of outputs: " + str(outputs.shape))
    exit(0)