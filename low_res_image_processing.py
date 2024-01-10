import webdataset as wds
import importlib
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

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


image_size=256

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
    #.batched(32)
)

def collate_fn(batch):
    images = torch.cat([x[0].unsqueeze(0) for x in batch], dim=0)
    js = [x[1] for x in batch]
    return images, js

dloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=16, prefetch_factor=4, pin_memory=True, collate_fn = collate_fn)

for img, js in tqdm(dloader):
    print("IMg: " + str(img))