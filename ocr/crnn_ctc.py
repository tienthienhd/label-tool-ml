import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import requests
import io
import hashlib
import urllib

from PIL import Image
from torch.nn import CTCLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped, get_local_path
from torchvision.models import ResNet18_Weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_transforms = transforms.Compose([
    transforms.Resize((216, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


def get_transformed_image(url):
    filepath = get_local_path(url,
                              hostname='http://localhost:8080',
                              access_token='70f77f55f9c77c00cc688a885bf49a319f5b8e5e')

    with open(filepath, mode='rb') as f:
        image = Image.open(f).convert('RGB')

    return image_transforms(image)


class ImageOcrDataset(Dataset):

    def __init__(self, image_urls, label_texts, charset: str):
        self.charset = sorted(charset)

        self.c2i = {c: i for i, c in enumerate(self.charset)}
        self.i2c = {i: c for c, i in self.c2i.items()}

        self.images, self.labels = [], []
        for image_url, text in zip(image_urls, label_texts):
            try:
                image = get_transformed_image(image_url)
            except Exception as exc:
                print(exc)
                continue
            self.images.append(image)
            self.labels.append([self.c2i[c] for c in text])

    def __getitem__(self, index):
        target_length = [len(self.labels[index])]
        return self.images[index], torch.LongTensor(self.labels[index]), torch.LongTensor(target_length)

    def __len__(self):
        return len(self.images)


class ImageOcr(object):

    def __init__(self, num_classes, freeze_extractor=False):
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_extractor:
            print('Transfer learning with a fixed ConvNet feature extractor')
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('Transfer learning with a full ConvNet finetuning')

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model = self.model.to(device)

        self.criterion = CTCLoss(reduction='sum', zero_infinity=True)
        self.criterion.to(device)


        if freeze_extractor:
            self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, image_urls):
        images = torch.stack([get_transformed_image(url) for url in image_urls]).to(device)
        with torch.no_grad():
            return self.model(images).to(device).data.numpy()

    def train(self, dataloader, num_epochs=5):
        since = time.time()

        self.model.train()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloader:
                images, targets, target_lengths = [d.to(device) for d in data]

                logits = self.model(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                batch_size = images.size(0)
                input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
                target_lengths = torch.flatten(target_lengths)

                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # statistics
                running_loss += loss.item() * data[0].size(0)
                self.scheduler.step(epoch)

            epoch_loss = running_loss / len(dataloader.dataset)

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, 0))

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model


class ImageOcrAPI(LabelStudioMLBase):

    def __init__(self, freeze_extractor=False, **kwargs):
        super(ImageOcrAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        self.freeze_extractor = freeze_extractor
        if self.train_output:
            self.classes = self.train_output['classes']
            self.model = ImageOcr(len(self.classes), freeze_extractor)
            self.model.load(self.train_output['model_path'])
        else:
            self.model = ImageOcr(len(self.classes), freeze_extractor)

    def reset_model(self):
        self.model = ImageOcr(len(self.classes), self.freeze_extractor)

    def predict(self, tasks, **kwargs):
        image_urls = [task['data'][self.value] for task in tasks]
        logits = self.model.predict(image_urls)
        predicted_label_indices = np.argmax(logits, axis=1)
        predicted_scores = logits[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.classes[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': float(score)})
        return predictions

    def fit(self, completions, workdir=None, batch_size=32, num_epochs=10, **kwargs):
        image_urls, image_classes = [], []
        print('Collecting annotations...')
        for completion in completions:
            if is_skipped(completion):
                continue
            image_urls.append(completion['data'][self.value])
            image_classes.append(get_choice(completion))

        print(f'Creating dataset with {len(image_urls)} images...')
        dataset = ImageOcrDataset(image_urls, image_classes)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print('Train model...')
        self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        print('Save model...')
        model_path = os.path.join(workdir, 'model.pt')
        self.model.save(model_path)

        return {'model_path': model_path, 'classes': dataset.classes}
