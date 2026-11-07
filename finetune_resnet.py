import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils.utils import extract_prims_from_usda
from tqdm import tqdm
import math
import argparse

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        target = torch.tensor(item['label'], dtype=torch.float32)  # Convert label to tensor
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# Load pretrained ResNet or create from scratch
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
output_size = 4  # Size of your output vector
model.fc = nn.Linear(model.fc.in_features, output_size)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr = 0.001,
    weight_decay=0.0001
)

# Prepare your data (example with dummy dataset)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# extract the light as [R, G, B, log(Intensity)]
hf_dataset = 'train_data'
train_data = load_dataset('imagefolder', data_dir=hf_dataset, split='train')
eval_data = load_dataset('imagefolder', data_dir=hf_dataset, split='validation')

def extract_light(sample):
    _, lights = extract_prims_from_usda(sample['text'], 'assets')
    rgb = list(lights.color)
    mag = (rgb[0]**2 + rgb[1]**2 + rgb[2]**2)**(1/2)
    rgb = [rgb[i]/mag for i in range(3)]
    intensity = lights.intensity*mag
    sample['label'] = rgb + [math.log10(intensity)]
    return sample

train_data = train_data.map(extract_light)
eval_data = eval_data.map(extract_light)
train_dataset = HFDataset(train_data, transform=transform)
eval_dataset = HFDataset(eval_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=True)

# Training loop
steps = len(train_loader)*20
min_steps = len(train_loader)
warmup_steps = 0.1*steps

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.0001,
    total_iters=warmup_steps
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max = steps - warmup_steps,
    eta_min=1e-6
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='finetuned_resnet')
args = parser.parse_args()


eval_step = min_steps
log_step = 15
step = 0
model_dir = args.model_dir
os.makedirs(args.model_dir, exist_ok=True)
prev_loss = 1e+8
patience = 0
max_patience = 3
best_loss = 1e+8
best_id = 1
stop_train = False
while step < steps:
    model.train()
    train_loss = 0.0
    eval_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        step += 1

        if step < warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        if step % log_step == 0:
            train_loss /= step
            print(f'Step {step}/{steps}, Train Loss: {train_loss}')

        if step % eval_step == 0:
            model.eval()
            with torch.no_grad():
                for images, labels in tqdm(eval_loader, desc='Evaluation'):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()
            eval_loss /= len(eval_loader)
            checkpoint_id = step // eval_step
            print(f'Eval Step {checkpoint_id}, Eval Loss: {eval_loss}')
            torch.save(model.state_dict(), os.path.join(model_dir, f'checkpoint-{checkpoint_id}'))
            
            if steps > min_steps:
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_id = checkpoint_id
                    patience = 0
                if eval_loss > prev_loss:
                    patience += 1
                prev_loss = eval_loss
            
        if patience == max_patience:
            stop_train = True
            break
    
    if stop_train == True:
        break

# Mark best model
print(f'Best model saved at checkpoint-{best_id} with loss {best_loss}.')