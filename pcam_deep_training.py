# %% [markdown]
# # PCam Training Script (Hugging Face version)
# This script loads the PatchCamelyon dataset from Hugging Face,
# prepares it for PyTorch, and trains CNN backbones using ShuffleSplit cross-validation.
# It supports command-line arguments for configuration and saves results to a Parquet file.

# %%
import argparse
import gc
import os
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as torch_models
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import ShuffleSplit, train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms

# %% [markdown]
# ## Argument Parsing & Configuration

# %%
def get_available_models():
    """Returns a list of available model names."""
    return [
        "alexnet", "resnet18", "resnet34", "resnet50", "resnet101",
        "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2",
        "wide_resnet101_2", "densenet121", "densenet161", "densenet169",
        "densenet201", "inception_v3", "googlenet", "mobilenet_v2",
        "mobilenet_v3_large", "mobilenet_v3_small"
    ]

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PCam training benchmarks.")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use.")
    parser.add_argument("--n-splits", type=int, nargs='+', default=[1], help="List of n_splits for ShuffleSplit.")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 1], metavar=('START', 'END_EXCLUSIVE'), help="Range of seeds.")
    parser.add_argument("--study-sizes", type=int, nargs='+', default=[1000, 10000], help="List of study set sizes.")
    parser.add_argument("--model-choices", type=str, nargs='+', default=["resnet18"], choices=get_available_models(), help="Algorithms to train.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training and evaluation batch size.")
    return parser.parse_args()

args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Training Configuration
epochs = args.epochs
loss_function = nn.CrossEntropyLoss()
test_size = 1/6
valid_size = 1/6
BatchSize = args.batch_size
study_set_sizes = args.study_sizes
backbone_list = args.model_choices
n_splits_values = args.n_splits
seeds = range(args.seeds[0], args.seeds[1])

OUTPUT_DIR = Path("results_output_pcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("\n--- Configuration ---")
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("---------------------\n")


# %% [markdown]
# ## Data Loading and Preparation

# %%
# Define data transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset from Hugging Face
HF_NAME = "1aurent/PatchCamelyon"
data_dir = "./pcam_data/"
hf = load_dataset(data_dir)

class HuggingFacePCam(Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        label = int(item["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# Create a single, unified dataset
full_dataset = ConcatDataset([
    HuggingFacePCam(hf["train"], transform=transform),
    HuggingFacePCam(hf["validation"], transform=transform)
])

print(f"Loaded Hugging Face PCam dataset. Total samples: {len(full_dataset)}")

# %% [markdown]
# ## Model Definition

# %%
def get_model_features(backbone, **kwargs):
    """Get a feature extractor model from torchvision."""
    backbone_dict = {model: getattr(torch_models, model) for model in get_available_models()}
    if backbone not in backbone_dict:
        raise ValueError(f"Backbone `{backbone}` is not supported.")

    # Use the new 'weights' parameter for pretrained models
    model = backbone_dict[backbone](weights='IMAGENET1K_V1')

    if "resnet" in backbone or "resnext" in backbone:
        return nn.Sequential(*list(model.children())[:-2])
    elif "densenet" in backbone or "mobilenet" in backbone or "alexnet" in backbone:
        return model.features
    elif "inception_v3" in backbone or "googlenet" in backbone:
        # Inception_v3 needs special handling for aux logits during training
        if backbone == "inception_v3" and kwargs.get('aux_logits', True):
             # Return a model that can handle aux logits
            return nn.Sequential(*list(model.children())[:-1])
        return nn.Sequential(*list(model.children())[:-3])
    raise ValueError(f"Feature extraction logic not defined for {backbone}")

class CNN_Patch_Model(nn.Module):
    """A classification model using a torchvision backbone."""
    def __init__(self, backbone, nr_classes=2):
        super().__init__()
        self.backbone_name = backbone
        # InceptionV3 requires specific input size and handling of aux logits
        is_inception = backbone == "inception_v3"
        self.feat_extract = get_model_features(backbone, aux_logits=is_inception)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy_input = torch.rand(2, 3, 299 if is_inception else 96, 299 if is_inception else 96)
            out_features_shape = self.feat_extract(dummy_input)
            # Handle tuple output from InceptionV3
            if isinstance(out_features_shape, tuple):
                out_features_shape = out_features_shape[0]
            out_features = out_features_shape.shape[1]

        self.classifier = nn.Linear(out_features, nr_classes)

    def forward(self, imgs):
        # Handle InceptionV3 aux logits during training
        if self.backbone_name == "inception_v3" and self.training:
            output = self.feat_extract(imgs)
            # In training, InceptionV3 returns InceptionOutputs(logits, aux_logits)
            # We only use the main logits for classification head
            feat = output.logits
        else:
            feat = self.feat_extract(imgs)

        pooled_feat = self.pool(feat)
        flat_feat = torch.flatten(pooled_feat, 1)
        logit = self.classifier(flat_feat)
        return logit

# %% [markdown]
# ## Training and Evaluation Functions

# %%
def train_model(model, dataloaders, criterion, optimizer, num_epochs, backbone_name, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataset_size = len(dataloaders[phase].dataset)

            for inputs, labels in dataloaders[phase]:
                # Resize for InceptionV3
                if model.backbone_name == "inception_v3":
                    inputs = transforms.functional.resize(inputs, [299, 299])
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Finished training {backbone_name}. Best valid Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device, set_name):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    dataset_size = len(dataloader.dataset)
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in dataloader:
            if model.backbone_name == "inception_v3":
                inputs = transforms.functional.resize(inputs, [299, 299])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    eval_time = time.time() - start_time
    acc = running_corrects.double() / dataset_size
    loss = running_loss / dataset_size
    print(f"Accuracy on {set_name}: {acc:.4f}, Loss: {loss:.4f}, Time: {eval_time:.2f}s")
    return {"accuracy": acc.item(), "loss": loss, "runtime": eval_time}

# %% [markdown]
# ## Cross-validation Training Loop

# %%
all_results = []

# Extract all labels for stratified splitting
print("Extracting labels for stratification...")
if Path("./pcam_data/all_labels.npz").exists():
    labels_data = np.load("./pcam_data/all_labels.npz")
    all_labels = labels_data['labels']
else:
    Path("./pcam_data").mkdir(exist_ok=True)
    all_labels = np.array([label for _, label in full_dataset])
    np.savez("./pcam_data/all_labels.npz", labels=all_labels)
print(f"Total labels extracted: {len(all_labels)}")

# Split full dataset into a large training pool and a smaller pool for study sets
benchmarking_indices, leftout_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=50000,
    stratify=all_labels,
    random_state=42
)
benchmarking_set = Subset(full_dataset, benchmarking_indices)
benchmarking_loader = torch.utils.data.DataLoader(
    benchmarking_set, batch_size=512,
    shuffle=False, pin_memory=True, num_workers=8
)
print(f"Benchmarking set size: {len(benchmarking_set)}")
print(f"Pool for study sets size: {len(leftout_indices)}")

for seed in seeds:
    for n_splits in n_splits_values:
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        for backbone in backbone_list:
            for study_size in study_set_sizes:
                print(f'\n{"="*40}\nStarting {backbone} with study set size {study_size}, n_splits {n_splits}, seed {seed}\n{"="*40}\n')

                leftout_labels = np.array(all_labels)[leftout_indices]
                if study_size < len(leftout_indices):
                    study_indices_relative, _ = train_test_split(
                        np.arange(len(leftout_indices)), train_size=study_size,
                        stratify=leftout_labels, random_state=seed
                    )
                else:
                    study_indices_relative = np.arange(len(leftout_indices))
                study_indices_absolute = leftout_indices[study_indices_relative]
                study_set_for_split = Subset(full_dataset, study_indices_absolute)

                for fold, (train_val_ids, test_ids) in enumerate(ss.split(study_set_for_split)):
                    print(f'--- Seed {seed} | n_splits {n_splits} | Study Size {study_size} | Fold {fold+1}/{n_splits} ---')
                    torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

                    # Split the train_val_ids further into train and validation
                    train_val_labels = np.array(all_labels)[np.array(study_indices_absolute)[train_val_ids]]

                    # Calculate the proportion of the validation set relative to the combined train+validation set
                    validation_proportion = valid_size / (1.0 - test_size)

                    train_ids_rel, valid_ids_rel = train_test_split(
                        np.arange(len(train_val_ids)),
                        test_size=validation_proportion,
                        random_state=seed,
                        stratify=train_val_labels
                    )
                    train_ids = train_val_ids[train_ids_rel]
                    valid_ids = train_val_ids[valid_ids_rel]

                    train_subset = Subset(study_set_for_split, train_ids)
                    valid_subset = Subset(study_set_for_split, valid_ids)
                    test_subset = Subset(study_set_for_split, test_ids)
                    print(f"Train set size: {len(train_subset)}, Validation set size: {len(valid_subset)}, Test set size: {len(test_subset)}")

                    loaders_dict = {
                        'train': DataLoader(train_subset, batch_size=BatchSize, shuffle=True, pin_memory=True, num_workers=4),
                        'valid': DataLoader(valid_subset, batch_size=BatchSize, shuffle=False, pin_memory=True, num_workers=4),
                        'test': DataLoader(test_subset, batch_size=BatchSize, shuffle=False, pin_memory=True, num_workers=4)
                    }

                    model_ = CNN_Patch_Model(backbone, nr_classes=2).to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_.parameters()))
                    split_backbone_name = f'{backbone}-study_{study_size}-seed_{seed}-split_{fold+1}'

                    train_start_time = time.time()
                    trained_model = train_model(
                        model_, loaders_dict, loss_function, optimizer,
                        num_epochs=epochs, backbone_name=split_backbone_name, device=device
                    )
                    train_runtime = time.time() - train_start_time

                    val_metrics = evaluate_model(trained_model, loaders_dict['valid'], loss_function, device, f"Validation Fold {fold+1} Set")
                    test_metrics = evaluate_model(trained_model, loaders_dict['test'], loss_function, device, f"Test Fold {fold+1} Set")
                    bench_metrics = evaluate_model(trained_model, benchmarking_loader, loss_function, device, "Benchmarking Set")

                    result_entry = {
                        "model": backbone, "study_size": study_size,
                        "n_splits": n_splits, "seed": seed, "fold": fold + 1,
                        "train_runtime": train_runtime,
                        "val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"],
                        "val_runtime": val_metrics["runtime"],
                        "test_accuracy": test_metrics["accuracy"], "test_loss": test_metrics["loss"],
                        "test_runtime": test_metrics["runtime"],
                        "benchmark_accuracy": bench_metrics["accuracy"], "benchmark_loss": bench_metrics["loss"],
                        "benchmark_runtime": bench_metrics["runtime"],
                    }
                    all_results.append(result_entry)
                    print("-" * 25)

                    del model_, trained_model, optimizer
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\n--- All Benchmarking Runs Finished ---")
if all_results:
    results_df = pd.DataFrame(all_results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    models_str = "_".join(args.model_choices)
    nsplits_str = "_".join(map(str, args.n_splits))
    seeds_str = f"{args.seeds[0]}-{args.seeds[1]-1}"
    sizes_str = "_".join(map(str, args.study_sizes))

    filename = f"pcam_results_{models_str}_nsplits_{nsplits_str}_seeds_{seeds_str}_sizes_{sizes_str}_{timestamp}.parquet"
    parquet_path = OUTPUT_DIR / filename

    results_df.to_parquet(parquet_path)
    print(f"\nResults saved to {parquet_path}")
    print("\n--- Results Summary ---")
    print(results_df.to_string())
else:
    print("No results were generated.")

# %%