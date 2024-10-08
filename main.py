import math
import os
from typing import Tuple, Union

import foolbox as fb
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from robustbench.data import load_cifar10
from robustbench.utils import clean_accuracy
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)

_ACTIVATION = {
    "relu": nn.ReLU,
    "swish": nn.SiLU,
}


class _Block(nn.Module):
    """
    WideResNet Block.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=True)
        # We manually pad to obtain the same effect as `SAME` (necessary when `stride` is different than 1).
        self.conv_0 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=True)
        self.conv_1 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride`.")
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)

        feature = out

        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out, feature


class _BlockGroup(nn.Module):
    """
    WideResNet block group.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """

    def __init__(
        self, num_blocks, in_planes, out_planes, stride, activation_fn=nn.ReLU
    ):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    activation_fn=activation_fn,
                )
            )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        features = []
        for block in self.block:
            x, feature = block(x)
            features.append(feature)
        return x, features


class WideResNet(nn.Module):
    """
    WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation_fn: nn.Module = nn.ReLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
    ):
        super().__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(
            num_input_channels,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer = nn.Sequential(
            _BlockGroup(
                num_blocks,
                num_channels[0],
                num_channels[1],
                1,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[1],
                num_channels[2],
                2,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[2],
                num_channels[3],
                2,
                activation_fn=activation_fn,
            ),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std

        features = []
        out = self.init_conv(out)
        for layer in self.layer:
            out, feature = layer(out)
            features.append(feature)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        logits = self.logits(out)

        if return_features:
            return logits, features
        else:
            return logits


def wideresnetwithswish(name, dataset="cifar10", num_classes=10, device="cuda"):
    """
    Returns suitable Wideresnet model with Swish activation function from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        device (str or torch.device): device to work on.
        dataset (str): dataset to use.
    Returns:
        torch.nn.Module.
    """
    name_parts = name.split("-")
    depth = int(name_parts[1])
    widen = int(name_parts[2])
    act_fn = name_parts[3]

    # print (f'WideResNet-{depth}-{widen}-{act_fn} uses normalization.')
    if "cifar100" in dataset:
        return WideResNet(
            num_classes=num_classes,
            depth=depth,
            width=widen,
            activation_fn=_ACTIVATION[act_fn],
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        )
    elif "svhn" in dataset:
        return WideResNet(
            num_classes=num_classes,
            depth=depth,
            width=widen,
            activation_fn=_ACTIVATION[act_fn],
            mean=SVHN_MEAN,
            std=SVHN_STD,
        )
    return WideResNet(
        num_classes=num_classes,
        depth=depth,
        width=widen,
        activation_fn=_ACTIVATION[act_fn],
    )


# hyperparameters
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # set device to cuda if available, otherwise cpu
TEACHER_CKPT_URL = "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_linf_wrn28-10.pt"
TEACHER_CKPT_PATH = "cifar10_linf_wrn28-10.pt"
BATCH_SIZE = 500  # number of samples per batch for training
EVAL_BATCH_SIZE = 500  # number of samples per batch for evaluation
NUM_WORKERS = 8  # number of subprocesses to use for data loading
MOMENTUM = 0.9  # momentum factor for sgd optimizer
WEIGHT_DECAY = 5e-4  # weight decay (l2 penalty) for optimization
NUM_EPOCHS = 200  # total number of training epochs
EVAL_EVERY = 10  # evaluate model every n epochs
ACCUMULATION_STEPS = 1  # number of gradient accumulation steps
RSA_RESCALE = True  # whether to rescale rsa loss
RSA_BLOCK_GROUP = "middle"  # which block group to use for rsa loss


def load_teacher_model():
    # create a wideresnet model with swish activation
    teacher_model = wideresnetwithswish("WideResNet-28-10-swish").to(DEVICE)

    # download the teacher checkpoint if it doesn't exist

    import requests

    if not os.path.exists(TEACHER_CKPT_PATH):
        print(f"Downloading teacher checkpoint from {TEACHER_CKPT_URL}")
        response = requests.get(TEACHER_CKPT_URL)
        with open(TEACHER_CKPT_PATH, "wb") as f:
            f.write(response.content)
        print(f"Checkpoint saved to {TEACHER_CKPT_PATH}")

    # load the teacher checkpoint
    teacher_ckpt = torch.load(
        TEACHER_CKPT_PATH, map_location=DEVICE, weights_only=True
    )

    # extract the state dictionary from the checkpoint
    state_dict = teacher_ckpt["model_state_dict"]

    # create a new state dictionary
    new_state_dict = {}

    # remove 'module.0.' prefix from keys in the state dictionary
    for key, value in state_dict.items():
        new_key = key.replace("module.0.", "")
        new_state_dict[new_key] = value

    # load the modified state dictionary into the teacher model
    teacher_model.load_state_dict(new_state_dict)

    # move the model to cuda
    teacher_model.cuda()

    # set the model to evaluation mode
    teacher_model.eval()

    # return the prepared teacher model
    return teacher_model


@torch.compile()
def compute_rsa_batched(X, Y):
    # compute representational similarity analysis (rsa) between two feature maps
    # x and y are expected to be of shape (batch_size, channels, height, width)
    batch_size = X.size(0)
    X = X.reshape(batch_size, -1)  # flatten feature maps
    Y = Y.reshape(batch_size, -1)  # flatten feature maps

    # center and normalize feature maps
    X_ = X - X.mean(dim=1, keepdim=True)
    Y_ = Y - Y.mean(dim=1, keepdim=True)
    X_ = X_ / X_.norm(dim=1, keepdim=True)
    Y_ = Y_ / Y_.norm(dim=1, keepdim=True)

    # compute rsa matrices
    X_rsa = 1 - X_ @ X_.T
    Y_rsa = 1 - Y_ @ Y_.T

    # compute mean squared error between rsa matrices
    return ((X_rsa - Y_rsa) ** 2).sum() / (batch_size * (batch_size - 1))


class teacher_augmented_cifar10(Dataset):
    def __init__(self, cifar10_dataset, teacher_features, teacher_outputs):
        # store the original cifar10 dataset
        self.dataset = cifar10_dataset
        # store the precomputed teacher features
        self.teacher_features = teacher_features
        # store the precomputed teacher outputs (logits)
        self.teacher_outputs = teacher_outputs
        # define data augmentation transforms
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        # return the length of the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # get the image and label from the original dataset
        img, label = self.dataset[idx]
        # apply data augmentation to the image
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        augmented_img = self.transform(img)
        augmented_img = self.transform(img)
        # return the augmented image, original label, teacher features, and teacher outputs
        return (
            augmented_img,
            label,
            self.teacher_features[idx],
            self.teacher_outputs[idx],
        )


def precompute_teacher_features(
    teacher_model, dataset, noise_level, rsa_block_group
):
    # set the teacher model to evaluation mode
    teacher_model.eval()
    # initialize lists to store all features and outputs
    all_features = []
    all_outputs = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset.transform = transform

    # create a dataloader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # disable gradient computation for efficiency
    with torch.no_grad():
        # iterate over the dataloader with a progress bar
        for inputs, _ in tqdm(dataloader, desc="Precomputing teacher features"):
            # move inputs to the specified device (e.g., GPU)
            inputs = inputs.to(DEVICE)
            # forward pass through the teacher model, returning both outputs and features
            outputs, features = teacher_model(inputs, return_features=True)

            # select the appropriate block index based on the rsa_block_group
            if rsa_block_group == "early":
                block_index = 0
            elif rsa_block_group == "middle":
                block_index = 1
            elif rsa_block_group == "late":
                block_index = -1
            else:
                # raise an error if an invalid rsa_block_group is provided
                raise ValueError("Invalid RSA_BLOCK_GROUP value")

            # extract the selected features from the specified block
            selected_features = features[block_index][block_index]

            # generate gaussian noise with the same shape as the selected features
            noise = torch.randn_like(selected_features) * noise_level
            # add the generated noise to the selected features
            noisy_features = selected_features + noise

            # append the noisy features and outputs to their respective lists
            all_features.append(noisy_features.cpu())
            all_outputs.append(outputs.cpu())

    # concatenate all features and outputs along the batch dimension
    all_features = torch.cat(all_features, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

    # return the precomputed features and outputs
    return all_features, all_outputs


def cache_teacher_data(
    teacher_model, dataset, noise_level, rsa_block_group, cache_dir
):
    # construct the cache file path using the noise level
    cache_file = os.path.join(
        cache_dir, f"teacher_data_noise_{noise_level:.4f}.pt"
    )

    # check if the cache file already exists
    if os.path.exists(cache_file):
        # if it exists, load the cached data
        print(f"loading cached teacher data from {cache_file}")
        cached_data = torch.load(cache_file)
        # return the features and outputs from the cached data
        return cached_data["features"], cached_data["outputs"]

    # if the cache file doesn't exist, precompute the teacher data
    print("precomputing and caching teacher data...")
    # call the precompute_teacher_features function to get features and outputs
    all_features, all_outputs = precompute_teacher_features(
        teacher_model, dataset, noise_level, rsa_block_group
    )

    # create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    # save the computed features and outputs to the cache file
    torch.save({"features": all_features, "outputs": all_outputs}, cache_file)

    # print a message indicating where the cached data was saved
    print(f"cached teacher data saved to {cache_file}")
    # return the computed features and outputs
    return all_features, all_outputs


def create_augmented_dataloader(
    dataset, teacher_features, teacher_outputs, batch_size, num_workers
):
    # create an augmented dataset using the teacher_augmented_cifar10 class
    augmented_dataset = teacher_augmented_cifar10(
        dataset, teacher_features, teacher_outputs
    )
    # create and return a DataLoader for the augmented dataset
    return DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate_model(model, x_test, y_test):
    # evaluate model on clean and adversarial examples
    model.eval()
    # compute accuracy on clean examples
    clean_acc = clean_accuracy(
        model, x_test, y_test, device=DEVICE, batch_size=EVAL_BATCH_SIZE
    )

    # create foolbox model for adversarial attacks
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD()  # use projected gradient descent attack
    epsilons = [8 / 255]  # perturbation budget
    robust_correct = 0
    for i in range(0, len(x_test), EVAL_BATCH_SIZE):
        x_batch = x_test[i : i + EVAL_BATCH_SIZE].to(DEVICE)
        y_batch = y_test[i : i + EVAL_BATCH_SIZE].to(DEVICE)

        # perform adversarial attack
        _, _, success = attack(fmodel, x_batch, y_batch, epsilons=epsilons)
        robust_correct += (1 - success.float()).sum().item()

    # compute accuracy on adversarial examples
    robust_accuracy = robust_correct / len(x_test)
    return clean_acc, robust_accuracy


def train_epoch(
    student_model,
    augmented_trainloader,
    optimizer,
    criterion,
    scaler,
    beta,
    rsa_scale,
):
    # set student model to training mode
    student_model.train()
    # initialize running loss
    running_loss = 0.0
    # initialize list to store rsa losses
    rsa_losses = []
    # initialize list to store student losses
    student_losses = []

    # zero out the gradients
    optimizer.zero_grad()

    # iterate over batches in the trainloader
    for i, (
        augmented_inputs,
        labels,
        teacher_features,
        teacher_outputs,
    ) in enumerate(augmented_trainloader):
        # move inputs and labels to the specified device
        augmented_inputs, labels = (
            augmented_inputs.to(DEVICE),
            labels.to(DEVICE),
        )
        teacher_features, teacher_outputs = (
            teacher_features.to(DEVICE),
            teacher_outputs.to(DEVICE),
        )
        # use automatic mixed precision for faster training
        with autocast(device_type="cuda"):
            # get student model outputs and features
            student_outputs, student_features = student_model(
                augmented_inputs, return_features=True
            )

            teacher_labels = torch.argmax(teacher_outputs, dim=1)

            student_loss = criterion(student_outputs, teacher_labels)
            student_losses.append(student_loss.item())

            # initialize rsa loss
            rsa_loss = 0
            # determine which block group to use for rsa loss
            if RSA_BLOCK_GROUP == "early":
                block_index = 0
            elif RSA_BLOCK_GROUP == "middle":
                block_index = 1
            elif RSA_BLOCK_GROUP == "late":
                block_index = -1
            else:
                raise ValueError("Invalid RSA_BLOCK_GROUP value")

            # compute rsa loss for the selected feature group
            rsa_loss = compute_rsa_batched(
                student_features[block_index][block_index],
                teacher_features,
            )

            # scale rsa loss
            rsa_loss *= rsa_scale
            # store rsa loss for later averaging
            rsa_losses.append(rsa_loss.item())

            # compute total loss as weighted sum of student loss and rsa loss
            loss = (1 - beta) * student_loss + beta * rsa_loss
            # divide loss by accumulation steps for gradient accumulation
            loss = loss / ACCUMULATION_STEPS

        # scale the loss and compute gradients
        scaler.scale(loss).backward()

        # perform optimizer step and scaler update every ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # unscale gradients for clipping
            scaler.unscale_(optimizer)

            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm=1.0
            )

            # perform optimizer step and update scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # accumulate running loss
        running_loss += loss.item() * ACCUMULATION_STEPS

    # compute average loss over the epoch
    avg_loss = running_loss / len(augmented_trainloader)
    # compute average rsa loss over the epoch
    avg_rsa_loss = sum(rsa_losses) / len(rsa_losses)
    # compute average student loss over the epoch
    avg_student_loss = sum(student_losses) / len(student_losses)

    # rescale rsa loss if enabled
    if RSA_RESCALE:
        new_rsa_scale = 1.0 / avg_rsa_loss if avg_rsa_loss > 0 else 1.0
    else:
        new_rsa_scale = rsa_scale

    # return average loss, average rsa loss, average student loss, and new rsa scale
    return avg_loss, avg_rsa_loss, avg_student_loss, new_rsa_scale


def train_and_evaluate(beta, noise_level, learning_rate):
    # initialize wandb for experiment tracking
    wandb.init(
        project="roadmap-robustness-cifar10-feature-distillation",
        name=f"lr_{learning_rate}_beta_{beta}_noise_{noise_level}",
        config={
            "beta": beta,
            "noise_level": noise_level,
            "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "num_epochs": NUM_EPOCHS,
        },
    )

    # load pre-trained teacher model
    teacher_model = load_teacher_model()

    # Load CIFAR-10 dataset without augmentation
    trainset = datasets.CIFAR10(root="./data", train=True, download=True)
    # Load and cache teacher data
    cache_dir = "./teacher_cache"
    teacher_features, teacher_outputs = cache_teacher_data(
        teacher_model, trainset, noise_level, RSA_BLOCK_GROUP, cache_dir
    )
    # Create new dataloader with augmented data and cached teacher features/outputs
    augmented_trainloader = create_augmented_dataloader(
        trainset, teacher_features, teacher_outputs, BATCH_SIZE, NUM_WORKERS
    )

    # initialize student model
    student_model = wideresnetwithswish("WideResNet-28-10-swish").to(DEVICE)
    student_model = torch.compile(student_model, mode="max-autotune")

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    scaler = GradScaler(device=DEVICE)

    # training loop
    rsa_scale = 1.0
    for epoch in tqdm(range(NUM_EPOCHS), desc="training epochs"):
        loss, rsa_loss, student_loss, rsa_scale = train_epoch(
            student_model,
            augmented_trainloader,
            optimizer,
            criterion,
            scaler,
            beta,
            rsa_scale,
        )
        scheduler.step()

        # log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss,
                "rsa_loss": rsa_loss,
                "student_loss": student_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_scale": scaler.get_scale(),
                "rsa_scale": rsa_scale,
            }
        )

        # evaluate model periodically
        if (epoch + 1) % EVAL_EVERY == 0:
            x_test, y_test = load_cifar10(n_examples=10000)
            clean_acc = clean_accuracy(
                student_model,
                x_test,
                y_test,
                device=DEVICE,
                batch_size=EVAL_BATCH_SIZE,
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "test_acc": clean_acc,
                }
            )

    # final evaluation
    x_test, y_test = load_cifar10(n_examples=10000)

    import time

    start_time = time.time()
    clean_acc, robust_acc = evaluate_model(student_model, x_test, y_test)
    end_time = time.time()
    evaluation_time = end_time - start_time
    print(
        f"Student evaluation at end of training took {evaluation_time:.2f} seconds"
    )
    wandb.log(
        {"final_clean_accuracy": clean_acc, "final_robust_accuracy": robust_acc}
    )
    wandb.finish()
    return student_model


import argparse


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the model")
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Beta value for loss calculation",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Noise level for feature distillation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.4, help="Initial learning rate"
    )
    args = parser.parse_args()

    print(f"Training with beta={args.beta}, noise_level={args.noise_level}")
    train_and_evaluate(args.beta, args.noise_level, args.learning_rate)


if __name__ == "__main__":
    main()
