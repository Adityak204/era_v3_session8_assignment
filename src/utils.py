from tqdm import tqdm
import torch
import torch.nn as nn
import multiprocessing
from src.support_model import CIFAR10Dataset


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Compute accuracy
        train_loss += loss.item()  # accumulate batch loss
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_description(desc=f"loss={loss.item():.4f} batch_id={batch_idx}")

    # Compute average loss and accuracy for the epoch
    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total

    print(
        f"\nEpoch {epoch}: Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
    )
    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item()

            # Compute accuracy
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()

    # Compute average loss and accuracy for the test set
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy


# Load CIFAR-10 Dataset with Albumentations
def load_cifar10_with_albumentations(
    device, use_cuda, config_dict, albumentations_transform
):
    torch.manual_seed(config_dict["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(config_dict["seed"])
    batch_size = config_dict["batch_size"]
    kwargs = (
        {"num_workers": multiprocessing.cpu_count(), "pin_memory": True}
        if use_cuda
        else {}
    )

    train_dataset = CIFAR10Dataset(
        root="./data", train=True, transform=albumentations_transform
    )
    test_dataset = CIFAR10Dataset(
        root="./data", train=False, transform=albumentations_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, test_loader
