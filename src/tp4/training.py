import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print as rprint
from torch.utils.data import DataLoader
from tp4.data import balancedDataLoader
from tqdm import tqdm


def epoch_repr(
    epoch, num_epochs, train_loss, accuracy, lr, test_loss=None, test_accuracy=None
) -> str:
    epoch_width = len(str(num_epochs))
    ret = f"Epoch {str(epoch).zfill(epoch_width)}/{num_epochs} \
| Average loss {train_loss:.2f} \
| Accuracy {100 * accuracy:04.1f}% \
| lr {lr:.1e}"
    if test_loss is not None:
        ret += f" | Test loss {test_loss:.2f}"
    if test_accuracy is not None:
        ret += f" | Test accuracy {100 * test_accuracy:04.1f}%"
    return ret


def train_stepLR(
    net,
    train_set,
    num_epochs=20,
    batch_size=30,
    lrs=(1e-3, 1e-2),
    parameters=None,
    test_set=None,
    device="cpu",
):
    lr_min, lr_max = lrs
    train_loader = balancedDataLoader(train_set, batch_size)

    criterion = nn.CrossEntropyLoss()
    if parameters is None:
        parameters = net.parameters()

    optimizer = optim.SGD(parameters, lr=lr_max, momentum=0.9)
    gamma = math.exp(math.log(lr_min / lr_max) / (num_epochs // 5))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

    loss_hist = []
    logging_frequency = 1 + (num_epochs - 1) // 20
    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        correct = 0
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = net(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            sum_loss += loss.item()
            _, arg_pred = y_pred.max(1)
            correct += arg_pred.eq(y).sum().item()

        scheduler.step()
        avg_loss = sum_loss / len(train_loader)
        accuracy = correct / len(train_set)
        (lr,) = scheduler.get_last_lr()

        if epoch == num_epochs - 1 or epoch % logging_frequency == 0:
            test_loss, test_accuracy = None, None
            if test_set is not None:
                test_loss, test_accuracy = test(
                    net, test_set, batch_size, progress=False
                )
            rprint(
                epoch_repr(
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                    accuracy,
                    lr,
                    test_loss,
                    test_accuracy,
                )
            )

    return loss_hist


def train_one_cycle(
    net,
    train_set,
    num_epochs=20,
    batch_size=30,
    lrs=(1e-3, 1e-2),
    parameters=None,
    test_set=None,
    device="cpu",
):
    lr_min, lr_max = lrs
    train_loader = balancedDataLoader(train_set, batch_size)

    criterion = nn.CrossEntropyLoss()
    if parameters is None:
        parameters = net.parameters()

    optimizer = optim.SGD(parameters, lr=lr_min, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max, total_steps=len(train_loader) * num_epochs
    )

    loss_hist = []
    logging_frequency = 1 + (num_epochs - 1) // 20
    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        correct = 0
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = net(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_hist.append(loss.item())
            sum_loss += loss.item()
            _, arg_pred = y_pred.max(1)
            correct += arg_pred.eq(y).sum().item()
        avg_loss = sum_loss / len(train_loader)
        accuracy = correct / len(train_set)
        (lr,) = scheduler.get_last_lr()
        if test_set is not None:
            if epoch == num_epochs - 1 or epoch % logging_frequency == 0:
                test_loss, test_accuracy = test(
                    net, test_set, batch_size, progress=False
                )
                rprint(
                    epoch_repr(
                        epoch + 1,
                        num_epochs,
                        avg_loss,
                        accuracy,
                        lr,
                        test_loss,
                        test_accuracy,
                    )
                )
        elif epoch == num_epochs - 1 or epoch % logging_frequency == 0:
            rprint(epoch_repr(epoch + 1, num_epochs, avg_loss, accuracy, lr))

    return loss_hist


def predict(net, data_set, batch_size=10):
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(data_set, batch_size)
    net.eval()

    predictions = []
    sum_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(data_loader):
            y_pred = net(X)
            loss = criterion(y_pred, y)
            sum_loss += loss.item()
            _, arg_pred = y_pred.max(1)
            predictions.append(arg_pred.numpy())
            correct += arg_pred.eq(y).sum().item()

    avg_loss = sum_loss / len(data_loader)
    accuracy = correct / len(data_set)

    return np.concatenate(predictions), avg_loss, accuracy


def test(net, test_set, batch_size=10, progress=True):
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_set, batch_size)
    net.eval()

    sum_loss = 0.0
    correct = 0
    with torch.no_grad():
        tqdm_loader = tqdm(test_loader) if progress else test_loader
        for X, y in tqdm_loader:
            y_pred = net(X)
            loss = criterion(y_pred, y)
            sum_loss += loss.item()
            _, arg_pred = y_pred.max(1)
            correct += arg_pred.eq(y).sum().item()

    avg_loss = sum_loss / len(test_loader)
    accuracy = correct / len(test_set)

    return avg_loss, accuracy


def find_lr(
    net: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 50,
    batch_size: int = 10,
    lr_min=1e-5,
    lr_max=1e0,
):

    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = optim.SGD(net.parameters(), lr=lr_min, momentum=0.9)
    gamma = np.exp(np.log(lr_max / lr_min) / num_epochs)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: gamma
    )
    net.train()
    loss_history = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        running_count = 0
        for i, (X, y) in enumerate(dataloader):

            y_pre = net(X).squeeze()
            loss = criterion(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_count += 1

            if i % max(1, len(dataloader) // 10) == 0:
                loss_history.append(running_loss / running_count)
                running_loss = 0
                running_count = 0
        scheduler.step()

    return loss_history, (lr_min, scheduler.get_last_lr())
