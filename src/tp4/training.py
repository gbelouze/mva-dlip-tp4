import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print as rprint
from torch.utils.data import DataLoader
from tp4.data import balancedDataLoader
from tqdm import tqdm


def epoch_repr(
    epoch, num_epochs, train_loss, accuracy, lr, test_loss=-1.0, test_accuracy=-1.0
):
    epoch_width = len(str(num_epochs))
    return f"Epoch {str(epoch).zfill(epoch_width)}/{num_epochs} \
| Average loss {train_loss:.2f} \
| Accuracy {100 * accuracy:04.1f}% \
| lr {lr:.1e} \
| Test loss {test_loss:.2f} \
| Test accuracy {100 * test_accuracy:04.1f}%"


def train(
    net,
    train_set,
    num_epochs=20,
    batch_size=30,
    lr=1e-3,
    schedule=False,
    parameters=None,
    test_set=None,
):
    train_loader = balancedDataLoader(train_set, batch_size)

    criterion = nn.CrossEntropyLoss()
    if parameters is None:
        parameters = net.parameters()
    optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-3)
    if schedule:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=10 * lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
        )

    loss_hist = []
    logging_frequency = 1 + (num_epochs - 1) // 10
    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        correct = 0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = net(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if schedule:
                scheduler.step()

            loss_hist.append(loss.item())
            sum_loss += loss.item()
            _, arg_pred = y_pred.max(1)
            correct += arg_pred.eq(y).sum().item()
        avg_loss = sum_loss / len(train_loader)
        accuracy = correct / len(train_set)
        if schedule:
            (lr,) = scheduler.get_last_lr()
        if test_set is not None:
            test_loss, test_accuracy = test(net, test_set, batch_size, progress=False)
            if epoch == num_epochs - 1 or epoch % logging_frequency == 0:
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
