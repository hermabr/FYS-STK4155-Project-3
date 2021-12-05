import time
import torch
import numpy as np
from data.fall import FallData

#  from cnn import CNNModel


import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.transforms import transforms

from cnn.model import Classifier

#  from cnn.model import CNN as Classifier

#  from cnn_model import CNNModel as Classifier
from cnn.metrics import (
    _one_sample_positive_class_precisions,
    calculate_per_class_lwlrap,
)

#  def train_model(, train_transforms):
def train(data):
    num_epochs = 80
    #  batch_size = 64
    test_batch_size = 256
    lr = 3e-3
    eta_min = 1e-5
    t_max = 10
    num_classes = 2

    #  num_classes = y_train.shape[1]
    #
    #  x_trn, x_val, y_trn, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED)
    #
    #  train_dataset = FATTrainDataset(x_trn, y_trn, train_transforms)
    #  valid_dataset = FATTrainDataset(x_val, y_val, train_transforms)
    #
    #  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #  valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False)
    train_loader = data.train_loader
    valid_loader = data.test_loader

    #  model = Classifier(num_classes=num_classes).cuda()
    model = Classifier(num_classes=2)
    #  criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    best_epoch = -1
    best_lwlrap = 0.0
    #  mb = master_bar(range(num_epochs))

    for epoch in range(num_epochs):
        print("Epoch", epoch)
        start_time = time.time()
        model.train()
        avg_loss = 0.0

        for x_batch, y_batch in train_loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            #  preds = model(x_batch.cuda())
            #  loss = criterion(preds, y_batch.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        #  valid_preds = np.zeros((len(x_val), num_classes))
        valid_preds = np.zeros((len(data.x_test), num_classes))
        avg_val_loss = 0.0

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            preds = model(x_batch).detach()
            loss = criterion(preds, y_batch)
            #  preds = model(x_batch.cuda()).detach()
            #  loss = criterion(preds, y_batch.cuda())

            preds = torch.sigmoid(preds)
            valid_preds[
                i * test_batch_size : (i + 1) * test_batch_size
            ] = preds.cpu().numpy()

            avg_val_loss += loss.item() / len(valid_loader)

        score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time

        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), "weight_best.pt")

    return {
        "best_epoch": best_epoch,
        "best_lwlrap": best_lwlrap,
    }


def main():
    from cnn.test import main_test

    main_test()
    #  data = FallData(test_size=0.2)
    #  result = train(data)
