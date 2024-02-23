import torch
from torch import optim
import torch.nn.functional as F
from ser.constants import RESULTS_DIR

def train(params, device, train_dataloader, validation_dataloader, model):
    #set up params
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    #train 
    for epoch in range(params["epochs"]):
        _train_batch(epoch, train_dataloader, model, optimizer, device)
        _val_batch(epoch, validation_dataloader, model, device) # validates after each epoch as opposed to after each batch

    # save the model
    torch.save(model, RESULTS_DIR/f"{params['name']}/model.pt")

def _train_batch(epoch, train_dataloader, model, optimizer, device):
    '''
    Train the model for one epoch
    *Underscore means this function is not meant to be called directly
    '''

    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(train_dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )

@torch.no_grad() # this decorator is used to disable gradient calculation
def _val_batch(epoch, validation_dataloader, model, device):
    '''
    Validate the model for one epoch
    *Underscore means this function is not meant to be called directly
    '''
    # record highest validation accuracy
    highest_accuracy = 0
    best_epoch = 0

    for images, labels in validation_dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(validation_dataloader.dataset)
    val_acc = correct / len(validation_dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

        