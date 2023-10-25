import torch

def train_step(model, data_loader, loss_func, optimizer, device) -> float:
    '''
    Defines a single training step in frames of one epoch. Returns a loss,
    which could be used for evaluating the efficiency of the approach.
    '''
    train_loss = 0
    for batch, (x, Y) in enumerate(data_loader):
        x = x.to(device)
        Y = Y.to(device)
        model.train()
        y_pred = model(x).to(device)

        loss = loss_func(y_pred, Y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)

    return train_loss

def valid_step(model, data_loader, loss_func, device) -> float:
    '''
    Defines a single validation step in frames of one epoch. Returns a loss,
    which could be used for evaluating the efficiency of the approach.
    '''
    valid_loss = 0
    model.eval()
    with torch.inference_mode():
        for x, Y in data_loader:
            x = x.to(device)
            Y = Y.to(device)
            valid_pred = model(x)
            valid_loss += loss_func(valid_pred, Y)

    valid_loss /= len(data_loader)

    return valid_loss
