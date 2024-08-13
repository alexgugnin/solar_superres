import torch

def train_step(model, data_loader, loss_func, optimizer, device):
  train_loss = 0
  nan_counter = 0
  for x, Y in data_loader:
    x = x.to(device)
    Y = Y.to(device)
    model.train()
    y_pred = model(x).to(device)

    loss = loss_func(y_pred, Y)
    if torch.isnan(loss).item():
      nan_counter+=1
      continue
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del loss, y_pred, x, Y

  train_loss /= (len(data_loader)-nan_counter)
  #print(f"Epoch Train_loss: {train_loss}")

  return train_loss

def valid_step(model, data_loader, loss_func, device):
  valid_loss = 0
  nan_counter = 0
  model.eval()
  with torch.inference_mode():
    for x, Y in data_loader:
      x = x.to(device)
      Y = Y.to(device)
      valid_pred = model(x)
      loss = loss_func(valid_pred, Y)
      if torch.isnan(loss).item():
        nan_counter+=1
        continue
      valid_loss += loss.item()
    
      del loss, valid_pred, x, Y

    valid_loss /= (len(data_loader)-nan_counter)
  #print(f"Validation_loss: {valid_loss}")

  return valid_loss

