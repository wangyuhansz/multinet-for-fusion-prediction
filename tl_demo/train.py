import os
import torch
from torch.utils.tensorboard import SummaryWriter


def train_model(model, train_data, val_data, optimizer, loss_fn, device, 
                epochs=1000, val_interval=50, save_path=None, log_path=None):
    writer = SummaryWriter(log_dir=log_path)
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output.squeeze(1), target)
            writer.add_scalar('training loss, batch', loss.item(), epoch * len(train_data) + batch_idx)
            loss.backward()
            optimizer.step()
        
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for data, target in val_data:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += loss_fn(output.squeeze(1), target).item()
                val_loss /= len(val_data)
            # print(f"Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss}")
            writer.add_scalar('validation loss', val_loss, epoch)
    
    writer.close()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
