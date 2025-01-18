import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter


def train_model(model, train_data, test_data, val_data, optimizer, loss_fn, device, 
                epochs=1001, val_interval=50, save_path=None, log_path=None):
    """ Train the model and save the best and final model """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)
    print(f"Logs saving at {log_path}")
    print(f"Models saving at {save_path}")

    model = model.to(device)

    train_stats = []
    
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for _, (data, target) in enumerate(train_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(train_data)
        writer.add_scalar('training loss', training_loss, epoch)

        torch.save(model.state_dict(), save_path + f"{epoch}.pth")
        
        model.eval()
        with torch.no_grad():
            test_loss = 0
            val_loss = 0
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output.squeeze(1), target).item()
            for data, target in val_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output.squeeze(1), target).item()
            test_loss /= len(test_data)
            val_loss /= len(val_data)
            writer.add_scalar('test loss', test_loss, epoch)
            writer.add_scalar('validation loss', val_loss, epoch)

        train_stats.append([epoch, training_loss, test_loss, val_loss])

        if epoch % val_interval == 0:
            print(f"Epoch: {epoch}, Training Loss: {training_loss}, Test Loss: {test_loss}, Val Loss: {val_loss}")

    writer.close()

    csv_path = os.path.join(save_path, 'train_stats.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Test Loss', 'Validation Loss'])
        writer.writerows(train_stats)
    
    print(f"Logs saved at {log_path}")
    print(f"Models saved at {save_path}")
    print(f"Training stats saved at {csv_path}")
