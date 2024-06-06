from tqdm import tqdm
import torch


def train(epoch, model, train_loader, optimizer, criterion, loss_weights, device="cuda"):
    # Set the model to training mode
    model.train()

    # Initialize the running loss and correct predictions
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        exit1_out, exit2_out, exit3_out, final_out = model(images)
        
        # Calculate the loss for each exit
        loss1 = criterion(exit1_out, labels)
        loss2 = criterion(exit2_out, labels)
        loss3 = criterion(exit3_out, labels)
        loss_final = criterion(final_out, labels)

        # Calculate the weighted loss
        loss = sum(w * l for w, l in zip(loss_weights, [loss1, loss2, loss3, loss_final]))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate the total loss
        running_loss += loss.item()

        # Calculate the number of correct predictions
        _, predicted1 = torch.max(exit1_out.data, 1)
        _, predicted2 = torch.max(exit2_out.data, 1)
        _, predicted3 = torch.max(exit3_out.data, 1)
        _, predicted_final = torch.max(final_out.data, 1)

        total += labels.size(0)

        correct += (predicted1 == labels).sum().item() + (predicted2 == labels).sum().item() + (predicted3 == labels).sum().item() + (predicted_final == labels).sum().item()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / total
    avg_acc = correct / total

    # Store the training metrics
    train_metrics = {
        "epoch": epoch+1,
        "loss": avg_loss,
        "accuracy": avg_acc,
        "exit1_loss": loss1.item(),
        "exit2_loss": loss2.item(),
        "exit3_loss": loss3.item(),
        "final_loss": loss_final.item(),
        "exit1_accuracy": (predicted1 == labels).sum().item() / labels.size(0),
        "exit2_accuracy": (predicted2 == labels).sum().item() / labels.size(0),
        "exit3_accuracy": (predicted3 == labels).sum().item() / labels.size(0),
        "final_accuracy": (predicted_final == labels).sum().item() / labels.size(0),
    }

    return train_metrics
