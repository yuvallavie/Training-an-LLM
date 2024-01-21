"""Training methods


This file includes the training function and other utilities needed for 
quick training of models.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from colorama import init, Fore, Back, Style
from torch import save, __version__, cuda, version, set_grad_enabled, argmax
# Initialize Colorama
init(autoreset=True)


def train(model, optimizer, loaders, criterion, num_epochs, device):
    """Train a model using gradient based methods.

    This function trains the model by minimizing the criterion
    based on the data inside the loaders and the optimizer.
    It expects the loaders to be a dictionary with "train" and "validate" keys
    each consisting of a pytorch dataloader.
    """

    # Print the starting information.
    print("-" * 20)
    print("Training information")
    print("-" * 20)
    print("Torch version:", __version__)
    print('CUDA available:', cuda.is_available())
    print('CUDA version:', version.cuda)
    print("Optimizer:", optimizer.__class__.__name__)
    print("Epochs:", num_epochs)
    print("Batch size:", loaders["train"].batch_size)
    print("Train set size:", len(loaders["train"]))
    print("Validation set size:", len(loaders["validate"]))
    print("Device:", device)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    # Variables to keep track of the accuracy
    # loss and weights.
    process_data = {
        "train": {
            "loss": [],
            "accuracy": []
        },
        "validate": {
            "loss": [],
            "accuracy": []
        }
    }
    best_accuracy = 0.0
    best_weights = None
    # Run over each epoch and either train or validate.
    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print("-" * 30)
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

        # Run over both phases each epoch.
        for phase in loaders.keys():
            if (phase == "train"):
                model.train()
            else:
                model.eval()

            # Statistics values
            correct = 0
            total = 0
            running_loss = 0.0

            # Run over all examples in the batches.
            for inputs, labels in tqdm(loaders[phase], desc=phase[0]):
                # Enable the gradients only in the training phase.
                with set_grad_enabled(phase == 'train'):
                    # Extract the data and move to device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs.squeeze(), labels)

                    # Accumulate loss
                    running_loss += loss.item()

                    # Backpropogate the errors
                    # and optimize only in the
                    # training phase.
                    if (phase == "train"):
                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()

                # Calculate accuracy
                # Threshold at 0.5 for binary classification
                # predicted = argmax(outputs.squeeze(), dim=1)
                predicted = model.predict(logits=outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate the average loss.
            phase_loss = running_loss / len(loaders[phase])
            process_data[phase]["loss"].append(phase_loss)
            # Calculate accuracy of this phase.
            phase_accuracy = 100 * correct / total
            process_data[phase]["accuracy"].append(phase_accuracy)

            if (phase == "validate" and phase_accuracy > best_accuracy):
                # Update the best weights.
                best_epoch = epoch
                best_accuracy = phase_accuracy
                best_weights = model.state_dict().copy()
                # Save the model with the best weights.
                # We do every epoch in case the training
                # process halts for an unknown reason.
                save(best_weights, 'best_model_weights.pth')

        for phase in loaders.keys():
            # Print the epoch's statistics.
            # We print it red if its worse than the last round
            # and green if its better.
            if (epoch > 0):
                loss_color = Fore.GREEN if process_data[phase][
                    'loss'][-1] < process_data[phase]['loss'][-2] else Fore.RED
                acc_color = Fore.GREEN if process_data[phase][
                    'accuracy'][-1] > process_data[phase]['accuracy'][-2] else Fore.RED
            else:
                loss_color = acc_color = Fore.WHITE
            # Print them.
            print(
                loss_color + f"{phase} loss: {process_data[phase]['loss'][-1]:.4f}")
            print(
                acc_color + f"{phase} accuracy: {process_data[phase]['accuracy'][-1]:.2f}%")
        # Print the currently best statistics
        print(
            f"- Best epoch: {best_epoch + 1} , best accuracy: {best_accuracy}")

    # Print the training results
    print("Results:")
    print("-"*15)
    print("Best accuracy:", best_accuracy)
    print("Epoch:", best_epoch + 1)
    # Return the training results for analysis
    return process_data


def predict(model, test_loader):
    """Make a prediction on the entire test set

    This function accepts a model and expects it to have
    a function called "predict()" which accepts as input
    the features and returns classes.
    """

    # Make sure that the model has a predict() function.
    if not hasattr(model, "predict"):
        print("The model does not have a predict() function")
        return None, None

    # Get the device the model is using (CUDA/CPU).
    device = next(model.parameters()).device
    # Create two lists for the true values and the
    # predicted values.
    predicted_labels, true_labels = [], []
    # Iterate over each batch and predict on
    # its features, save their labels.
    for features, labels in tqdm(test_loader, desc="p"):
        # Send them to the same device.
        features, labels = features.to(device), labels.to(device)
        # Predict using the model.
        preds = model.predict(features)
        # Add the true labels and the
        # predicted labels to lists.
        predicted_labels.extend(preds.tolist())
        true_labels.extend(labels.tolist())
    # Return the lists.
    return predicted_labels, true_labels


def display_classification_results(y_true, y_pred, labels=None):
    """Display the results of the classification task"""
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Use labels for axis if provided, otherwise use unique values from y_true
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
