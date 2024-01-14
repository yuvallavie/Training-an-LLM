"""Training methods


This file includes the training function and other utilities needed for 
quick training of models.
"""

from tqdm import tqdm
from datetime import datetime
from colorama import init, Fore, Back, Style
from torch import save, __version__, cuda, version
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
            for batch in tqdm(loaders[phase], desc=phase[0]):
                # Extract the data and move to device
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs.squeeze(), labels.float())

                # Accumulate loss
                running_loss += loss.item()

                if (phase == "train"):
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                # Calculate accuracy
                # Threshold at 0.5 for binary classification
                predicted = (outputs.squeeze() >= 0.5).float()
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
            print(
                Fore.RED + f"{phase} loss: {process_data[phase]['loss'][-1]:.4f}")
            print(
                Fore.YELLOW + f"{phase} accuracy: {process_data[phase]['accuracy'][-1]:.2f}%")

    # Print the training results
    print("Results:")
    print("-"*15)
    print("Best accuracy:", best_accuracy)
    print("Epoch:", best_epoch)
