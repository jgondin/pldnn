import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import logging
import time

from torch.utils.data import DataLoader, TensorDataset

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Set device (GPU if available)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Define the Neural Network Model
class PseudoLabelNN(nn.Module,):
    def __init__(
        self,
        input_size,
        output_size,
        hidden=[400, 250, 150, 30, 10]
        ):
        super(PseudoLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.fc4 = nn.Linear(hidden[2], hidden[3])
        self.fc5 = nn.Linear(hidden[3], hidden[4])
        self.out = nn.Linear(hidden[4], output_size)
        
        self.norm1 =nn.BatchNorm1d(num_features=hidden[0])
        self.norm3 =nn.BatchNorm1d(num_features=hidden[2])
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.norm3(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.out(x)  # Return logits directly


if __name__ == '__main__':
    # Example loading of data
    logger.info("Loading dataset...")
    data = np.genfromtxt(
        "data/csv/feature_vectors_syscallsbinders_frequency_5_Cat.csv", 
        delimiter=",",
        skip_header=True)
    
    labels = data[:, -1].astype(int) - 1

    logger.info("Stratified train-valid Split")
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        data, labels, train_size=0.8, stratify=labels, random_state=42)
    
    logger.info("Normalized train and valid sets")
    norm = Normalizer('l2').fit(train_data)
    train_data = norm.transform(train_data)
    valid_data = norm.transform(valid_data)
    

    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {valid_data.shape}")

    # Convert labels to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long)


    # Initialize the model, optimizer, and loss function
    input_size = train_data.shape[1]
    output_size = 5
    model = PseudoLabelNN(
        input_size=input_size,
        output_size=output_size
        )
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # No need to apply softmax, raw logits required
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    logger.info("Model and optimizer initialized.")

    # Training the model
    num_epochs = 2000
    batch_size = 128
    lbl_samples = 1000  # Same as the pseudolabel sample size from original code

    logger.info("Starting training...")
    start_time = time.time()

    # Create TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        #   num_workers=5,
        drop_last=True, pin_memory=True)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False,
        #   num_workers=5
        )

    # Training loop using DataLoader
    for epoch in range(num_epochs):
        model.train()
        
        avg_loss = 0.0
        
        # Loop through the DataLoader for training data
        for batch_x, batch_y in train_loader:
            # Send to GPU if available
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

        avg_loss /= len(train_loader)  # Average loss over all batches
        
        # Evaluate on validation set
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                # Training accuracy (on entire training set)
                train_correct = 0
                train_total = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, train_pred = torch.max(outputs, 1)
                    train_total += batch_y.size(0)
                    train_correct += (train_pred == batch_y).sum().item()
                
                train_acc = train_correct / train_total

                # Validation accuracy (on entire validation set)
                valid_correct = 0
                valid_total = 0
                for batch_x, batch_y in valid_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, valid_pred = torch.max(outputs, 1)
                    valid_total += batch_y.size(0)
                    valid_correct += (valid_pred == batch_y).sum().item()

                valid_acc = valid_correct / valid_total

                log_result = f'Epoch [{epoch + 1}/{num_epochs}], '
                log_result += f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, '
                log_result += f'Valid Acc: {valid_acc:.4f}'
                logger.info(log_result)
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
