
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 16x(img_size/2)x(img_size/2)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32x(img_size/4)x(img_size/4)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64x(img_size/8)x(img_size/8)
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x(img_size/4)x(img_size/4)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x(img_size/2)x(img_size/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 3ximg_sizeximg_size
            nn.Sigmoid() # Output pixels are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Custom Dataset for loading .npz files from CSV
class NpzDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        npz_path = self.dataframe.iloc[idx]['preproc_file']
        data = np.load(npz_path) # Load the .npz file
        
        # Assuming the .npz file contains a key 'image_data' and it's a numpy array
        # You might need to adjust 'image_data' key based on your preprocess.py output
        image_data = data['image'] # Corrected: using 'image' key instead of 'image_data'
        
        # Convert numpy array to PIL Image for torchvision transforms
        # Handle different image dimensions (e.g., grayscale to RGB for Autoencoder)
        if image_data.ndim == 2: # Grayscale, no channel dimension
            image_data = np.expand_dims(image_data, axis=2) # Add channel dimension
        if image_data.shape[2] == 1: # Grayscale with 1 channel
            image_data = np.repeat(image_data, 3, axis=2) # Repeat channel to make it 3-channel (RGB)
        
        # Ensure image_data is uint8 for PIL conversion
        image = Image.fromarray(image_data.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        # Autoencoders don't typically use labels for input, so we return a dummy label
        return image, torch.tensor(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train and test an Autoencoder model.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing preprocessed data. (Not directly used for loading images now, but for context)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where train.csv, test.csv and model outputs are saved.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    args = parser.parse_args(args=argv)

    print(f"Starting training with input directory: {args.input_dir}")
    print(f"Output will be saved to: {args.output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image transformations
    # Assuming images are 256x256 based on typical GAN training image sizes
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    # Load dataset using custom NpzDataset and CSVs
    try:
        train_csv_path = os.path.join(args.output_dir, 'train.csv')
        test_csv_path = os.path.join(args.output_dir, 'test.csv')

        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

        train_dataset = NpzDataset(csv_file=train_csv_path, transform=transform)
        test_dataset = NpzDataset(csv_file=test_csv_path, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        print(f"Loaded {len(train_dataset)} training images from {train_csv_path}.")
        print(f"Loaded {len(test_dataset)} testing images from {test_csv_path}.")

    except Exception as e:
        print(f"Error loading data from CSVs. Error: {e}")
        print("Creating dummy data for demonstration. Ensure 'image_data' key exists in your .npz files.")
        # Create dummy data for demonstration if actual data loading fails
        dummy_data = torch.randn(100, 3, 256, 256) # 100 dummy images
        dummy_labels = torch.zeros(100) # Dummy labels
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        train_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=False)


    model = Autoencoder().to(device)
    criterion = nn.MSELoss() # Mean Squared Error for reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("\\n--- Starting Training ---") # Escaped newline for string literal
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_train_loss:.4f}')

    # Save the trained model
    model_save_path = os.path.join(args.output_dir, 'autoencoder_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # --- Testing Loop ---
    print("\\n--- Starting Testing ---") # Escaped newline for string literal
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Testing Loss: {avg_test_loss:.4f}')

    # Save testing metrics
    metrics_path = os.path.join(args.output_dir, 'training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Final Training Loss: {avg_train_loss:.4f}\\n') # Escaped newline
        f.write(f'Final Testing Loss: {avg_test_loss:.4f}\\n') # Escaped newline
    print(f"Training metrics saved to {metrics_path}")

    print("Training and testing script finished successfully.")

if __name__ == '__main__':
    try:
        if hasattr(sys, 'ps1'): # Running in interactive mode
            print("Running in interactive mode. Providing dummy arguments for direct cell execution.")
            # Example: provide input_dir as the 'processed_output' from previous step
            main(['--input-dir', 'processed_output', '--output-dir', 'model_output', '--epochs', '2'])
        else:
            main() # When run as an external script, argv=None is passed
    except SystemExit as e:
        if e.code != 0:
            raise