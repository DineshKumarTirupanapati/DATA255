import torch
import torch.nn as nn
import torch.fft

# === Fourier-based CNN ===
class FourierCNN(nn.Module):
    def __init__(self):
        super(FourierCNN, self).__init__()
        # print("Initializing FourierCNN...")
        # Input layer: 3 channels (RGB) -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Hidden layer: 64 channels -> 64 channels
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # Output layer: 64 channels -> 3 channels (RGB)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
        # print("FourierCNN initialized successfully")

    def forward(self, x):
        # x shape: [batch_size, 3, height, width]
        
        # Apply FFT to each channel separately
        freq = torch.fft.fft2(x)  # Shape: [batch_size, 3, height, width]
        
        # Calculate magnitude spectrum for each channel
        freq_abs = torch.abs(freq)  # Shape: [batch_size, 3, height, width]
        
        # Find maximum magnitude across spatial dimensions for each channel
        freq_max = freq_abs.amax(dim=(-2, -1), keepdim=True)  # Shape: [batch_size, 3, 1, 1]
        
        # Normalize magnitude spectrum
        freq_amp = freq_abs / (freq_max + 1e-8)  # Shape: [batch_size, 3, height, width]
        
        # Ensure frequency amplitude has same number of channels as input
        if freq_amp.shape[1] != x.shape[1]:
            freq_amp = freq_amp.repeat(1, x.shape[1], 1, 1)  # Repeat for all 3 channels
        
        # Combine spatial and frequency domain information
        x = x + freq_amp
        
        # Process through convolutional layers
        x = self.relu(self.conv1(x))  # 3 -> 64 channels
        x = self.relu(self.conv2(x))  # 64 -> 64 channels
        x = self.conv3(x)  # 64 -> 3 channels
        
        return x


# === U-Net Style Diffusion Refiner ===
class DiffusionUNet(nn.Module):
    def __init__(self):
        super(DiffusionUNet, self).__init__()
        # print("Initializing DiffusionUNet...")

        # Encoder (downsampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path with 4x total upscaling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128 -> 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
        # print("DiffusionUNet initialized successfully")

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x  # Residual output

# === Combined ResDiff Model ===
class ResDiffModel(nn.Module):
    def __init__(self):
        super(ResDiffModel, self).__init__()
        # print("Initializing ResDiffModel...")
        self.coarse_net = FourierCNN()
        self.refiner = DiffusionUNet()
        # print("ResDiffModel initialized successfully")
        # print(f"ResDiffModel parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        # Step 1: Get coarse prediction using Fourier CNN
        coarse = self.coarse_net(x)

        # Step 2: Refine with U-Net as residual correction
        residual = self.refiner(coarse)

        # Step 3: Upscale coarse prediction to match residual size
        # Use interpolation to upscale coarse from 64x64 to 256x256
        coarse_upscaled = torch.nn.functional.interpolate(
            coarse, 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )

        # Step 4: Combine (ResDiff-style)
        output = coarse_upscaled + residual  # final high-resolution output

        return output