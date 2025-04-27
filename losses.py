import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torchvision import models

class FourierSpectrumLoss(nn.Module):
    def __init__(self):
        super(FourierSpectrumLoss, self).__init__()
        # print("Initializing FourierSpectrumLoss...")
        
    def forward(self, pred, target):
        # Convert to frequency domain
        pred_fft = fft.fft2(pred, dim=(-2, -1))
        target_fft = fft.fft2(target, dim=(-2, -1))
        
        # Calculate magnitude spectrum
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # Compute MSE loss between magnitude spectrums
        return nn.MSELoss()(pred_magnitude, target_magnitude)

class PhaseConsistencyLoss(nn.Module):
    def __init__(self):
        super(PhaseConsistencyLoss, self).__init__()
        # print("Initializing PhaseConsistencyLoss...")
        
    def forward(self, pred, target):
        # Convert to frequency domain
        pred_fft = fft.fft2(pred, dim=(-2, -1))
        target_fft = fft.fft2(target, dim=(-2, -1))
        
        # Calculate phase
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Compute cosine similarity between phases
        phase_diff = torch.cos(pred_phase - target_phase)
        return 1 - torch.mean(phase_diff)

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for PyTorch, expects input in [0,1] range."""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, 1)

    def gaussian_window(self, window_size, sigma):
        # Create a tensor of indices
        x_values = torch.arange(window_size)
        center = window_size // 2
        variance = 2 * sigma**2
        # Calculate the Gaussian values in one go as a tensor
        gauss = torch.exp(-((x_values - center)**2) / float(variance))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device).type(img1.dtype)
        if channel != self.channel or window.device != img1.device:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class VGGLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    def __init__(self, layer='relu2_2'):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26
        }
        self.vgg = nn.Sequential(*[vgg[x] for x in range(self.layer_name_mapping[layer]+1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # x, y: [B, 3, H, W], range [0,1]
        return F.l1_loss(self.vgg(x), self.vgg(y))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2, gamma=0.1, delta=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.fourier_loss = FourierSpectrumLoss()
        self.phase_loss = PhaseConsistencyLoss()
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = VGGLoss(layer='relu2_2')
        self.alpha = alpha  # Weight for Fourier loss
        self.beta = beta    # Weight for Phase loss
        self.gamma = gamma  # Weight for SSIM loss
        self.delta = delta  # Weight for VGG loss
        
    def forward(self, pred, target):
        # Ensure inputs are in valid range
        pred = pred.clamp(-1, 1)
        target = target.clamp(-1, 1)
        
        # Base MSE loss
        mse = self.mse_loss(pred, target)
        
        # Normalize to [0,1] for perceptual losses
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        
        # Component losses with scaling
        fourier = self.fourier_loss(pred, target)
        phase = self.phase_loss(pred, target)
        ssim = self.ssim_loss(pred_norm, target_norm)
        vgg = self.vgg_loss(pred_norm, target_norm)
        
        # Scale losses to similar ranges
        total_loss = mse + \
                    self.alpha * torch.clamp(fourier, 0, 10) + \
                    self.beta * torch.clamp(phase, 0, 10) + \
                    self.gamma * ssim + \
                    self.delta * torch.clamp(vgg, 0, 10)
        
        return total_loss