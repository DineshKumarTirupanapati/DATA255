import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SuperResolutionDataset(Dataset):
    def __init__(self, folder_path, lr_size=(64, 64), hr_size=(256, 256), transform=None):
        """
        Args:
            folder_path (string): Directory with all the source images.
            lr_size (tuple): Size for low-resolution images.
            hr_size (tuple): Size for high-resolution images.
            transform (callable, optional): Optional transform to be applied on HR images.
        """
        self.image_paths = sorted([
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".png")
        ])
        
        self.transform = transform

        # First transform to HR size (256x256)
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        # Then downscale to LR size (64x64 by default)
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # First get HR image (256x256)
        hr_image = self.hr_transform(image)
        
        # Apply data augmentation if provided (only to HR images)
        if self.transform is not None:
            # Convert tensor back to PIL image for transforms
            hr_pil = transforms.ToPILImage()(hr_image)
            hr_pil = self.transform(hr_pil)
            hr_image = transforms.ToTensor()(hr_pil)
        
        # Then create LR image by downscaling the augmented HR image
        lr_image = self.lr_transform(transforms.ToPILImage()(hr_image))
        
        # Normalize to [-1, 1] range if your model expects this
        hr_image = hr_image * 2 - 1
        lr_image = lr_image * 2 - 1

        return lr_image, hr_image