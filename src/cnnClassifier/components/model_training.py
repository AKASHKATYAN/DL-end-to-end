import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path
import tensorflow as tf
import numpy as np
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def get_base_model(self):
        """Load .h5 TensorFlow model"""
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        print(f"✓ Loaded model: {self.config.updated_base_model_path}")
        
    def train_validator_generator(self):
        if self.config.params_is_augmentation:
            train_transforms = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        valid_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        full_dataset = datasets.ImageFolder(
            self.config.training_data, 
            transform=train_transforms
        )
        
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(
            full_dataset, 
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.valid_dataset.dataset.transform = valid_transforms
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 on Windows
            pin_memory=True
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    @staticmethod
    def save_model(path: Path, model):
        model.save(path)
    
    def train(self):
        """Train using TensorFlow model with PyTorch DataLoader"""
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Helper to convert PyTorch tensors to TensorFlow format (B, H, W, C)
        def pytorch_to_tf_dataset(data_loader):
            for batch_x, batch_y in data_loader:
                # Convert to numpy and transpose (B, C, H, W) -> (B, H, W, C)
                batch_x = batch_x.numpy()
                batch_x = np.transpose(batch_x, (0, 2, 3, 1))
                
                # One-hot encode labels
                num_classes = self.model.layers[-1].output.shape[-1]
                batch_y = np.eye(num_classes)[batch_y.numpy()]
                
                yield batch_x, batch_y
        
        # FIXED: Using None for batch dimension to handle remainder batches
        output_sig = (
            tf.TensorSpec(
                shape=(None, 
                       self.config.params_image_size[0], 
                       self.config.params_image_size[1], 
                       self.config.params_image_size[2]), 
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, 
                       self.model.layers[-1].output.shape[-1]), 
                dtype=tf.float32
            )
        )

        train_tf_dataset = tf.data.Dataset.from_generator(
            lambda: pytorch_to_tf_dataset(self.train_loader),
            output_signature=output_sig
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE) # <--- ADD THIS LINE
        
        valid_tf_dataset = tf.data.Dataset.from_generator(
            lambda: pytorch_to_tf_dataset(self.valid_loader),
            output_signature=output_sig
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_tf_dataset = tf.data.Dataset.from_generator(
            lambda: pytorch_to_tf_dataset(self.valid_loader),
            output_signature=output_sig
        )
        
        self.model.fit(
            train_tf_dataset,
            epochs=self.config.params_epochs,
            validation_data=valid_tf_dataset,
            verbose=1
        )
        
    
        self.save_model(self.config.trained_model_path, self.model)
        print(f"✓ Model saved to {self.config.trained_model_path}")