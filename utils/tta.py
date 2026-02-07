import torch


class TestTimeAugmentation:
    """Test-Time Augmentation for image restoration models"""
    
    def __init__(self, model, dino_net, device, use_flip=True, use_rot=True, use_multi_scale=False, scales=None):
        """
        Args:
            model: The model to apply TTA to
            dino_net: DINO feature extractor
            device: Device to run inference on
            use_flip: Whether to use horizontal and vertical flips
            use_rot: Whether to use 90-degree rotations
            use_multi_scale: Whether to use multi-scale testing
            scales: List of scales to use for multi-scale testing, e.g. [0.8, 1.0, 1.2]
        """
        self.model = model
        self.dino_net = dino_net
        self.device = device
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.use_multi_scale = use_multi_scale
        self.scales = scales or [1.0]
        
    def _apply_augmentation(self, image, point, normal, aug_type):
        """Apply single augmentation to input images
        
        Args:
            image: Input RGB image
            point: Point map
            normal: Normal map
            aug_type: Augmentation type string (e.g., 'original', 'h_flip', etc.)
            
        Returns:
            Augmented versions of image, point map and normal map
        """
        if aug_type == 'original':
            return image, point, normal
            
        elif aug_type == 'h_flip':
            # Horizontal flip
            img_aug = torch.flip(image, dims=[3])
            point_aug = torch.flip(point, dims=[3])
            normal_aug = torch.flip(normal, dims=[3])
            # For normal map, x direction needs to be flipped
            normal_aug[:, 0, :, :] = -normal_aug[:, 0, :, :]
            return img_aug, point_aug, normal_aug
            
        elif aug_type == 'v_flip':
            # Vertical flip
            img_aug = torch.flip(image, dims=[2])
            point_aug = torch.flip(point, dims=[2])
            normal_aug = torch.flip(normal, dims=[2])
            # For normal map, y direction needs to be flipped
            normal_aug[:, 1, :, :] = -normal_aug[:, 1, :, :]
            return img_aug, point_aug, normal_aug
            
        elif aug_type == 'rot90':
            # 90-degree rotation
            img_aug = torch.rot90(image, k=1, dims=[2, 3])
            point_aug = torch.rot90(point, k=1, dims=[2, 3])
            normal_aug = torch.rot90(normal, k=1, dims=[2, 3])
            # Swap x and y channels in normal map and negate x
            normal_x = -normal_aug[:, 1, :, :].clone()
            normal_y = normal_aug[:, 0, :, :].clone()
            normal_aug[:, 0, :, :] = normal_x
            normal_aug[:, 1, :, :] = normal_y
            return img_aug, point_aug, normal_aug
            
        elif aug_type == 'rot180':
            # 180-degree rotation
            img_aug = torch.rot90(image, k=2, dims=[2, 3])
            point_aug = torch.rot90(point, k=2, dims=[2, 3])
            normal_aug = torch.rot90(normal, k=2, dims=[2, 3])
            # For normal map, both x and y directions need to be flipped
            normal_aug[:, 0, :, :] = -normal_aug[:, 0, :, :]
            normal_aug[:, 1, :, :] = -normal_aug[:, 1, :, :]
            return img_aug, point_aug, normal_aug
            
        elif aug_type == 'rot270':
            # 270-degree rotation
            img_aug = torch.rot90(image, k=3, dims=[2, 3])
            point_aug = torch.rot90(point, k=3, dims=[2, 3])
            normal_aug = torch.rot90(normal, k=3, dims=[2, 3])
            # Swap x and y channels in normal map and negate y
            normal_x = normal_aug[:, 1, :, :].clone()
            normal_y = -normal_aug[:, 0, :, :].clone()
            normal_aug[:, 0, :, :] = normal_x
            normal_aug[:, 1, :, :] = normal_y
            return img_aug, point_aug, normal_aug
        
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def _reverse_augmentation(self, result, aug_type):
        """Reverse the augmentation on the result
        
        Args:
            result: Model output to reverse augmentation on
            aug_type: Augmentation type string
            
        Returns:
            De-augmented result
        """
        if aug_type == 'original':
            return result
            
        elif aug_type == 'h_flip':
            return torch.flip(result, dims=[3])
            
        elif aug_type == 'v_flip':
            return torch.flip(result, dims=[2])
            
        elif aug_type == 'rot90':
            return torch.rot90(result, k=3, dims=[2, 3])
            
        elif aug_type == 'rot180':
            return torch.rot90(result, k=2, dims=[2, 3])
            
        elif aug_type == 'rot270':
            return torch.rot90(result, k=1, dims=[2, 3])
        
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def __call__(self, sliding_window, input_img, point, normal):
        """
        Apply TTA to the model and return ensemble result
        
        Args:
            sliding_window: SlidingWindowInference class instance
            input_img: Input RGB image [B, C, H, W]
            point: Point map [B, C, H, W]
            normal: Normal map [B, C, H, W]
            
        Returns:
            Ensemble result with TTA [B, C, H, W]
        """
        # Define all augmentations to use
        augmentations = ['original']
        if self.use_flip:
            augmentations.extend(['h_flip', 'v_flip'])
        if self.use_rot:
            augmentations.extend(['rot90', 'rot180', 'rot270'])
        
        # Initialize the result tensor
        ensemble_result = torch.zeros_like(input_img)
        ensemble_weight = 0.0
        
        # For each scale and augmentation
        for scale in self.scales:
            scale_weight = 1.0
            if scale != 1.0:
                # Resize inputs for multi-scale testing
                h, w = input_img.shape[2], input_img.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Resize all inputs
                resize_fn = torch.nn.functional.interpolate
                input_img_scaled = resize_fn(input_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                point_scaled = resize_fn(point, size=(new_h, new_w), mode='bilinear', align_corners=False)
                normal_scaled = resize_fn(normal, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # Normalize normal vectors after resizing
                normal_norm = torch.sqrt(torch.sum(normal_scaled**2, dim=1, keepdim=True) + 1e-6)
                normal_scaled = normal_scaled / normal_norm
            else:
                input_img_scaled = input_img
                point_scaled = point
                normal_scaled = normal
            
            # Apply each augmentation
            for aug_type in augmentations:
                # Apply augmentation
                img_aug, point_aug, normal_aug = self._apply_augmentation(
                    input_img_scaled, point_scaled, normal_scaled, aug_type
                )
                
                # Run model inference with sliding window
                with torch.cuda.amp.autocast():
                    result_aug = sliding_window(
                        model=self.model,
                        input_=img_aug,
                        point=point_aug,
                        normal=normal_aug,
                        dino_net=self.dino_net,
                        device=self.device
                    )
                
                # Reverse augmentation on the result
                result_aug = self._reverse_augmentation(result_aug, aug_type)
                
                # Resize back to original size if using multi-scale
                if scale != 1.0:
                    result_aug = resize_fn(result_aug, size=(h, w), mode='bilinear', align_corners=False)
                
                # Add to ensemble
                ensemble_result += result_aug * scale_weight
                ensemble_weight += scale_weight
        
        # Average results
        ensemble_result = ensemble_result / ensemble_weight
        
        return ensemble_result