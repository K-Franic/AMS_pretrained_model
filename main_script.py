import torch

# Path to the .tar file
model_path = "C:/Users/frani/Desktop/AMS_pretrained_model/ViT-V-Net/experiments/pretrained_ViT-V-Net.tar"

try:
    checkpoint = torch.load(model_path, map_location='cpu')  # Load directly if PyTorch-compatible
    print("Checkpoint loaded successfully.")
    
    if 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        model_state_dict = checkpoint

    print("Model state dictionary extracted.")
    # model.load_state_dict(model_state_dict) # Assuming you have a model instance

except Exception as e:
    print(f"Error loading model: {e}")
