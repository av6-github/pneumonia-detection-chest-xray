import torch
print("Torch version:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("Built with CUDA?:", torch.cuda.is_available())

# was running on CNN model built manually from pytorch , but issue was that there number of images with pneumonia almost 3 times that of normal, 
# so the model was biased towards predicting pneumonia all the time.
# So switched to ResNet18 with weighted sampler to fix class imbalance issue.
# the problem still persisted, so we told the model to pay more attention to the minority class (normal) during training using weighted random sampler.
# This way, the model gets a balanced view of both classes during training, helping it learn to identify normal cases better.