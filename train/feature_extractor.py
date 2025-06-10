from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=208):
        super().__init__(observation_space, features_dim)

        # Image processing CNN
        n_input_channels = observation_space["image"].shape[2]
        self.cnn = nn.Sequential(
          nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Flatten()
      )

        # Calculate CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space["image"].sample()[None]).permute(0, 3, 1, 2).float()
            ).shape[1]

        '''
        try:
            dummy_input = torch.as_tensor(observation_space["image"].sample()[None]).permute(0, 3, 1, 2).float()
            print("Dummy input shape before passing to CNN:", dummy_input.shape)
            n_flatten = self.cnn(dummy_input).shape[1]
            print("CNN output shape:", self.cnn(dummy_input).shape)
        except Exception as e:
            print("Error during CNN initialization:", str(e))
            raise
        '''


        # Fully connected layer for vector input
        self.fc = nn.Sequential(
            nn.Linear(observation_space["vector"].shape[0],8),
            nn.ReLU()
        )

        # Combine CNN and vector outputs
        self.fc_combined = nn.Sequential(
            nn.Linear(n_flatten + 8, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
      # Move tensors to the appropriate device (GPU or CPU)
      device = next(self.cnn.parameters()).device  # Get the device of the model
      #print("Image shape before permute:", observations["image"].shape)
      image = observations["image"].to(device).permute(0, 3, 1, 2)  # Move image to device and permute
      #print("Image shape after permute:", image.shape)
      vector = observations["vector"].to(device)  # Move vector to device

      # Process image through CNN
      cnn_out = self.cnn(image)

      # Process vector through fully connected layer
      vector_out = self.fc(vector)

      # Concatenate and process combined features
      combined = torch.cat([cnn_out, vector_out], dim=1)
      return self.fc_combined(combined)