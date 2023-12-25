import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)


def patchify(images, n_patches: int):
    n, c, h, w = images.shape

    assert h == w, "Patches can be done for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches  # Patch size = width / no of patches

    for idx, image in enumerate(images):

        # For each i and j, iterating through the image
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)

    # Calculating the positioning embedding for the image
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))

    return result

class MultiHeadSelfAttention

class VisualTransformer(nn.Module):

    def __init__(self, chw = (1, 28, 28), n_patches: int = 7) -> None:
        # Super Constructor
        super().__init__()

        # Self Attributes:
        self.chw = chw # (Channel, Height, Width)
        self.n_patches = n_patches # Total number of patches

        # Check if the number of patches are perfectly divisible with the Width and Height        
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False # To know that it is not a learning variable 


    def forward(self, images):
        # Dividing into patches
        patches = patchify(images, self.n_patches)

        # Linearly mapping the patches
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        output = tokens + pos_embed

        return output
