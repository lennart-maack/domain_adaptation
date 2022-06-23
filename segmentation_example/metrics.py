import torch
from torch import nn


def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels.to(torch.long)]

class Dice_Coefficient(nn.Module):

    def __init__(self, epsilon=1e-7):
        """
        Does not calculate mean over batch dimension! Returns result for every example in the batch independently.
        This is not appropriate when performing statistical tests later because this approach decreases the variance when means of batch means are taken!
        
        Output:
        # outputs a tensor of shape (batch_size, n_classes(channels) + 2)
            # each row belongs to the dice values of one output['out'], target combination
            # the first n_classes columns belong the dice coeff of each channel
            # the pre last column [-2] belongs to the mean dice value over all channels/classes per image
            # the last column [-1] belongs to the mean dice value over all channels/classes except the first one per image
        
        """
        super().__init__()
        self.epsilon = torch.tensor(epsilon, device=torch.device('cpu'))

    def __str__(self):
        return 'DICE'

    def forward(self, input, target):
        # shape for both, input and target: [n_batch, num_classes, h, w]
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        batch_size, num_classes, _, _ = input.shape

        # needed for binary segmentation
        input = torch.sigmoid(input, out=None) 
        input = torch.round(input)

        # Initializes a tensor of size (batch_size, n_channels+2) - this is where the values for dice 
        dice = torch.zeros(size=(batch_size, num_classes + 2), dtype=torch.float32)

        input = input.cpu()
        target = target.cpu()

        # Calculate the nominator and denominator for each channel(class) and each image --> shape: ([batchSize, n_channels])
        nominators = (input * target).sum(dim=(-2, -1)) + self.epsilon
        denominators = (input + target).sum(dim=(-2, -1)) + self.epsilon

        # Calculates the dice score for each channel and each image and puts it into the first four columns of dice
        dice[:, :num_classes] = (2.0 * nominators / denominators)

        # Calculates two different means 
        # dice[:,-2] (pre last column): mean over all channels/classes per input&target
        # dice[:,-1] (last column): mean over all channels/classes except the first one per input&target
        dice[:, -2] = torch.mean(dice[:, :-2], dim=1)
        dice[:, -1] = torch.mean(dice[:, 1:-2], dim=1)

        return dice * 100