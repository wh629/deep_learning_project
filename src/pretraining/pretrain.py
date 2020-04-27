import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision

from pretraining.data_helper import UnlabeledDataset


# def combine_6_images(sample):
#     """
#     Combine a stack of 6 images into a big image of 2 rows and 3 columns.
#
#     Sample shape is (6, 3, H, W), while output shape is (3, 2H, 3W).
#
#     Args:
#         sample (Tensor): a sample of size (6, 3, H, W) that contains 6 images.
#     """
#     return torchvision.utils.make_grid(sample, nrow=3)
#
#
# def repackage_batch(samples):
#     samples = [combine_6_images(sample) for sample in samples]
#     samples = torch.stack(samples, 0)
#     return samples


def get_k_random_permutations_over_n_elements(k, n):
    """
    Generate k random unique permutations from n elements.
    """
    perms = set()
    original = list(range(n))

    while len(perms) < k:
        random.shuffle(original)
        new_perm = tuple(original)
        if new_perm not in perms:
            perms.add(new_perm)

    return list(perms)


class CameraEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=False, progress=False)
        self.decoder = nn.Sequential(
            nn.Linear(6000, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):

        # first apply resnet to each of 6 images for all samples in current batch
        xs = [self.resnet(x[:, i]) for i in range(x.shape[1])]

        # reshape the resnet outputs
        x = torch.stack(xs, dim=1)
        x = x.view(x.shape[0], -1)

        # feed into decoder of fc layers
        x = self.decoder(x)
        return x


def generate_random_image_mask(channels, height, width):
    p = 0.7
    h = int(p * height)
    w = int(p * width)

    random_x = random.randint(0, height - h - 1)
    random_y = random.randint(0, width - w - 1)

    mask = torch.zeros(size=(channels, height, width))
    mask[:, random_x:random_x + h, random_y:random_y + w] = torch.ones(size=(channels, h, w))

    return mask


def pretrain(batch_size=5, permutations_k=64):
    learning_rate = 1e-4
    weight_decay = 1e-8
    max_grad_bound = 1
    num_epochs = 50

    # pre-training tasks aim to restore the original image order.
    permutations = get_k_random_permutations_over_n_elements(k=permutations_k, n=6)

    # Set up your device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    image_folder = 'data'
    pretrain_scene_index = np.arange(106)

    transform = torchvision.transforms.ToTensor()

    pretrain_dataset = UnlabeledDataset(image_folder=image_folder,
                                        scene_index=pretrain_scene_index,
                                        first_dim='sample',
                                        transform=transform)
    pre_train_loader = torch.utils.data.DataLoader(pretrain_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

    model = CameraEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=weight_decay)

    for epoch in range(num_epochs):
        for batch in pre_train_loader:

            # generate answers
            indices = [random.randint(0, permutations_k - 1) for _ in range(batch_size)]
            answers = torch.zeros(size=(batch_size, permutations_k))
            for ith, index in enumerate(indices):
                answers[ith, index] = 1
            answers.to(device)

            # prepare input
            for ith in range(batch_size):
                batch[ith, list(range(6))] = batch[ith, permutations[indices[ith]]]
                random_mask = generate_random_image_mask(*batch[0, 0].shape)
                for jth in range(6):
                    batch[ith, jth] *= random_mask
            batch = batch.to(device)

            output = model(batch)
            loss = criterion(output, answers)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_bound)
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        if epoch % 10 == 0:
            loss_val = loss.item()
            filename = 'out/pretrain_encoder_by_batchSize_{}_numPermutations_{}_epochs_{}_loss_{}.pt'.format(
                batch_size, permutations_k, epoch, int(loss_val * 1000))
            torch.save({'model': model, 'resnet18': model.resnet, 'optimizer': optimizer.state_dict()}, filename)


def get_args():
    args = argparse.ArgumentParser(description='Deep Learning Competition')
    args.add_argument('--batch_size',
                      type=int,
                      default=8,
                      help='batch size')
    args.add_argument('--permutations_k',
                      type=int,
                      default=64,
                      help='number of image permutations in pre-training.')
    return args.parse_args()


if __name__ == '__main__':

    parser = get_args()

    pretrain(parser.batch_size, parser.permutations_k)
