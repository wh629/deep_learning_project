import argparse
import random
import itertools
import numpy as np
from tqdm import trange, tqdm
import os
import torch
import torch.nn as nn
import torchvision
import logging as log
from datetime import datetime as dt

from data_helper import UnlabeledDataset


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
def repackage_image_batch(samples: torch.Tensor):
    """
    Repackage a batch of images from [batch_size, 6, 3, 256, 306] to [batch_size, 6, 3, 800, 800]
    """
    samples_new = []
    for sample in samples:
        samples_new.append(nn.functional.interpolate(sample, size=(800, 800), mode='nearest'))  #resize
    return torch.stack(samples_new, dim=0)

def get_hamming_distance(perm_i, perm_j):
    return sum([e1 != e2 for e1, e2 in zip(perm_i, perm_j)])

def get_k_permutations_of_n_elements(k, n):
    """
    Generate k random unique permutations of n elements. These k permutations are better to have long hamming distances.
    """
    perms_available = list(itertools.permutations(range(n)))

    # shuffle permutations
    random.shuffle(perms_available)
    perms_output = []
    t = range(n)
    threshold = n - 2

    for perm in perms_available:
        if len(perms_output) == k:
            break
        d_true = get_hamming_distance(perm, t)

        # include permutation only if hamming distance from original is at least threshold
        if d_true >= threshold:

            # include permutation only if hamming distance with all included is at least threshold
            add = True
            for included in perms_output:
                d = get_hamming_distance(perm, included)
                if d_true < threshold:
                    add = False

            if add:
                perms_output.append(perm)

    # get average hamming distance
    running = 0
    for i, perm_i in enumerate(perms_output):
        run_i = 0
        for j, perm_j in enumerate(perms_output):
            if i != j:
                run_i += get_hamming_distance(perm_i, perm_j)

        running += run_i/(len(perms_output)-1)


    # distance_mapping = {}
    # while len(perms_output) < k:
    #     dists = []
    #     for perm_i in perms_available:
    #         dist_i = 0
    #         for perm_j in perms_output:
    #             if (perm_i, perm_j) in distance_mapping:
    #                 dist_i += distance_mapping[(perm_i, perm_j)]
    #             else:
    #                 dist_ij = sum([e1 != e2 for e1, e2 in zip(perm_i, perm_j)])  # hamming distance
    #                 dist_i += dist_ij
    #                 distance_mapping[(perm_i, perm_j)] = dist_ij
    #                 distance_mapping[(perm_j, perm_i)] = dist_ij
    #
    #         dists.append(dist_i)
    #
    #     idx_max = dists.index(max(dists))
    #     perms_output.append(perms_available.pop(idx_max))

    return perms_output, running/len(perms_output)


class CameraEncoder(nn.Module):
    def __init__(self, permutations_k=8, hidden_size=4096):
        super().__init__()

        self.image_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                                 progress=True,
                                                                                 num_classes=9,
                                                                                 pretrained_backbone=False)

        self.resnet = self.image_detect.backbone
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.conv_256_1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        self.double_dim_minus1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2,
                                                    padding=1)
        self.relu = nn.ReLU()
        self.target_size = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((self.target_size, self.target_size))

        self.decoder = nn.Sequential(
            nn.Linear(6 * 13 * 13, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, permutations_k)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(6*5*self.target_size*self.target_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, permutations_k)
        # )

    def forward(self, x):
        # first apply resnet to each of 6 images for all samples in current batch
        # x is (batch_size, 6, 3, 800, 800)
        bs, _, _, _, _ = x.shape

        xs = [self.resnet(x[:, i]) for i in range(x.shape[1])]
        # The features is a OrderDict as follows:
        #     key: 0    -> value: a tensor of shape [batch_size, 256, 200, 200]
        #     key: 1    -> value: a tensor of shape [batch_size, 256, 100, 100]
        #     key: 2    -> value: a tensor of shape [batch_size, 256, 50, 50]
        #     key: 3    -> value: a tensor of shape [batch_size, 256, 25, 25]
        #     key: pool -> value: a tensor of shape [batch_size, 256, 13, 13]

        xs_new = []
        for features in xs:
            xs_new.append(self.relu(self.conv_256_1(features['pool']).view(bs, -1)))
            # [batch_size, 13*13]

            # features_new = []
            # for key, feature in features.items():
            #     # try just with pool layer
            #     features_new.append(self.relu(self.conv_256_1(feature).view(bs, -1)))

            #     temp = self.relu(self.avg_pool(feature))
            #     features_new.append(self.relu(self.conv_256_1(temp).view(bs, -1)))
            #     # list with entries of (batch_size, out_dim*out_dim)
            #
            # thin_feature for single image
            # thin_feature = torch.cat(features_new, dim=1)
            # size (batch_size, 5*10*10 = 500)

            # for each image
            # xs_new.append(thin_feature)
            # entries of xs_new are [batch_size, 13*13]

        # combined images
        # (batch_size, 6*13*13)
        x = torch.cat(xs_new, dim=1)

        # feed into decoder of fc layers
        x = self.decoder(x)

        # get predictions
        pred = torch.argmax(x, dim = 1)
        return x, pred


def generate_random_image_mask(channels, height, width):
    p = 0.7
    h = int(p * height)
    w = int(p * width)

    random_x = random.randint(0, height - h - 1)
    random_y = random.randint(0, width - w - 1)

    mask = torch.zeros(size=(channels, height, width))
    mask[:, random_x:random_x + h, random_y:random_y + w] = torch.ones(size=(channels, h, w))

    return mask

def eval(loader, model, permutations, permutations_k, device, idx, debug):
    log.info(f"Evaluating at step {idx}")
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, batch_old in enumerate(tqdm(loader, desc='Eval', mininterval=30)):
            batch = repackage_image_batch(batch_old)

            batch_size_now = batch.shape[0]
            # generate answers
            answers = torch.randint(permutations_k, (batch_size_now,)).to(device)

            # prepare input
            for ith in range(batch_size_now):
                batch[ith] = batch[ith, permutations[answers[ith].item()]]
                for jth in range(6):
                    random_mask = generate_random_image_mask(*batch[0, 0].shape)
                    batch[ith, jth] *= random_mask

            batch = batch.to(device)
            output, pred = model(batch)

            correct += torch.eq(pred.float(), answers.float()).sum().item()

            if i == 1 and debug:
                log.info("Debug break evaluation")
                break

    model.train()
    return correct/len(loader.dataset) # returns the accuracy

def pretrain(parser, batch_size=5, permutations_k=64):
    max_grad_bound = 1

    print('Start pre-training, batch_size = {}, permutations_k = {}'.format(batch_size, permutations_k))
    log.info('Start pre-training, batch_size = {}, permutations_k = {}'.format(batch_size, permutations_k))

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

    # pre-training tasks aim to restore the original image order.
    permutations, avg_hamming = get_k_permutations_of_n_elements(k=permutations_k, n=6)

    pretrain_scene_index = np.arange(106)

    transform = torchvision.transforms.ToTensor()

    # load and split data
    pretrain_dataset = UnlabeledDataset(image_folder=parser.data_dir,
                                        scene_index=pretrain_scene_index,
                                        first_dim='sample',
                                        transform=transform)

    val_size = int(len(pretrain_dataset) * parser.split)
    train_size = int(len(pretrain_dataset) - val_size)

    train, val = torch.utils.data.random_split(pretrain_dataset,[train_size, val_size])

    pre_train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

    pre_train_val = torch.utils.data.DataLoader(val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)

    model = CameraEncoder(permutations_k).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)

    accumulated = 0
    cum_loss = 0
    global_step = 1
    logged = False
    checked = False
    best_step = 0
    best_acc = -1
    stop = False
    n_no_improve = 0
    saved = False

    filename = os.path.join(parser.save_dir, f"{parser.experiment}_best.pt")

    model.train()
    for epoch in trange(0, parser.num_epochs, desc='Epochs', mininterval = 30):
        for batch_old in tqdm(pre_train_loader, desc='Iteration', mininterval = 30):
            batch = repackage_image_batch(batch_old)

            batch_size_now = batch.shape[0]
            # generate answers
            answers = torch.randint(permutations_k, (batch_size_now,)).to(device)

            # prepare input
            for ith in range(batch_size_now):
                batch[ith] = batch[ith, permutations[answers[ith].item()]]
                for jth in range(6):
                    random_mask = generate_random_image_mask(*batch[0, 0].shape)
                    batch[ith, jth] *= random_mask

            batch = batch.to(device)
            output, pred = model(batch)
            loss = criterion(output, answers)
            cum_loss += loss.item()

            if accumulated == 0:
                model.zero_grad()

            assert type(loss.item()) == float, f"Loss {loss} is not a float and is {type(loss)}."
            loss.backward()
            accumulated += 1
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_bound)

            if accumulated == parser.accum_grad:
                optimizer.step()
                accumulated = 0
                global_step += 1
                logged = False
                checked = False

            if global_step % parser.log_steps == 0 and not logged:
                logged = True
                log.info('Epoch [{}/{}] | Step {} | Avg Loss:{:.8f}'.format(epoch + 1, parser.num_epochs, global_step, cum_loss/global_step))

            if global_step % parser.save_steps == 0 and not checked:
                checked = True
                current_acc = eval(pre_train_val, model, permutations, permutations_k, device, global_step, parser.debug)
                log.info(f"Current Acc {current_acc} | Current Step {global_step} | Previous Best {best_acc} | Best Step {best_step}")

                if current_acc >= best_acc:
                    # if new best accuracy
                    best_acc = current_acc
                    best_step = global_step
                    torch.save(model.resnet.state_dict(), filename)
                    log.info(f"Weights saved to {filename}")
                    saved = True

                if current_acc <= best_acc:
                    # if no improvement
                    n_no_improve += 1
                    log.info(f"No Improvement Counter {n_no_improve} out of {parser.patience}")

                    if n_no_improve > parser.patience:
                        stop = True
                        log.info(f"Early stop at step {global_step}")
                        break
        if stop:
            break

    if not os.path.exists(filename) and not saved:
        torch.save(model.resnet.state_dict(), filename)
        log.info(f"Weights saved to {filename}")

    log.info('Finished')
    log.info(f"Best accuracy {best_acc} | Best step {best_step} | Current Step {global_step} | "
             f"Number of Permutations {permutations_k} | Avg Hamming {avg_hamming} | "
             f"Total Epochs {parser.num_epochs} | Best weights saved to {filename}")


def get_args():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args = argparse.ArgumentParser(description='Pretraining')

    args.add_argument('--batch_size',
                      type=int,
                      default=2,
                      help='batch size')
    args.add_argument('--permutations_k',
                      type=int,
                      default=35,
                      help='number of image permutations in pre-training.')
    args.add_argument('--num_epochs',
                      type=int,
                      default=2,
                      help='number of epochs to train')
    args.add_argument('--accum_grad',
                      type=int,
                      default=4,
                      help='number of gradient accumulation steps')
    args.add_argument('--data_dir',
                      type=str,
                      default=os.getenv('DL_DATA_DIR', os.path.join(repo_dir, "data")),
                      help='directory with data')
    args.add_argument('--save_dir',
                      type=str,
                      default=os.getenv('DL_RESULTS_DIR', os.path.join(repo_dir, "results")),
                      help='directory for results')
    args.add_argument('--experiment',
                      type=str,
                      default='default',
                      help='name of experiment')
    args.add_argument('--log_steps',
                      type=int,
                      default = 100,
                      help='number of iterations before logging')
    args.add_argument('--save_steps',
                      type=int,
                      default = 500,
                      help='number of iterations between evaluations')
    args.add_argument('--patience',
                      type=int,
                      default = 5,
                      help='number of evaluations without improvement before early stop')
    args.add_argument('--split',
                      type=float,
                      default = 0.1,
                      help='percentage of data for validation')
    args.add_argument('--lr',
                      type=float,
                      default = 0.1,
                      help='learning rate')
    args.add_argument('--debug',
                      action='store_true',
                      help='whether debugging code for evaluation')
    return args.parse_args()


if __name__ == '__main__':
    parser = get_args()

    parser.run_log = os.path.join(parser.save_dir, 'log')
    if not os.path.exists(parser.run_log):
        os.mkdir(parser.run_log)

    log_name = os.path.join(parser.run_log, '{}_run_log_{}.log'.format(
        parser.experiment,
        dt.now().strftime("%Y%m%d_%H%M")
    )
                            )
    log.basicConfig(filename=log_name,
                    format='%(asctime)s | %(name)s -- %(message)s',
                    level=log.INFO)

    parser = get_args()

    pretrain(parser, parser.batch_size, parser.permutations_k)
