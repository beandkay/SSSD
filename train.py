import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from data import DATASET_GETTERS
from resnet import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from autoaugment import CIFAR10Policy
from cutout import Cutout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                   "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default='cifar100', type=str,
                    choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--epoch', default=250, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
args = parser.parse_args()
print(args)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


if args.seed is not None:
    set_seed(args)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args)

if args.local_rank == 0:
    torch.distributed.barrier()

train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
labeled_loader = DataLoader(
    labeled_dataset,
    sampler=train_sampler(labeled_dataset),
    batch_size=args.batch_size,
    num_workers=args.workers,
    drop_last=True)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    sampler=train_sampler(unlabeled_dataset),
    batch_size=args.batch_size*args.mu,
    num_workers=args.workers,
    drop_last=True)

test_loader = DataLoader(test_dataset,
                            sampler=SequentialSampler(test_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.workers)

# if args.autoaugment:
#     transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
#                              transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
#                              Cutout(n_holes=1, length=16),
#                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# else:
#     transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
#                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(),
#                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                (0.2023, 0.1994, 0.2010))])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# if args.dataset == "cifar100":
#     trainset = torchvision.datasets.CIFAR100(
#         root=args.dataset_path,
#         train=True,
#         download=True,
#         transform=transform_train
#     )
#     testset = torchvision.datasets.CIFAR100(
#         root=args.dataset_path,
#         train=False,
#         download=True,
#         transform=transform_test
#     )
# elif args.dataset == "cifar10":
#     trainset = torchvision.datasets.CIFAR10(
#         root=args.dataset_path,
#         train=True,
#         download=True,
#         transform=transform_train
#     )
#     testset = torchvision.datasets.CIFAR10(
#         root=args.dataset_path,
#         train=False,
#         download=True,
#         transform=transform_test
#     )
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=args.batchsize,
#     shuffle=True,
#     num_workers=4
# )
# testloader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=args.batchsize,
#     shuffle=False,
#     num_workers=4
# )

if args.model == "resnet18":
    net = resnet18()
if args.model == "resnet34":
    net = resnet34()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()
if args.model == "wideresnet50":
    net = wide_resnet50_2()
if args.model == "wideresnet101":
    net = wide_resnet101_2()
if args.model == "resnext50_32x4d":
    net = resnet18()
if args.model == "resnext101_32x8d":
    net = resnext101_32x8d()

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
init = False

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(args.epoch):
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        if epoch in [args.epoch // 3, args.epoch * 2 // 3, args.epoch - 10]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(labeled_loader, 0):
            length = len(labeled_loader)
            inputs_l, labels = data
            inputs_l, labels = inputs_l.to(device), labels.to(device)
            
            unlabeled_iter = iter(unlabeled_loader)
            (inputs_uw, inputs_us), _ = unlabeled_iter.next()
            inputs_uw, inputs_us = inputs_uw.to(device), inputs_us.to(device)

            batch_size = inputs_l.shape[0]
            t_images = torch.cat((inputs_l, inputs_uw, inputs_us))
            t_logits, t_feature_logits = net(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, labels)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach()/args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((inputs_l, inputs_us))
            s_logits = net(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), labels)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

            ensemble = sum(t_logits[:-1])/len(t_logits)
            ensemble.detach_()

            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = t_feature_logits[0].size(1)
                for index in range(1, len(t_feature_logits)):
                    student_feature_size = t_feature_logits[index].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                net.adaptation_layers = nn.ModuleList(layer_list)
                net.adaptation_layers.cuda()
                optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
                #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            teacher_output = t_logits[0].detach()
            teacher_feature = t_feature_logits[0].detach()

            #   for shallow classifiers
            for index in range(1, len(t_logits)):
                #   logits distillation
                loss += CrossEntropy(t_logits[index], teacher_output) * args.loss_coefficient
                loss += criterion(t_logits[index], labels) * (1 - args.loss_coefficient)
                #   feature distillation
                if index != 1:
                    loss += torch.dist(net.adaptation_layers[index-1](t_feature_logits[index]), teacher_feature) * \
                            args.feature_loss_coefficient
                    #   the feature distillation loss will not be applied to the shallowest classifier

            with torch.no_grad():
                s_logits_l = net(images_l)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), labels)
                # dot_product = s_loss_l_new - s_loss_l_old
                # test
                dot_product = s_loss_l_old - s_loss_l_new
                # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
                # dot_product = dot_product - moving_dot_product
                _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
                t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
                t_loss = t_loss_uda + t_loss_mpl

            sum_loss += loss.item()
            sum_loss += t_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.zero_grad()
            total += float(labels.size(0))
            t_logits.append(ensemble)

            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            if epoch % 100 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                    ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                            100 * correct[0] / total, 100 * correct[1] / total,
                                            100 * correct[2] / total, 100 * correct[3] / total,
                                            100 * correct[4] / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct = [0 for _ in range(5)]
            predicted = [0 for _ in range(5)]
            total = 0.0
            for data in test_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, outputs_feature = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.append(ensemble)
                for classifier_index in range(len(outputs)):
                    _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                  ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                         100 * correct[2] / total, 100 * correct[3] / total,
                                         100 * correct[4] / total))
            if correct[4] / total > best_acc:
                best_acc = correct[4]/total
                print("Best Accuracy Updated: ", best_acc * 100)
                torch.save(net.state_dict(), "./checkpoints/"+str(args.model)+".pth")

    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.3f" % (args.epoch, best_acc))
