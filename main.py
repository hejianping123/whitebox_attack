import torch
# from models.wideresnet import WideResNet
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from autoattack import AutoAttack, PGD
from robustbench.utils import load_model
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD


class AlphaModel(nn.Module):
    def __init__(self, model, T=1):
        super(AlphaModel, self).__init__()
        self.model = model
        self.T = T

    def forward(self, images):
        logits = self.model(images)
        return logits / self.T

    def reset(self):
        self.T = 1


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])
    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def dlr_loss_2(outputs, labels, target_labels=None, targeted=False):
    outputs_sorted, ind_sorted = outputs.sort(dim=1)
    if targeted:
        cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs[np.arange(outputs.shape[0]), target_labels]) \
               / (outputs_sorted[:, -1] - .5 * outputs_sorted[:, -3] - .5 * outputs_sorted[:, -4] + 1e-12)
    else:
        ind = (ind_sorted[:, -1] == labels).float()
        cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs_sorted[:, -2] * ind - outputs_sorted[:, -1] * (1. - ind)) \
               / (outputs_sorted[:, -1] - outputs_sorted[:, -3] + 1e-12)
    return cost.sum()


transformer = transforms.Compose([
    transforms.ToTensor()
]
)

cifar10_test = datasets.CIFAR10(
    root=r'/data/hjp/blackbox/cifar10',
    train=False,
    download=False,
    transform=transformer
)

attack_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=500, shuffle=False, num_workers=4)
device = torch.device("cuda")


def select_top_k(logits, y, k):
    sorted_logits, sorted_index = logits.sort(dim=1, descending=True)
    y_list = y.cpu().numpy()
    top_k_index = [x[0:k+1] for x in sorted_index.numpy()]
    for i in range(len(top_k_index)):
        if y_list[i] in top_k_index[i]:
            y_index = [x for x in range(len(top_k_index[i])) if top_k_index[i][x] == y_list[i]][0]
            top_k_index[i] = np.concatenate((top_k_index[i][0:y_index], top_k_index[i][y_index+1:]), axis=0)
    for i in len(top_k_index):
        print(len(top_k_index[i]))
    top_k_index = torch.tensor(top_k_index)
    return top_k_index


def MT_Loss(logits, y, index):
    # sorted_logits, sorted_index = logits.sort(dim=1)
    # equal = y == index
    # temp_index = index.clone().detach()
    # temp_index[equal] += 1
    x_one_hot = torch.eye(len(logits[0]))[index].bool()
    y_one_hot = torch.eye(len(logits[0]))[y].bool()
    selected_logits_x = torch.masked_select(logits, x_one_hot)
    selected_logits_y = torch.masked_select(logits, y_one_hot)
    return selected_logits_x - selected_logits_y

if __name__ == '__main__':
    device = torch.device("cuda")
    model = load_model(model_name='Gowal2021Improving_R18_ddpm_100m', dataset='cifar10', threat_model='Linf',
                       model_dir='/data/hjp/models').to(device)
    # model = WideResNet().to(device)
    # model.load_state_dict(torch.load('/data/hjp/autoattack/models/checkpoints/model_cifar_wrn.pt'))
    loop = tqdm(attack_loader, total=len(attack_loader))
    model = AlphaModel(model, T=1)
    model.eval()

    # x = torch.tensor([[11, 3, 4, 5, 6],
    #                  [2, 32, 4, 5, 6]])
    # y = torch.tensor([0, 3])
    # # select_top_k(x, y, 3)
    # print(MT_Loss(x, y, torch.tensor([1, 1])))
    # foolbox:
    # fmodel = PyTorchModel(model, bounds=(0, 1))
    # attack = LinfPGD(steps=40)
    # total = 0
    # for x, y in loop:
    #     x, y = x.to(device), y.to(device)
    #     _, _, success = attack(fmodel, x, y, epsilons=0.031)
    #     total += success.float().sum().item()
    # print(total/10000)

    pgd = PGD(model)
    total = 0
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        adv_x = pgd.combine_attack(x, y, loss_list=['DLR', 'MT-DLR'], k=8)
        acc_num = (torch.max(model(adv_x), dim=1)[1] == y).float().sum().item()
        total += acc_num
    print(total/10000)

    # adversary = AutoAttack(model, norm='Linf', eps=0.031, version='standard')
    # adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    # x_total = []
    # y_total = []
    # for x, y in loop:
    #     x, y = x.to(device), y.to(device)
    #     x_total.append(x), y_total.append(y)
    # x = torch.cat(x_total, 0)
    # y = torch.cat(y_total, 0)
    # with torch.no_grad():
    #     x_adv = adversary.run_standard_evaluation(x, y, bs=500)
    # print(adversary.model.T)
