import math
import time

import numpy as np
import torch
from torch.nn import functional as F
from other_utils import Logger
import torch.nn as nn
import checks


def select_top_k(logits, y, k):
    sorted_logits, sorted_index = logits.sort(dim=1, descending=True)
    y_list = y.cpu().numpy()
    top_k_index = [x[0:k+1] for x in sorted_index.cpu().numpy()]
    for i in range(len(top_k_index)):
        if y_list[i] in top_k_index[i]:
            y_index = [x for x in range(len(top_k_index[i])) if top_k_index[i][x] == y_list[i]][0]
            top_k_index[i] = np.concatenate((top_k_index[i][0:y_index], top_k_index[i][y_index+1:]), axis=0)
        else:
            top_k_index[i] = top_k_index[i][:-1]

    top_k_index = torch.tensor(np.array(top_k_index))
    return top_k_index


def dlr_loss(outputs, labels, target_labels=None, targeted=False):
    outputs_sorted, ind_sorted = outputs.sort(dim=1)
    if targeted:
        cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs[np.arange(outputs.shape[0]), target_labels]) \
               / (outputs_sorted[:, -1] - .5 * outputs_sorted[:, -3] - .5 * outputs_sorted[:, -4] + 1e-12)
    else:
        ind = (ind_sorted[:, -1] == labels).float()
        cost = -(outputs[np.arange(outputs.shape[0]), labels] - outputs_sorted[:, -2] * ind - outputs_sorted[:, -1] * (1. - ind)) \
               / (outputs_sorted[:, -1] - outputs_sorted[:, -3] + 1e-12)
    return cost.sum()


class PGD(nn.Module):

    def __init__(self, model, eps=8 / 255, iter_eps=2 / 255, nb_iter=40, clip_min=0.0, clip_max=1.0,
                 flag_target=False, num_class=10):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model.to(self.device)
        self.flag_target = flag_target
        self.num_class = num_class
        self.loss_dict = {'MT': self.MT_Loss, 'CE': nn.CrossEntropyLoss(),
                          'DLR': dlr_loss}
        self.ODI_iter_eps = eps
        self.ODI_nb_iter = 2

    def MT_Loss(self, logits, y, index, loss_func='MT'):
        if loss_func == 'MT':
            x_one_hot = torch.eye(len(logits[0]))[index].bool().to(self.device)
            y_one_hot = torch.eye(len(logits[0]))[y].bool().to(self.device)
            selected_logits_x = torch.masked_select(logits, x_one_hot)
            selected_logits_y = torch.masked_select(logits, y_one_hot)
            loss = (selected_logits_x - selected_logits_y).sum()
        elif loss_func == 'MT-CE':
            loss = -self.loss_dict['CE'](logits, index.to(self.device))
        elif loss_func == 'MT-DLR':
            loss = self.loss_dict['DLR'](logits, y, target_labels=index.to(self.device), targeted =True)
        else:
            return
        return loss

    def sigle_step_attack(self, x, perturbation, labels, loss_func=nn.CrossEntropyLoss()):
        adv_x = x + perturbation
        adv_x.requires_grad = True
        preds = self.model(adv_x)

        if self.flag_target:
            loss = -loss_func(preds, labels)
        else:
            loss = loss_func(preds, labels)

        self.model.zero_grad()
        loss.backward(loss.clone().detach())
        grad = adv_x.grad.data
        perturbation = self.iter_eps * torch.sign(grad)
        adv_x = adv_x + perturbation
        adv_x = torch.clamp(torch.min(torch.max(adv_x, x - self.eps), x + self.eps), 0.0, 1.0)
        return adv_x - x

    def attack(self, x, labels, x_init=None, init_flag=False, loss_func=nn.CrossEntropyLoss()):
        labels = labels.to(self.device)
        if init_flag:
            perturbation = x_init - x
        else:
            perturbation = torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).to(self.device)
        for i in range(self.nb_iter):
            perturbation = self.sigle_step_attack(x, perturbation=perturbation, labels=labels,
                                                  loss_func=loss_func)
            perturbation = perturbation.clone().detach()
        adv_x = x + perturbation
        return adv_x

    def get_adv_x(self, x, y, init_flag=False, x_init=None, loss_func=nn.CrossEntropyLoss(),
                  MT_flag=False, MT_index=None, odi_flag=False, odi_loss='MSE'):
        if init_flag:
            delta = (x_init - x).clone().detach()
        elif odi_flag:
            delta = self.ODI(x, loss_func=odi_loss) - x
        else:
            delta = torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).to(self.device)
        delta.requires_grad_()
        if MT_flag:
            for _ in range(self.nb_iter):
                logits = self.model(x+delta)
                loss = self.MT_Loss(logits, y, MT_index, loss_func)
                loss.backward()
                grad_sign = delta.grad.data.sign()
                delta.data += self.iter_eps * grad_sign
                delta.data = torch.clamp(torch.min(torch.max(x + delta, x - self.eps), x + self.eps), 0.0, 1.0) - x
                delta.grad.data.zero_()
        else:
            for _ in range(self.nb_iter):
                logits = self.model(x+delta)
                loss = loss_func(logits, y)
                if self.flag_target:
                    loss = -loss
                loss.backward()
                grad_sign = delta.grad.data.sign()
                delta.data += self.iter_eps * grad_sign
                delta.data = torch.clamp(torch.min(torch.max(x + delta, x - self.eps), x + self.eps), 0.0, 1.0) - x
                delta.grad.data.zero_()
        return x + delta

    def get_MT_adv_x(self, x, y, k=5, loss_func='MT', x_init=None, init_flag=False):
        logits = self.model(x)
        MT_index = select_top_k(logits, y, k)
        adv_x = self.get_adv_x(x, y, loss_func=loss_func, MT_flag=True, MT_index=MT_index[..., 0],
                               x_init=x_init, init_flag=init_flag)
        invalid = torch.max(self.model(adv_x), dim=1)[1] == y
        for i in range(1, k):
            if invalid.float().sum() <= 0:
                break
            if init_flag:
                adv_x[invalid] = self.get_adv_x(x[invalid], y[invalid], loss_func=loss_func, MT_flag=True,
                                                MT_index=MT_index[..., i][invalid],
                                                x_init=x_init[invalid], init_flag=init_flag)
            else:
                adv_x[invalid] = self.get_adv_x(x[invalid], y[invalid], loss_func=loss_func, MT_flag=True,
                                                MT_index=MT_index[..., i][invalid])
            invalid_copy = invalid.clone()
            invalid[invalid_copy] = torch.max(self.model(adv_x[invalid_copy]), dim=1)[1] == y[invalid_copy]
        return adv_x

    def combine_attack(self, x, y, loss_list=None, k=5, odi_flag=False, odi_loss='ODI'):
        if loss_list is None:
            return x
        else:
            if loss_list[0][:2] == 'MT':
                adv_x = self.get_MT_adv_x(x, y, k, loss_list[0])
            else:
                adv_x = self.get_adv_x(x, y, x_init=None, loss_func=self.loss_dict[loss_list[0]], odi_flag=odi_flag,
                                       odi_loss=odi_loss)
            invalid = torch.max(self.model(adv_x), dim=1)[1] == y
            for loss_func in loss_list[1:]:
                self.model.reset()
                if invalid.float().sum() <= 0:
                    break
                elif loss_func[:2] == 'MT':
                    adv_x[invalid] = self.get_MT_adv_x(x[invalid], y[invalid], k, loss_func,
                                                       x_init=adv_x[invalid], init_flag=True)
                else:
                    adv_x[invalid] = self.get_adv_x(x[invalid], y[invalid], x_init=adv_x[invalid], init_flag=True,
                                                    loss_func=self.loss_dict[loss_func])
                invalid_copy = invalid.clone()
                invalid[invalid_copy] = torch.max(self.model(adv_x[invalid_copy]), dim=1)[1] == y[invalid_copy]
        return adv_x

    def ODI(self, x, loss_func='MSE'):
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        for _ in range(self.ODI_nb_iter):
            output = self.model(x_adv)
            if loss_func == 'OID':
                randvecter = torch.Tensor(np.random.uniform(-1, 1, output.shape)).type_as(output).to(self.device)
                loss = (randvecter * output).sum()
            elif loss_func == 'KL':
                loss = F.kl_div(F.log_softmax(self.model(x_adv), dim=1), F.softmax(self.model(x), dim=1), reduction='sum')
            elif loss_func == 'MSE':
                loss = ((F.softmax(self.model(x_adv), dim=1) - F.softmax(self.model(x), dim=1)) ** 2).mean()
            else:
                pass
            loss.backward()
            x_adv.data += self.ODI_iter_eps * x_adv.grad.data.sign()
            x_adv.data = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)
            x_adv.grad.data.zero_()
        return x_adv.clone().detach()



class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")

        if not self.is_tf_model:
            from autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                                   eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                                   device=self.device, logger=self.logger)

            from fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                                    norm=self.norm, verbose=False, device=self.device)

            from square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                                       n_restarts=1, seed=self.seed, verbose=False, device=self.device,
                                       resc_schedule=False)

            from autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                                                     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                     seed=self.seed, device=self.device,
                                                     logger=self.logger)

        else:
            from autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                                   eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                                   device=self.device,
                                   is_tf_model=True, logger=self.logger)

            from fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                                    norm=self.norm, verbose=False, device=self.device)

            from square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                                       n_restarts=1, seed=self.seed, verbose=False, device=self.device,
                                       resc_schedule=False)

            from autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                                                     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                     seed=self.seed, device=self.device,
                                                     is_tf_model=True, logger=self.logger)

        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)

    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                                    y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
                                          logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
                             logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
                               self.fab.n_target_classes, logger=self.logger)

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            y_adv = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            attack_times = 1
            for attack in self.attacks_to_run:
                if attack_times > 1:
                    self.model.reset()
                attack_times += 1
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    x_init = x_adv[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)  # cheap=True

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, x_init=x_init)  # cheap=True

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y)  # cheap=True

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    # x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)
                    x_adv[batch_datapoint_idcs] = adv_curr.detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv, time.time() - startt))

        return adv

    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))

        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            # self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000

        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))

        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20
