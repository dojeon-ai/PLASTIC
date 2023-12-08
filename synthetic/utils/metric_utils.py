import torch
import torch.nn as nn

# calculate dead ratio
def get_param_cnt_ratio(cnt_dict, param_dict, post_fix='', activation=False, name_idx=3):
    param_cnt_ratio = {}

    _cnt_dict, _param_dict = {}, {}
    for layer_name, _ in cnt_dict.items():
        # when computing activation output, only consider the output after ReLU
        if activation:
            name_idx = 4
            if (layer_name.split('.')[2] != 'ReLU') and (layer_name.split('.')[2] != 'CReLU'):
                continue

        if '.'.join(layer_name.split('.')[:1]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:1])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:1])] += param_dict[layer_name]

        if '.'.join(layer_name.split('.')[:2]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:2])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:2])] += param_dict[layer_name]

        if '.'.join(layer_name.split('.')[:name_idx]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:name_idx])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:name_idx])] += param_dict[layer_name]

        if '.'.join(layer_name.split('.')[:1]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:1])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:1])] = param_dict[layer_name]

        if '.'.join(layer_name.split('.')[:2]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:2])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:2])] = param_dict[layer_name]

        if '.'.join(layer_name.split('.')[:name_idx]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:name_idx])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:name_idx])] = param_dict[layer_name]

    for k, v in _cnt_dict.items():
        num_cnt = v
        num_param = _param_dict[k]

        param_cnt_ratio[k + '_' + post_fix] = num_cnt / num_param

    if ('backbone' in _cnt_dict) and ('policy' in _cnt_dict):
        param_cnt_ratio['total' + '_' + post_fix] = (
                (_cnt_dict['backbone'] + _cnt_dict['policy']) / (_param_dict['backbone'] + _param_dict['policy'])
        )

    return param_cnt_ratio


def get_grad_norm_stats(model):
    grad_norm = []
    stats = {}
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.append(p.grad.detach().data.norm(2))
    grad_norm = torch.stack(grad_norm)
    stats['min_grad_norm'] = torch.min(grad_norm).item()
    stats['mean_grad_norm'] = torch.mean(grad_norm).item()
    stats['max_grad_norm'] = torch.max(grad_norm).item()

    return stats


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)