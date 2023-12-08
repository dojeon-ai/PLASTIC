import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as aug


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomSpatialMaskAug(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w

        mask_ratio = self.mask_ratio
        s = int(h * w)
        len_keep = round(s * (1 - mask_ratio))

        # sample random noise
        noise = torch.cuda.FloatTensor(n, s).normal_()
        # noise = torch.rand(n, s)  # noise in cpu
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, s], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # repeat channel_wise
        mask = mask.repeat(1, c)
        mask = mask.reshape(n, c, h, w)

        # mask-out input
        x = x * mask

        return x


class Augmentation(nn.Module):
    def __init__(self, obs_shape, aug_types=[], mask_ratio=None):
        super().__init__()
        self.layers = []
        for aug_type in aug_types:
            # kornia random crop is 20x slower than default (2023.02.27)
            # https://github.com/kornia/kornia/issues/1559
            if aug_type == 'random_shift_kornia':
                _, _, W, H = obs_shape
                self.layers.append(nn.ReplicationPad2d(4))
                self.layers.append(aug.RandomCrop((W, H)))

            elif aug_type == 'random_shift':
                _, _, W, H = obs_shape
                self.layers.append(RandomShiftsAug(pad=4))

            elif aug_type == 'cutout':
                self.layers.append(aug.RandomErasing(p=0.5))
            
            elif aug_type == 'h_flip':
                self.layers.append(aug.RandomHorizontalFlip(p=0.1))

            elif aug_type == 'v_flip':
                self.layers.append(aug.RandomVerticalFlip(p=0.1))

            elif aug_type == 'rotate':
                self.layers.append(aug.RandomRotation(degrees=5.0))

            elif aug_type == 'intensity':
                self.layers.append(Intensity(scale=0.05))

            elif aug_type == 'spatial_mask':
                if mask_ratio is None:
                    mask_ratio = 0.25
                self.layers.append(RandomSpatialMaskAug(mask_ratio))

            else:
                raise ValueError

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__=='__main__':
    import time
    device = torch.device('cuda')

    obs_shape = (32, 4, 84, 84)
    input_tensor = torch.randn(obs_shape).to(device)
    aug_types = ['random_shift_kornia']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    
    # kornia augmentation speed
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)

    # PyTorch augmentation speed
    aug_types = ['random_shift']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)

    # Random Spaital Augmentation
    aug_types = ['spatial_mask']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)