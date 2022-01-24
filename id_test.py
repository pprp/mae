import torch
from PIL import Image
import numpy as np
import skdim
import einops
from torchvision import transforms
import matplotlib.pyplot as plt


def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = 14  # patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

    return x


def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 14  # patch_embed.patch_size[0]
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


IMG1 = "/home/pdluser/dataset/imagenet-mini/ILSVRC2012/train/n02490219/n02490219_4355.JPEG"
IMG2 = "/home/pdluser/dataset/imagenet-mini/ILSVRC2012/train/n02490219/n02490219_3143.JPEG"

img1 = Image.open(IMG1)
img2=  Image.open(IMG2)

img1 = img1.resize((224, 224))
img2 = img2.resize((224, 224))

T = transforms.ToTensor()

t_img1 = T(img1)
t_img2 = T(img2)

t_img = torch.cat([t_img1.unsqueeze(0),t_img2.unsqueeze(0)], dim=0)

print(t_img.shape)

print("before patchify: ",t_img.shape)

t_img = einops.rearrange(
    t_img, 'b c (k1 h) (k2 w) -> b (k1 k2) (c h w)', k1=14, k2=14)

print("after patchify: ", t_img.shape) # 1 256 588 

ranks = torch.var(t_img, dim=2)

print("var calculation shape: ", ranks.shape)

ids = torch.argsort(ranks, dim=1, descending=True)
ids_restore = torch.argsort(ids, dim=1, descending=False)

# 保留30个patch 不进行mask
ids_keep = ids[:, :25]

# t_img: b,l,d
# x_masked = torch.gather(
#     t_img, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 768))

# print(x_masked.shape)

mask = torch.ones([1, 196])

mask[:, :49] = 0

mask = torch.gather(mask, dim=1, index=ids_restore)

mask = mask.unsqueeze(-1).repeat(1, 1, 768)

print("after repeat: ", mask.shape )

t_img = t_img *(1 - mask)

print("mask shape: ", mask.shape)

plt.rcParams['figure.figsize'] = [24, 24]

# (t_img, 'b c (k1 h) (k2 w) -> b (k1 k2) (c h w)', k1=14, k2=14)

T2 = transforms.ToPILImage()

s_img = einops.rearrange(t_img, 'b (k1 k2) (c h w) -> b c (k1 h) (k2 w)', k1=14, k2=14, c=3, h=16, w=16).squeeze(0)

s_img = T2(s_img)

print(s_img.size)


# plt.imshow(s_img)

plt.subplot(1,2,1)
plt.imshow(T2(tmp_img.squeeze(0)))

plt.subplot(1, 2, 2)
plt.imshow(s_img)

plt.savefig("id_test.png")


# print(ids)
# print(ids_keep)

# id = skdim.id.ESS().fit_transform(X=t_img.reshape(196,-1).numpy())

# print(id)
