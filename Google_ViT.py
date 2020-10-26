# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:34:49 2020

@author: mthossain
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:37:52 2020

@author: mthossain based on LucidRains
"""
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

class ImageTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  # e.g. (32/4)**2= 64
        patch_dim = channels * patch_size ** 2  # e.g. 3*8**2 = 64*3

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper
        self.patch_conv= nn.Conv2d(3,dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
        
        #self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, mlp_dim)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        # self.af1 = nn.GELU()
        # self.do1 = nn.Dropout(dropout)
        # self.nn2 = nn.Linear(mlp_dim, num_classes)
        # torch.nn.init.zeros_(self.nn2.weight)
        # torch.nn.init.zeros_(self.nn2.bias)
        # self.do2 = nn.Dropout(dropout)

    def forward(self, img, mask = None):
        p = self.patch_size

        x = self.patch_conv(img) # each of 64 vecotrs is linearly transformed with a FFN equiv to E matmul
        #x = torch.matmul(x, self.E)
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors in rows representing 64 patches, each 64*3 long

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask) #main game

        x = self.to_cls_token(x[:, 0])
        
        x = self.nn1(x)
        # x = self.af1(x)
        # x = self.do1(x)
        # x = self.nn2(x)
        # x = self.do2(x)
        
        return x


BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100

DL_PATH = "C:\Pytorch\Spyder\CIFAR10_data"
# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
transform = torchvision.transforms.Compose(
     [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=True,
                                        download=False, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=False,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

N_EPOCHS = 150

model = ImageTransformer(image_size=32, patch_size=4, num_classes=10, channels=3,
            dim=64, depth=6, heads=8, mlp_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)

print('Execution time')

PATH = ".\ViTnet_Cifar10_4x4_aug_1.pt"
torch.save(model.state_dict(), PATH)


# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
