import torch.nn as nn
import torch
from torchvision import models

class MultiModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiModalFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn_img = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attn_tir = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.cross_attn_ffn_textual = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, img_tensor, tir_tensor):
        img_tensor = img_tensor.reshape(*img_tensor.shape[:-2], -1)
        tir_tensor = tir_tensor.reshape(*tir_tensor.shape[:-2], -1)
        
        img_tensor = img_tensor.permute(2, 0, 1)
        tir_tensor = tir_tensor.permute(2, 0, 1)
        
        img_self_attention, _ = self.self_attn_img(img_tensor, img_tensor, img_tensor)
        tir_self_attention, _ = self.self_attn_tir(tir_tensor, tir_tensor, tir_tensor)
        
        img_tensor = img_tensor + img_self_attention
        tir_tensor = tir_tensor + tir_self_attention
        
        out, _ = self.cross_attn_ffn_textual(query=img_tensor, key=tir_tensor, value=tir_tensor)
        out_res = self.ffn1(out.permute(1, 0, 2)).permute(1, 0, 2)
        
        out = out + out_res
        # print(out.shape)
        return out    
        
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.fusion = MultiModalFusion(512, 8)
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i, (key, value) in enumerate(list(self.frontend.state_dict().items())):
                self.frontend.state_dict()[key].data[:] = list(
                    mod.state_dict().items())[i][1].data[:]

    def forward(self, img, tir=None):
        if tir is not None:
            img = self.frontend(img)
            tir = self.frontend(tir) 
            
            fusion = self.fusion(img, tir).permute(1,2,0)
            fusion = fusion.view(*fusion.shape[:-1], 64, 80)

            fusion = self.backend(fusion)

            fusion = self.output_layer(fusion)
            return fusion
            
        else:
            img = self.frontend(img)
            img = self.backend(img)
            img = self.output_layer(img)
            return img 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



