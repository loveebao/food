#方案3：注意力引导的特征融合

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet
import timm

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
    @staticmethod

    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out
    
class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        if len(stages_repeats) != 4:
            raise ValueError('expected stages_repeats as list of 4 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        # Patchify stem 
        #将Patchify stem 作为降采样层
        input_channels = 3
        output_channels = 58
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   移除maxpooling
        # Stages
        stage_names = ['stage{}'.format(i) for i in [1, 2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage1(x)           #增加stage1
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)
    
class AttentionFusionFoodNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(AttentionFusionFoodNet, self).__init__()
        
        # 初始化ShuffleNetV2
        self.shufflenet = ShuffleNetV2([2, 2, 6, 2], [58, 116, 232, 464, 1024])
        
        # EfficientNetV2-s 分支
        self.efficientnet = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=num_classes)
        # 初始化EfficientNet-B3
        #self.efficientnet = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
        
        if pretrained:
            # 加载ShuffleNetV2预训练权重
            try:
                state_dict = load_state_dict_from_url(
                    model_urls['shufflenetv2_x1.0'], 
                    progress=True,
                    map_location='cpu'
                )
                # 过滤不匹配的权重
                model_dict = self.shufflenet.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                # 只加载backbone权重，不加载分类头
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if not k.startswith('fc.')}
                model_dict.update(pretrained_dict)
                self.shufflenet.load_state_dict(model_dict, strict=False)
                print(f'Loaded {len(pretrained_dict)}/{len(state_dict)} ShuffleNetV2 pretrained weights')
            except Exception as e:
                print(f'ShuffleNetV2 pretrained weight loading failed: {str(e)}')
                print('Training ShuffleNetV2 from scratch...')
            
            # 加载EfficientNet预训练权重
            try:
                self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                print('Loaded EfficientNet pretrained weights')
            except Exception as e:
                print(f'EfficientNet pretrained weight loading failed: {str(e)}')
                print('Training EfficientNet from scratch...')
        
        # 移除两个网络最后的分类层
        self.shufflenet.fc = nn.Identity()
        self.efficientnet._fc = nn.Identity()
        
        # 特征尺寸适配层:输出尺寸
        self.shufflenet_adapter = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.efficientnet_adapter = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(1024+1536, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 分类头（与融合特征维度512匹配）
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),    # 输入维度改为512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # ShuffleNet特征提取
        x1 = self.shufflenet.conv1(x)
        x1 = self.shufflenet.stage1(x1)
        x1 = self.shufflenet.stage2(x1)
        x1 = self.shufflenet.stage3(x1)
        x1 = self.shufflenet.stage4(x1)
        x1 = self.shufflenet.conv5(x1)  # [B, 1024, H1, W1] (如7×7)
        
        # EfficientNet特征提取
        x2 = self.efficientnet.extract_features(x)  # [B, 1536, H2, W2] (可能不同于H1,W1)
        
        # 统一特征图尺寸
        if x1.size()[2:] != x2.size()[2:]:
            # 方法1: 使用自适应平均池化统一尺寸
            target_size = min(x1.size(2), x2.size(2)), min(x1.size(3), x2.size(3))
            x1 = F.adaptive_avg_pool2d(x1, target_size)
            x2 = F.adaptive_avg_pool2d(x2, target_size)
            
            # 或者方法2: 插值调整尺寸(二选一)
            # x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # 特征尺寸适配
        x1_adapted = self.shufflenet_adapter(x1)  # [B, 512, H, W]
        x2_adapted = self.efficientnet_adapter(x2)  # [B, 512, H, W]
        
        # 计算注意力权重(基于融合后的特征)
        attention_input = torch.cat([x1, x2], dim=1)  # 使用原始特征计算注意力
        attention_weights = self.attention(attention_input)  # [B, 2, H, W]
        
        # 注意力加权融合(确保所有张量尺寸一致)
        x1_weighted = x1_adapted * attention_weights[:, 0:1, :, :]
        x2_weighted = x2_adapted * attention_weights[:, 1:2, :, :]
        
        # 融合后的特征形状应该是[B,512,H,W]
        fused_features = x1_weighted + x2_weighted
        
        # 全局平均池化后形状变为[B,512]
        pooled = fused_features.mean([2, 3])  
        
        # 分类
        out = self.classifier(pooled)
        
        return out

def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def shufflenet3_v2_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [2, 2, 6, 2], [58, 116, 232, 464, 1024], **kwargs)