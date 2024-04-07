
import torch
import torch.nn as nn
import torch.nn.functional as F

class DConv(nn.Module):#深度可分离卷积
    def __init__(self,inchannel,outchannel):
        super(DConv,self).__init__()
        self.dconv=nn.Sequential(
            nn.Conv2d(inchannel,inchannel,kernel_size=3,padding=1,groups=inchannel),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=False),
            nn.Conv2d(inchannel,outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=False)
        )
    def forward(self,x):
        out=self.dconv(x)
        return out
class MDTA(nn.Module):
    def __init__(self, in_dim,layers):
        super(MDTA, self).__init__()
        self.query_conv = DConv(in_dim,in_dim)
        self.key_conv=DConv(in_dim,in_dim)
        self.value_conv=DConv(in_dim,in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.layers=layers
        #self.ln = nn.LayerNorm()

    def forward(self, x):
        # x = x.to(next(self.parameters()).device)
            # x = nn.LayerNorm([x.size(1), x.size(2), x.size(3)])(x)
        x = F.layer_norm(x,[x.size(1), x.size(2), x.size(3)])
        #x = self.ln(x)  # 使用输入张量的维度来定义 LayerNorm 的归一化形状

        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        if self.layers<2:
            energy = torch.bmm(proj_key, proj_query)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)

        else:
            energy = torch.bmm(proj_query, proj_key)
            # print("第几层", self.layers)
            # print("energy大小", energy.shape)
            proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
            #print("v的大小",proj_value.shape)
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), -1, x.size(2), x.size(3))
        out = self.gamma * out + x
        return out

class GDFN(nn.Module):
    def __init__(self,in_dim,ratio):
        super(GDFN,self).__init__()
        self.upconv1=nn.Conv2d(in_dim,ratio*in_dim,kernel_size=1)
        self.lowconv1 = nn.Conv2d(in_dim, ratio*in_dim, kernel_size=1)

        self.uppath=DConv(ratio*in_dim,ratio*in_dim)
        self.lowpath=DConv(ratio*in_dim,ratio*in_dim)
        self.relu=nn.SiLU(inplace=False)
        self.finconv=nn.Conv2d(ratio*in_dim,in_dim,kernel_size=1)
    def forward(self,x):
        temp=x

        #x = nn.LayerNorm(x.size()[1:])(x)  # 使用输入张量的维度来定义 LayerNorm 的归一化形状
        x = F.layer_norm(x,[x.size(1), x.size(2), x.size(3)])
        #x = nn.LayerNorm([x.size(1), x.size(2), x.size(3)])(x)
        #上层
        up=self.upconv1(x)
        up=self.uppath(up)
        #下层
        low=self.lowconv1(x)
        low=self.lowpath(low)
        low=self.relu(low)
        out=low*up
        out=self.finconv(out)
        out=temp+out
        return out
class Transformerblock(nn.Module):
    def __init__(self,in_dim,layers):
        super(Transformerblock,self).__init__()
        self.first=MDTA(in_dim,layers)
        self.second=GDFN(in_dim,ratio=2)
    def forward(self,x):
        x=self.first(x)
        x=self.second(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double depthwise separable conv"""

    def __init__(self, in_channels,out_channels,nums,layers):
        super().__init__()
        self.blocks=nn.Sequential(*[Transformerblock(in_channels,layers)for _ in range(nums)])
        self.maxpool_conv = nn.MaxPool2d(2)
        self.conv=DConv(in_channels,out_channels)
    def forward(self, x):
        x=self.maxpool_conv(x)
        x=self.blocks(x)
        x=self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then double depthwise separable conv"""
    def __init__(self, in_channels,out_channels,nums,layers,bilinear=True):
        super().__init__()
        self.layers=layers
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.blocks = nn.Sequential(*[Transformerblock(in_channels,self.layers) for _ in range(nums)])
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DConv(in_channels,out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.blocks = nn.Sequential(*[Transformerblock(in_channels,self.layers) for _ in range(nums)])
            self.conv = DConv(in_channels,out_channels)

    def forward(self, x1, x2):
        x1= self.blocks(x1)
        x1 =self.up(x1)
        x1= self.conv(x1)
        #print("底层放大后的大小",x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        #print("合并后的大小",x.shape)

        return self.conv(x)

class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(OutConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels,out_channels,kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )

        def forward(self, x):
            return self.conv(x)
class Encoder(nn.Module):
    def __init__(self,Downnumblocks, n_channels=3, n_classes=64,bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DConv(n_channels, 64)
        self.down1 = Down(64,128,Downnumblocks[0],0)
        self.down2 = Down(128,256,Downnumblocks[1],1)
        self.down3 = Down(256,512,Downnumblocks[2],2)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,1024,Downnumblocks[3],3)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, x4, x3, x2, x1
class WeightedFeatureAggregation(nn.Module):
    def __init__(self, num_features=4, initial_weights=[0.5,0.3,0.15,0.05]):
        super(WeightedFeatureAggregation, self).__init__()
        self.num_features = num_features
        # 将权重定义为可学习参数
        self.weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=True)

    def forward(self, feature_maps):
        # 对每个特征图应用权重
        weighted_feature_maps = [feature_maps[i] * self.weights[i] for i in range(self.num_features)]
        # 将加权后的特征图相加
        fused_feature_map = torch.stack(weighted_feature_maps, dim=0).sum(dim=0)
        return fused_feature_map

class Decoder(nn.Module):
    def __init__(self, downblocks, Upblocks, n_classes=64, bilinear=True):
        super(Decoder, self).__init__()
        self.downblocks = downblocks
        self.Upblocks = Upblocks
        self.branch1 = Encoder(Downnumblocks=self.downblocks)
        self.up1 = Up(1024, 512, self.Upblocks[0], 3, bilinear)
        self.up2 = Up(512, 256, self.Upblocks[1], 2, bilinear)
        self.up3 = Up(256, 128, self.Upblocks[2], 1, bilinear)
        self.up4 = Up(128, 64, self.Upblocks[3], 0, bilinear)
        self.upconv1=OutConv(512,n_classes)
        self.upconv2 = OutConv(256, n_classes)
        self.upconv3 = OutConv(128, n_classes)

        self.upsample =nn.Sequential(
            nn.Conv2d(64, 64 * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor=2))
        self.weight=WeightedFeatureAggregation()
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x, x4, x3, x2, x1 = self.branch1(x)
        up1 = self.up1(x, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        up1p=self.upconv1(up1)
        up1p=self.upsample(up1p)
        up1p=self.upsample(up1p)
        up1pp = self.upsample(up1p)
        up2p = self.upconv2(up2)
        up2p = self.upsample(up2p)
        up2pp = self.upsample(up2p)
        up3p = self.upconv3(up3)
        up3pp = self.upsample(up3p)
        feature_maps = [up1pp, up2pp, up3pp, up4]
        out=self.weight(feature_maps)
        logits = self.outc(out)
        return logits


class CBAMLayer(nn.Module):  #注意力机制
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Upsamleblock(nn.Module):
    def __init__(self, in_channels):
        super(Upsamleblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
        # 添加 CBAM 模块
        self.cbam = CBAMLayer(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        # 添加 CBAM 模块
        x = self.cbam(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):

    def __init__(self,in_channel,filters,num_upsample,upblocks=[2,2,2,2],downblocks=[2,2,2,2]):
        super(Generator, self).__init__()
        self.mod_pad_h=0
        self.mod_pad_w=0
        self.decoder=Decoder(downblocks=downblocks,Upblocks=upblocks,n_classes=filters)
        upsamplelayers=[]
        for _ in range(num_upsample):
            upsamplelayers+=[Upsamleblock(filters)]
        self.upsampling=nn.Sequential(*upsamplelayers)
        self.finconv=nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, in_channel, kernel_size=3, stride=1, padding=1),
        )
    def pre_process(self,x):
        
        _,_,h,w=x.shape
        if h % 8!= 0:
            self.mod_pad_h = 8 - h % 8
        if w % 8 != 0:
            self.mod_pad_w = 8 - w % 8
        x= F.pad(x, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
        return x
    def post_process(self,x):
        _,_,h,w=x.shape
        x = x[:, :, 0:h - self.mod_pad_h*4 , 0:w - self.mod_pad_w*4]
        return x

    def forward(self,x):
        x=self.pre_process(x)
        x=self.decoder(x)
        x=self.upsampling(x)
        x=self.finconv(x)
        x=self.post_process(x)
        return  x

batch_size = 1
num_channels = 3
image_height = 35
image_width = 36

# 创建随机输入图像（仅用于示例）
dummy_images = torch.randn(batch_size, num_channels, image_height, image_width)
dummy_images = torch.clamp(dummy_images, 0, 1)


u_model = Generator(in_channel=3,filters=64,num_upsample=2)


# 前向传播
output_features = u_model(dummy_images)

# 打印最终特征图的大小
print(output_features.shape)