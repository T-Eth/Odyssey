# Better Trigger Inversion code
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pow
from tqdm import tqdm

class EncoderBlock(nn.Module):
    """
    Instances the Encoder block that forms a part of a U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False, dropout_rate=0.3):

        super(EncoderBlock,self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.filter_num,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)
        self.bn_op_2 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)

        # Use Dropout ops as nn.Module instead of nn.functional definition
        # So using .train() and .eval() flags, can modify their behavior for MC-Dropout
        if dropout is True:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
            self.dropout_2 = nn.Dropout(p=dropout_rate)

    def apply_manual_dropout_mask(self, x, seed):
        # Mask size : [Batch_size, Channels, Height, Width]
        dropout_mask = torch.bernoulli(input=torch.empty(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).fill_(self.dropout_rate),
                                       generator=torch.Generator().manual_seed(seed))

        x = x*dropout_mask.to(x.device)

        return x

    def forward(self, x, seeds=None):

        if seeds is not None:
            assert(seeds.shape[0] == 2)

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_1(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[0].item())

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_2(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[1].item())

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block used in the U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False, dropout_rate=0.3):

        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        # Upsample by interpolation followed by a 3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='bilinear',
                                                               align_corners=True),

                                                   nn.Conv2d(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             kernel_size=3,
                                                             padding=1)
                                                  )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_tranposed = nn.ConvTranspose2d(in_channels=self.in_channels,
                                                      out_channels=self.in_channels,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1)

        self.down_sample = EncoderBlock(in_channels=self.in_channels+self.concat_layer_depth,
                                        filter_num=self.filter_num,
                                        dropout=self.dropout,
                                        dropout_rate=self.dropout_rate)

    def forward(self, x, skip_layer, seeds=None):
        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_tranposed(x))
            
        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out, seeds=seeds)
        return out


class EncoderBlock3D(nn.Module):

    """
    Instances the 3D Encoder block that forms a part of a 3D U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False):

        super(EncoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv3d(in_channels=self.filter_num,
                               out_channels=self.filter_num*2,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm3d(num_features=self.filter_num)
        self.bn_op_2 = nn.InstanceNorm3d(num_features=self.filter_num*2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)

        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        return x

class DecoderBlock3D(nn.Module):
    """
    Decoder block used in the 3D U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False):

        super(DecoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout

        # Upsample by interpolation followed by a 3x3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='nearest'),

                                                  nn.Conv3d(in_channels=self.in_channels,
                                                            out_channels=self.in_channels,
                                                            kernel_size=3,
                                                            padding=1)
                                                 )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1)

        if self.dropout is True:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3))
        else:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU())

    def forward(self, x, skip_layer):

        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_transposed(x))
        
        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out)
        return out


class UNet(nn.Module):
    """
     PyTorch class definition for the U-Net architecture for image segmentation
     Parameters:
         n_channels (int) : Number of image channels
         base_filter_num (int) : Number of filters for the first convolution (doubled for every subsequent block)
         num_blocks (int) : Number of encoder/decoder blocks
         num_classes(int) : Number of classes that need to be segmented
         mode (str): 2D or 3D
         use_pooling (bool): Set to 'True' to use MaxPool as downnsampling op.
                             If 'False', strided convolution would be used to downsample feature maps (http://arxiv.org/abs/1908.02182)
         dropout (bool) : Whether dropout should be added to central encoder and decoder blocks (eg: BayesianSegNet)
         dropout_rate (float) : Dropout probability
     Returns:
         out (torch.Tensor) : Prediction of the segmentation map
     """
    def __init__(self, n_channels=1, base_filter_num=64, num_blocks=4, num_classes=5, mode='2D', dropout=False, dropout_rate=0.3, use_pooling=True):

        super(UNet, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()

        self.num_blocks = num_blocks
        self.n_channels = int(n_channels)
        self.n_classes = int(num_classes)
        self.base_filter_num = int(base_filter_num)
        self.enc_layer_depths = []  # Keep track of the output depths of each encoder block
        self.mode = mode
        self.pooling = use_pooling
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if mode == '2D':
            self.encoder = EncoderBlock
            self.decoder = DecoderBlock
            self.pool = nn.MaxPool2d

        elif mode == '3D':
            self.encoder = EncoderBlock3D
            self.decoder = DecoderBlock3D
            self.pool = nn.MaxPool3d
        else:
            print('{} mode is invalid'.format(mode))

        for block_id in range(num_blocks):
            # Due to GPU mem constraints, we cap the filter depth at 512
            enc_block_filter_num = min(int(pow(2, block_id)*self.base_filter_num), 512)  # Output depth of current encoder stage of the 2-D variant
            if block_id == 0:
                enc_in_channels = self.n_channels
            else:
                if self.mode == '2D':
                    if int(pow(2, block_id)*self.base_filter_num) <= 512:
                        enc_in_channels = enc_block_filter_num//2
                    else:
                        enc_in_channels = 512
                else:
                    enc_in_channels = enc_block_filter_num  # In the 3D UNet arch, the encoder features double in the 2nd convolution op


            # Dropout only applied to central encoder blocks -- See BayesianSegNet by Kendall et al.
            if self.dropout is True and block_id >= num_blocks-2:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=True,
                                                          dropout_rate=self.dropout_rate))
            else:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=False))
            if self.mode == '2D':
                self.enc_layer_depths.append(enc_block_filter_num)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv2d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm2d(num_features=self.filter_num),
                                                                nn.LeakyReLU()))
            else:
                self.enc_layer_depths.append(enc_block_filter_num*2) # Specific to 3D U-Net architecture (due to doubling of #feature_maps inside the 3-D Encoder)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv3d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm3d(num_features=self.enc_layer_depths[-1]),
                                                                nn.LeakyReLU()))

        # Bottleneck layer
        if self.mode == '2D':
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            self.bottle_neck_layer = self.encoder(filter_num=bottle_neck_filter_num,
                                                  in_channels=bottle_neck_in_channels)

        else:  # Modified for the 3D UNet architecture
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            self.bottle_neck_layer =  nn.Sequential(nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_in_channels,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_in_channels),

                                                    nn.LeakyReLU(),

                                                    nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_filter_num,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_filter_num),

                                                    nn.LeakyReLU())

        # Decoder Path
        dec_in_channels = int(bottle_neck_filter_num)
        for block_id in range(num_blocks):
            if self.dropout is True and block_id < 2:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=True,
                                                        dropout_rate=self.dropout_rate))
            else:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=False))

            dec_in_channels = self.enc_layer_depths[-1-block_id]

        # Output Layer
        if mode == '2D':
            self.output = nn.Conv2d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)
        else:
            self.output = nn.Conv3d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)

    def forward(self, x, seeds=None):

        if self.mode == '2D':
            h, w = x.shape[-2:]
        else:
            d, h, w = x.shape[-3:]

        # Encoder
        enc_outputs = []
        seed_index = 0
        for stage, enc_op in enumerate(self.contracting_path):
            if stage >= len(self.contracting_path) - 2:
                if seeds is not None:
                    x = enc_op(x, seeds[seed_index:seed_index+2])
                else:
                    x = enc_op(x)
                seed_index += 2 # 2 seeds required per block
            else:
                x = enc_op(x)
            enc_outputs.append(x)

            if self.pooling is True:
                x = self.pool(kernel_size=2)(x)
            else:
                x = self.downsampling_ops[stage](x)

        # Bottle-neck layer
        x = self.bottle_neck_layer(x)
        # Decoder
        for block_id, dec_op in enumerate(self.expanding_path):
            if block_id < 2:
                if seeds is not None:
                    x = dec_op(x, enc_outputs[-1-block_id], seeds[seed_index:seed_index+2])
                else:
                    x = dec_op(x, enc_outputs[-1-block_id])
                seed_index += 2
            else:
                x = dec_op(x, enc_outputs[-1-block_id])


        # Output
        x = self.output(x)

        return x
    
    def train_generator(self, Sa, mask, dataloader, device, loss_func, transform,
                        epochs=30, lr=0.01, tau=0.3, p=2, *, delta: bool=False):
        """
        Eq.(2) objective with τ enforced by branching.
        If delta=True, the UNet outputs Δ and we set Gx = clip(x+Δ, 0,1).
        """
        self.train()
        Sa.eval()
        mask = mask.detach().to(device)
    
        opt = torch.optim.Adam(self.parameters(), lr=lr)
    
        for epoch in range(epochs):
            total = 0.0
            for x, y, *_ in dataloader:
                x = x.to(device)   # [0,1]
                Gx_raw, d_used = self._make_Gx(x, delta)
    
                # τ in input space (L2 per sample)
                dist = d_used.view(x.size(0), -1).norm(p=2, dim=1).mean()
    
                if dist <= tau:
                    # Eq.(2) on features
                    x_n  = transform(x)
                    gx_n = transform(Gx_raw)
                    loss = loss_func(x_n, gx_n, Sa, mask, p=p)
                else:
                    # pull back under τ in input space
                    loss = d_used.view(x.size(0), -1).norm(p=2, dim=1).mean()
    
                opt.zero_grad()
                loss.backward()
                opt.step()
    
                total += loss.item()
    
            loss = total / max(1, len(dataloader))
        return loss


    def train_generator_projection(self, Sa, mask, dataloader, device, loss_func, transform,
                               epochs=30, lr=0.01, tau=0.3, p=2, *, delta: bool=False):
        """
        Eq.(2) with *hard projection* of Δ onto the per-sample L2 τ-ball.
        """
        self.train()
        Sa.eval()
        mask = mask.detach().to(device)
    
        opt = torch.optim.Adam(self.parameters(), lr=lr)
    
        for epoch in range(epochs):
            total = 0.0
            for x, y, *_ in dataloader:
                x = x.to(device)   # [0,1]
                Gx_raw, d_used = self._make_Gx(x, delta)
    
                # Project Δ to L2 τ-ball around x
                delta_vec = d_used.view(x.size(0), -1)
                norm = delta_vec.norm(p=2, dim=1, keepdim=True) + 1e-8
                scale = torch.clamp(tau / norm, max=1.0)
                delta_proj = (delta_vec * scale).view_as(x)
    
                Gx = torch.clamp(x + delta_proj, 0.0, 1.0)
    
                x_n  = transform(x)
                gx_n = transform(Gx)
                loss = loss_func(x_n, gx_n, Sa, mask, p=p)
    
                opt.zero_grad()
                loss.backward()
                opt.step()
    
                total += loss.item()
    
            loss = total / max(1, len(dataloader))
            print(f"epoch: {epoch}, projection version. loss: {loss}")
        return loss


    def train_generator_hinge(self, Sa, mask, dataloader, device, loss_func, transform,
                              epochs=30, lr=0.01, tau=0.3, p=2, lambda_tau=5000, *, delta: bool=False):
        """
        Eq.(2) with a hinge penalty on L2(Δ) - τ (squared mean).
        """
        self.train()
        Sa.eval()
        mask = mask.detach().to(device)
    
        opt = torch.optim.Adam(self.parameters(), lr=lr)
    
        for epoch in range(epochs):
            total = 0.0
            for x, y, *_ in dataloader:
                x = x.to(device)   # [0,1]
                Gx_raw, d_used = self._make_Gx(x, delta)
    
                x_n  = transform(x)
                gx_n = transform(Gx_raw)
    
                dist = d_used.view(x.size(0), -1).norm(p=2, dim=1)
                tau_pen = F.relu(dist - tau)   # per-sample
                loss = loss_func(x_n, gx_n, Sa, mask, p=p) + lambda_tau * (tau_pen.mean() ** 2)
    
                opt.zero_grad()
                loss.backward()
                opt.step()
    
                total += loss.item()
    
            loss = total / max(1, len(dataloader))
            print(f"epoch: {epoch}, hinge version. loss: {loss}")
        return loss

    def _make_Gx(self, x, delta: bool):
        """
        Returns (Gx_raw, delta_used) where:
          - Gx_raw is in [0,1]
          - delta_used = Gx_raw - x  (the actual per-sample perturbation applied)
        """
        if delta:
            # Learn the *delta*; keep unconstrained but softly bounded to [-1,1] via tanh
            d = torch.tanh(self(x))                 # [-1, 1]
            Gx_raw = torch.clamp(x + d, 0.0, 1.0)   # [0, 1]
            return Gx_raw, (Gx_raw - x)
        else:
            # Learn Gx directly; squash to [0,1]
            Gx_raw = torch.sigmoid(self(x))         # [0, 1]
            return Gx_raw, (Gx_raw - x)


def loss_bti_dbf_paper(x_norm, Gx_norm, Sa, m, p=2, eps=1e-12):
    with torch.no_grad():
        feat_x = Sa(x_norm)
    feat_gx = Sa(Gx_norm)

    if feat_x.dim() == 2:  # [B,D]
        if m.dim() == 1:
            m_b = m.unsqueeze(0).expand_as(feat_x)
        elif m.dim() == 2 and m.size(0) in (1, feat_x.size(0)) and m.size(1) == feat_x.size(1):
            m_b = m.expand_as(feat_x)
        else:
            raise ValueError("For flat features, mask must be [D] or [1,D].")
        diff = feat_x - feat_gx
        benign   = torch.norm(diff * m_b + eps, p=p, dim=1)
        backdoor = torch.norm(diff * (1.0 - m_b) + eps, p=p, dim=1)
        return (benign - backdoor).mean()

    elif feat_x.dim() == 4:  # [B,C,H,W]
        B, C, H, W = feat_x.shape
        if m.dim() == 4 and m.shape[1:] in [(C, H, W), (C, 1, 1)]:
            m_b = m
        elif m.dim() == 2 and m.size(1) == C:
            m_b = m.view(1, C, 1, 1)
        elif m.dim() == 1 and m.numel() == C:
            m_b = m.view(1, C, 1, 1)
        else:
            raise ValueError("For spatial features, mask must be [1,C,H,W], [1,C,1,1], [1,C], or [C].")
        m_b = m_b.to(feat_x.device).expand(B, -1, H, W)

        diff = feat_x - feat_gx
        benign   = (((diff * m_b).abs().reshape(B, -1).pow(p).sum(dim=1) + eps).pow(1.0/p))
        backdoor = (((diff * (1.0 - m_b)).abs().reshape(B, -1).pow(p).sum(dim=1) + eps).pow(1.0/p))
        return (benign - backdoor).mean()
    else:
        raise RuntimeError("Unsupported Sa(x) rank.")