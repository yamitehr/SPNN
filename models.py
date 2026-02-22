import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseOrthogonal1x1Conv(nn.Module):
    """
    Base class for orthogonal 1x1 convolutions.

    Subclasses must implement:
        _compute_W(device, dtype) -> [C, C] orthogonal matrix
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def _compute_W(self, device, dtype):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        W = self._compute_W(x.device, x.dtype)   # [C, C]
        weight = W.view(C, C, 1, 1)              # [C_out, C_in, 1, 1]
        return F.conv2d(x, weight)

    def inverse(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        W = self._compute_W(x.device, x.dtype)   # [C, C]
        W_inv = W.t()
        weight = W_inv.view(C, C, 1, 1)
        return F.conv2d(x, weight)

class Cayley1x1Conv(BaseOrthogonal1x1Conv):
    """
    Orthogonal 1x1 convolution using a Cayley transform.

    - Parametrize a skew-symmetric matrix A
    - Build W = (I - A)(I + A)^{-1}, which is orthogonal
    - Apply W as a 1x1 convolution across channels
    - Inverse uses W^T (since W is orthogonal)
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__(channels)
        self.eps = eps

        B = torch.zeros(channels, channels)
        self.A_unconstrained = nn.Parameter(B)

    def _compute_W(self, device, dtype):
        C = self.channels
        B = self.A_unconstrained.to(device=device, dtype=dtype)

        # skew-symmetric
        A = B - B.t()

        I = torch.eye(C, device=device, dtype=dtype)
        # Cayley transform: (I + A) W = (I - A)
        W = torch.linalg.solve(I + A + self.eps * I, I - A)
        return W  # [C, C]

class Householder1x1Conv(BaseOrthogonal1x1Conv):
    """
    Orthogonal 1x1 conv via product of Householder reflections:
        W = H_k ... H_1, H_i = I - 2 v_i v_i^T / ||v_i||^2
    """
    def __init__(self, channels, num_reflections=8, eps=1e-8):
        super().__init__(channels)
        self.num_reflections = num_reflections
        self.eps = eps

        if num_reflections > 0:
            V = torch.randn(num_reflections, channels)
            self.V = nn.Parameter(V)
        else:
            self.register_parameter("V", None)

    def _compute_W(self, device, dtype):
        C = self.channels

        if self.V is None or self.num_reflections == 0:
            return torch.eye(C, device=device, dtype=dtype)

        W = torch.eye(C, device=device, dtype=dtype)
        V = self.V.to(device=device, dtype=dtype)

        for i in range(self.num_reflections):
            v = V[i]
            v = v / (v.norm(p=2) + self.eps)
            H = torch.eye(C, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
            W = H @ W

        return W

class BasePatchOrthogonalMix(nn.Module):
    """
    Base class for orthogonal and invertible patch wise mixer.

    Pipeline (shared for all subclasses):
      - Unfold image into non overlapping patches of size p×p
      - Flatten each patch to a vector of size D = c_in * p * p
      - Apply the same orthogonal W ∈ R^{D×D} to every patch
      - Fold back to image

    Subclasses must implement:
      _compute_W(device, dtype) returns [D, D] orthogonal matrix
    """
    def __init__(self, in_ch, patch_size=4):
        super().__init__()
        self.in_ch = in_ch
        self.patch_size = patch_size
        self.D = in_ch * patch_size * patch_size  # patch vector dim

        # will take non overlapping patches of size patch_size×patch_size and flatten them
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def _compute_W(self, device, dtype):
        """
        need to return an orthogonal matrix W ∈ R^{D×D}
        """
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "H and W must be divisible by patch_size"

        patches = self.unfold(x)              # [B=batch_size, D=c_in * p * p, L=num of patches per images]
        B_, D, L = patches.shape
        patches = patches.transpose(1, 2)     # [B, L, D] – take row vectors

        W_mat = self._compute_W(x.device, x.dtype)  # [D, D]

        # Forward: apply W^T on row vectors
        patches_mixed = patches @ W_mat.T    # [B, L, D]
        patches_mixed = patches_mixed.transpose(1, 2)  # [B, D, L]

        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        y = fold(patches_mixed)              # back to [B, C, H, W]

        return y

    def inverse(self, x):
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "H and W must be divisible by patch_size"

        patches = self.unfold(x)              # [B, D, L]
        B_, D, L = patches.shape
        patches = patches.transpose(1, 2)     # [B, L, D]

        W_mat = self._compute_W(x.device, x.dtype)  # [D, D]

        # Inverse: apply W (since forward used W^T)
        patches_unmixed = patches @ W_mat     # [B, L, D]
        patches_unmixed = patches_unmixed.transpose(1, 2)  # [B, D, L]

        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        y = fold(patches_unmixed)             # [B, C, H, W]
        return y

class PatchCayleyMix(BasePatchOrthogonalMix):
    def __init__(self, in_ch, patch_size=4, eps=1e-6):
        super().__init__(in_ch, patch_size)
        self.eps = eps

        # learn unconstrained parameter B ∈ R^{D×D}
        B = torch.zeros(self.D, self.D)
        self.B = nn.Parameter(B)

    def _compute_W(self, device, dtype):
        B = self.B.to(device=device, dtype=dtype)

        # skew-symmetric A = B - B^T
        A = B - B.t()

        I = torch.eye(self.D, device=device, dtype=dtype)

        # Cayley transform: W = (I - A)(I + A)^{-1}
        W = torch.linalg.solve(I + A + self.eps * I, I - A)
        return W  # [D, D]


class PatchHouseholderMix(BasePatchOrthogonalMix):
    """
    Orthogonal, invertible patch-wise mixer using Householder reflections.
    """
    def __init__(self, in_ch, patch_size=2, num_reflections=4, eps=1e-8):
        super().__init__(in_ch, patch_size)
        self.num_reflections = num_reflections
        self.eps = eps

        if num_reflections > 0:
            V = torch.randn(num_reflections, self.D)
            self.V = nn.Parameter(V)
        else:
            self.register_parameter("V", None)

    def _compute_W(self, device, dtype):
        if self.V is None or self.num_reflections == 0:
            return torch.eye(self.D, device=device, dtype=dtype)

        W = torch.eye(self.D, device=device, dtype=dtype)
        V = self.V.to(device=device, dtype=dtype)

        for i in range(self.num_reflections):
            v = V[i]
            v = v / (v.norm(p=2) + self.eps)
            H = torch.eye(self.D, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)
            W = H @ W

        return W  # [D, D]


class ConvMLP(nn.Module):
    def __init__(self, in_ch, out_ch, scale_bound, hidden_ch,num_classes, img_size: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale_bound = scale_bound
        self.img_size = img_size

        # ===== img_size=256: Block1 (64x64) =====
        if self.img_size == 256 and in_ch == 36 and out_ch == 12:
            # [36,64,64] -> down 32 -> up 64 -> [12,64,64]
            self.net = nn.Sequential(
                nn.Conv2d(36, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                # nn.GroupNorm(1, 128),
                nn.ReLU(),

                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # 64->32
                nn.Conv2d(256, 256, 3, padding=1),
                # nn.GroupNorm(1, 256)
                nn.ReLU(),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 32->64
                nn.Conv2d(128, 12, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        elif self.img_size == 256 and in_ch == 12 and out_ch == 36:
            self.net = nn.Sequential(
                nn.Conv2d(12, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                # nn.GroupNorm(1, 128),
                nn.ReLU(),

                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # 64->32
                nn.Conv2d(256, 256, 3, padding=1),
                # nn.GroupNorm(1, 256),
                nn.ReLU(),


                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 32->64
                nn.Conv2d(128, 36, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ===== img_size=256: Block2 (16x16) =====
        elif self.img_size == 256 and in_ch == 144 and out_ch == 48:
            # [144,16,16] -> down 8 -> up 16 -> [48,16,16]
            self.net = nn.Sequential(
                nn.Conv2d(144, 256, 3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                #nn.GroupNorm(1, 256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),  # 16->8
                nn.Conv2d(512, 512, 3, padding=1),
                #nn.GroupNorm(1, 512),
                nn.ReLU(),

                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),  # 8->16
                nn.Conv2d(256, 48, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        elif self.img_size == 256 and in_ch == 48 and out_ch == 144:
            self.net = nn.Sequential(
                nn.Conv2d(48, 256, 3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                #nn.GroupNorm(1, 256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),  # 16->8
                nn.Conv2d(512, 512, 3, padding=1),
                #nn.GroupNorm(1, 512),
                nn.ReLU(),

                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),  # 8->16
                nn.Conv2d(256, 144, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ===== img_size=256: Block3 (4x4) =====
        elif self.img_size == 256 and in_ch == 576 and out_ch == 192:
            self.net = nn.Sequential(
                nn.Conv2d(576, 512, 3, padding=1), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                #nn.GroupNorm(1, 512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU(),  # 4->2
                nn.Conv2d(1024, 1024, 3, padding=1),
                #nn.GroupNorm(1, 1024),
                nn.ReLU(),

                nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), nn.ReLU(),  # 2->4
                nn.Conv2d(512, 192, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        elif self.img_size == 256 and in_ch == 192 and out_ch == 576:
            self.net = nn.Sequential(
                nn.Conv2d(192, 512, 3, padding=1), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                #nn.GroupNorm(1, 512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU(),  # 4->2
                nn.Conv2d(1024, 1024, 3, padding=1),
                #nn.GroupNorm(1, 1024),
                nn.ReLU(),

                nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), nn.ReLU(),  # 2->4
                nn.Conv2d(512, 576, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ===== img_size=256: Block4 (1x1) =====
        elif self.img_size == 256 and in_ch == 2048 and out_ch == 1024:
            self.net = nn.Sequential(
                nn.Conv2d(2048, 2048, 1), nn.ReLU(),
                nn.Conv2d(2048, 1536, 1), nn.ReLU(),
                nn.Conv2d(1536, 1024, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        elif self.img_size == 256 and in_ch == 1024 and out_ch == 2048:
            self.net = nn.Sequential(
                nn.Conv2d(1024, 1536, 1), nn.ReLU(),
                nn.Conv2d(1536, 2048, 1),
                #nn.GroupNorm(1, 2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)


        # ConvPINNBlock(6 -> 3)
        elif in_ch == 3 and out_ch == 3:
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ConvPINNBlock(48 -> 32)
        elif in_ch == 16 and out_ch == 32:
            self.net = nn.Sequential(
                PixelShuffleLayer(4),  # [16,8,8] -> [1,32,32]
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),  # [32,32,32]
                nn.Conv2d(32, 128, 3, padding=1), nn.ReLU(),  # [128,32,32]
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # [256,16,16]
                nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),  # [512,16,16]
                nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.ReLU(),  # [512,8,8]
                nn.Conv2d(512, 32, 3, padding=1),  # [32,8,8]
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ConvPINNBlock(2048 -> 1024)
        elif in_ch == 1024 and out_ch == 1024:
            self.net = nn.Sequential(
                PixelShuffleLayer(8),  # [1024,1,1] -> [16,8,8]
                nn.Conv2d(16, 128, 3, padding=1), nn.ReLU(),  # [128,8,8]
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),  # [256,8,8]
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),  # [512,4,4]
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),  # [512,4,4]
                nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU(),  # [1024,2,2]
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),  # [1024,1,1]
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        #ConvPINNBlock(1024 -> 40)
        elif in_ch == 984 and out_ch == num_classes:
            self.net = nn.Sequential(
                nn.Conv2d(984, 512, 1), nn.ReLU(),
                nn.Conv2d(512, 256, 1),
                #nn.GroupNorm(1, 256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 1),
                #nn.GroupNorm(1, 128),
                nn.ReLU(),
                nn.Conv2d(128, num_classes, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        elif in_ch == num_classes and out_ch == 984:
            self.net = nn.Sequential(
                nn.Conv2d(num_classes, 128, 1), nn.ReLU(),
                nn.Conv2d(128, 256, 1),
                #nn.GroupNorm(1, 256),
                nn.ReLU(),
                nn.Conv2d(256, 512, 1),
                #nn.GroupNorm(1, 512),
                nn.ReLU(),
                nn.Conv2d(512, 984, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # Default case: old simple 2-layer ConvMLP for all other (in_ch, out_ch)
        elif in_ch > 0:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden_ch, out_ch, 3, padding=1)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # If in_ch == 0, treat it as a learned constant bias per output channel
        else:
            self.net = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x, neg=False):
        if self.in_ch > 0:
            x = self.net(x)
        else:
            B, _, H, W = x.shape
            x = self.net.expand(B, self.out_ch, H, W)

        if self.scale_bound is not None:
            x = torch.tanh(x) * self.scale_bound
            if neg:
                x = -x
            x = x.exp()
        else:
            x = torch.tanh(x)
        return x


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x, return_latent=False):
        y = F.pixel_unshuffle(x, self.r)
        return y, None

    def pinv(self, y, x1_override=None):
        return F.pixel_shuffle(y, self.r)


class PINN(nn.Module):
    def __init__(self, block_cls, layer_channels, img_size: int = 64, num_classes=40, **block_kwargs):
        super().__init__()

        if img_size == 64:
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(2),              # [3,64,64] -> [12,32,32]
                ConvPINNBlock(12, 6, hidden=128, scale_bound=2.0),
                ConvPINNBlock(6, 3, hidden=128, scale_bound=2.0),

                PixelUnshuffleBlock(4),              # [3,32,32] -> [48,8,8]
                ConvPINNBlock(48, 32, hidden=128, scale_bound=2.0),

                PixelUnshuffleBlock(8),              # [32,8,8] -> [2048,1,1]
                ConvPINNBlock(2048, 1024, hidden=128, scale_bound=2.0),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0),
            ])
        elif img_size == 256:
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(4),  # [3,256,256] -> [48,64,64]
                ConvPINNBlock(48, 12, hidden=128, scale_bound=2.0, img_size=256),

                PixelUnshuffleBlock(4),  # [12,64,64] -> [192,16,16]
                ConvPINNBlock(192, 48, hidden=128, scale_bound=2.0, img_size=256),

                PixelUnshuffleBlock(4),  # [48,16,16] -> [768,4,4]
                ConvPINNBlock(768, 192, hidden=128, scale_bound=2.0, img_size=256),

                PixelUnshuffleBlock(4),  # [192,4,4] -> [3072,1,1]
                ConvPINNBlock(3072, 1024, hidden=128, scale_bound=2.0, img_size=256),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0, img_size=256),
            ])
        else: # if img_size == 32
            self.blocks = nn.ModuleList([
                PixelUnshuffleBlock(4),  # [3,32,32] -> [48,8,8]
                ConvPINNBlock(48, 32, hidden=128, scale_bound=2.0),

                PixelUnshuffleBlock(8),  # [32,8,8] -> [2048,1,1]
                ConvPINNBlock(2048, 1024, hidden=128, scale_bound=2.0),

                ConvPINNBlock(1024, num_classes, hidden=128, scale_bound=2.0),
            ])
        #if img_size == 28: # if img_size == 32
        #img_size == 28, img_ch == 1
        # self.blocks = nn.ModuleList([
        #     PixelUnshuffleBlock(4),  # [1,28,28] -> [16,7,7]
        #     ConvPINNBlock(16, 12, hidden=128, scale_bound=2.0),
        #
        #     PixelUnshuffleBlock(7),  # [12,7,7] -> [588,1,1]
        #     ConvPINNBlock(588, 256, hidden=128, scale_bound=2.0),
        #
        #     ConvPINNBlock(256, num_classes, hidden=128, scale_bound=2.0),
        # ])

        # linearizer mnist
        # img_size == 32, img_ch == 1
        # self.blocks = nn.ModuleList([
        #     PixelUnshuffleBlock(4),  # [1,32,32] -> [16,8,8]
        #     ConvPINNBlock(16, 12, hidden=128, scale_bound=2.0),
        #
        #     PixelUnshuffleBlock(8),  # [12,8,8] -> [768,1,1]
        #     ConvPINNBlock(768, 256, hidden=128, scale_bound=2.0),
        #
        #     ConvPINNBlock(256, num_classes, hidden=128, scale_bound=2.0),
        # ])

    def forward(self, x, return_latents=False):
        latents = []
        
        for b in self.blocks:
            # UNIFIED INTERFACE: Every block guarantees a return of (y, z)
            x, z = b(x, return_latent=return_latents)
            latents.append(z)
        return (x, latents) if return_latents else x

    def pinv(self, y, latents=None):
        if latents is not None:
            z_stack = list(reversed(latents))
        else:
            z_stack = [None] * len(self.blocks)
            
        for b in reversed(self.blocks):
            z = z_stack.pop(0)
            y = b.pinv(y, x1_override=z)
        return y



class ConvPINNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64, scale_bound=2., img_size: int = 32, mix_type: str = "cayley"):
        super().__init__()

        layers = {
            "t": (in_ch - out_ch, out_ch, None, hidden),
            "s": (in_ch - out_ch, out_ch, scale_bound, hidden),
            "r": (out_ch, in_ch - out_ch, None, hidden)
        }

        for name, (in_s, out_s, sb, hid) in layers.items():
            setattr(self, name, ConvMLP(in_s, out_s, sb, hid, img_size=img_size))

        if mix_type == "householder":
            self.mix = Householder1x1Conv(in_ch)
        else:
            self.mix = Cayley1x1Conv(in_ch)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, return_latent=False):
        x = self.mix.forward(x)
        x0 = x[:, :self.out_ch, :, :]
        x1 = x[:, self.out_ch:, :, :]
        y = x0 * self.s(x1) + self.t(x1)
        z = x1 if return_latent else None
        return y, z

    def pinv(self, y, x1_override=None):
        x1 = self.r(y) if x1_override is None else x1_override
        x0 = (y - self.t(x1)) * self.s(x1, neg=True)
        x = torch.cat([x0, x1], dim=1)
        return self.mix.inverse(x)

class SPNN(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,
        num_classes: int = 40,
        hidden: int = 128,
        scale_bound: float = 2.0,
        img_size: int = 64,
    ):
        super().__init__()
        assert img_size in (32, 64, 256), "img_size must be 32, 64 or 256"
        assert num_classes < 1024, "num of classes (output size) must be less then 1024"
        self.img_ch = img_ch
        self.num_classes = num_classes
        self.hidden = hidden
        self.scale_bound = scale_bound
        self.img_size = img_size

        self.pinn = PINN(block_cls=None, layer_channels=None, img_size=img_size)

    def forward(self, x_img, return_latents=False):
        B, C, H, W = x_img.shape
        assert C == self.img_ch

        out = self.pinn(x_img, return_latents=return_latents)
        if return_latents:
            y_map, latents = out
        else:
            y_map = out

        logits = y_map.view(B, self.num_classes)

        return (logits, latents) if return_latents else logits

    def pinv_logits(self, logits, latents=None):
        B, C = logits.shape
        assert C == self.num_classes

        y_map_hat = logits.view(B, self.num_classes, 1, 1)

        return self.pinn.pinv(y_map_hat, latents=latents)


class PixelShuffleLayer(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        return F.pixel_shuffle(x, self.r)
