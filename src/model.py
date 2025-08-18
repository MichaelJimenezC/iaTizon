import torch
import torch.nn as nn
from typing import Tuple

# ---------------- Utils ----------------
def get_norm(norm_type: str, num_channels: int, num_groups: int = 32):
    norm_type = (norm_type or "gn").lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        g = min(num_groups, num_channels)
        return nn.GroupNorm(g, num_channels)
    else:
        raise ValueError(f"norm_type desconocido: {norm_type} (usa 'bn' o 'gn')")

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.net(x)
        return x * w

class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask

# --------------- Bloque residual (Lite) ----------------
class BasicBlock(nn.Module):
    """
    Bloque residual básico con:
    - Norma configurable (GN por defecto)
    - Dropout2d tras la primera conv (spatial)
    - Squeeze-and-Excitation opcional
    - Stochastic Depth en el camino residual
    """
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        p: float = 0.1,         
        norm: str = "gn",
        use_se: bool = True,
        drop_path: float = 0.0,
        gn_groups: int = 32,
    ):
        super().__init__()
        Norm = lambda c: get_norm(norm, c, gn_groups)

        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = Norm(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.drop2d = nn.Dropout2d(p) if p and p > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = Norm(planes)

        self.se = SEBlock(planes) if use_se else nn.Identity()

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                Norm(planes),
            )

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop2d(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.drop_path(out)
        out = out + identity
        out = self.relu(out)
        return out

# --------------- ResNet18 Lite Anti-Overfit ----------------
class ResNet18(nn.Module):
    """
    Versión ligera pensada para entrenar desde cero con datasets medianos/pequeños:
    - Anchuras: 32-64-128 (en lugar de 64-128-256-512)
    - 3 etapas (layer1..layer3); layer4 = Identity para compatibilidad
    - GroupNorm por defecto (mejor que BN si el batch < 32)
    - Dropout2d en bloques + Dropout en la cabeza
    - Squeeze-and-Excitation opcional
    - Stochastic Depth ligero (sd_prob) lineal a lo largo de los bloques
    """
    def __init__(
        self,
        num_classes: int,
        p_fc: float = 0.3,
        norm: str = "gn",               
        use_se: bool = True,
        p_block: float = 0.10,           
        sd_prob: float = 0.10,          
        widths: Tuple[int, int, int] = (32, 64, 128),
        blocks_per_stage: Tuple[int, int, int] = (2, 2, 2),
        gn_groups: int = 32,
        use_stem_pool: bool = True,
    ):
        super().__init__()
        self.in_planes = widths[0]
        self.norm = norm
        self.use_se = use_se
        self.p_block = p_block
        self.sd_prob = float(sd_prob)
        self.gn_groups = gn_groups

        Norm = lambda c: get_norm(norm, c, gn_groups)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            Norm(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1) if use_stem_pool else nn.Identity(),
        )

        self._total_blocks = sum(blocks_per_stage)
        self._block_idx = 0

        self.layer1 = self._make_layer(widths[0], blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(widths[1], blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(widths[2], blocks_per_stage[2], stride=2)
        self.layer4 = nn.Identity() 

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p_fc) if p_fc and p_fc > 0 else nn.Identity()
        self.fc   = nn.Linear(widths[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def _next_drop_path(self) -> float:
        if self._total_blocks <= 1 or self.sd_prob <= 0:
            return 0.0
        frac = self._block_idx / (self._total_blocks - 1)
        dp = self.sd_prob * frac
        self._block_idx += 1
        return float(dp)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = []
        for b in range(blocks):
            s = stride if b == 0 else 1
            dp = self._next_drop_path()
            layers.append(BasicBlock(
                in_planes=self.in_planes,
                planes=planes,
                stride=s,
                p=self.p_block,
                norm=self.norm,
                use_se=self.use_se,
                drop_path=dp,
                gn_groups=self.gn_groups,
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
