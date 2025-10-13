import torch
import torch.nn as nn
from einops import rearrange
import math


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        '''Dual-Direction Distortion Modeling'''
        self.AFE_h = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 1, 5), stride=(1, 1, 5)),
            nn.BatchNorm3d(32),
            nn.PReLU()
        )
        self.SFE_h = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 1, 3), stride=(1, 1, 1), dilation=(1, 1, 5), padding=(0, 0, 5)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1, 1, 5), stride=(1, 1, 5)),
            nn.BatchNorm3d(32),
            nn.PReLU()
        )
        self.AFE_v = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 5, 1), stride=(1, 5, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU()
        )
        self.SFE_v = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 3, 1), stride=(1, 1, 1), dilation=(1, 5, 1), padding=(0, 5, 0)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1, 5, 1), stride=(1, 5, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU()
        )

        '''Spatial-Angular Quality Learning'''
        self.SAQL_h = SAQL()
        self.SAQL_v = SAQL()

        '''TransGRU Temporal Sequence Learning'''
        self.TransGRU_h = TransGRU()
        self.TransGRU_v = TransGRU()

        '''Regression'''
        self.linear_1 = nn.Linear(512, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.relu = nn.PReLU()
        self.flat = nn.Flatten()

    def forward(self, x_h, x_v):
        # Dual-Direction Distortion Modeling
        x_h_1 = self.SFE_h(x_h)
        x_h_2 = self.AFE_h(x_h)
        x_h = torch.cat((x_h_1, x_h_2), dim=1)
        x_v_1 = self.SFE_v(x_v)
        x_v_2 = self.AFE_v(x_v)
        x_v = torch.cat((x_v_1, x_v_2), dim=1)

        # Spatial-Angular Quality Learning
        x_h = self.SAQL_h(x_h)
        x_v = self.SAQL_v(x_v)

        # TransGRU Temporal Sequence Learning
        x_h = self.TransGRU_h(x_h)
        x_v = self.TransGRU_v(x_v)

        x_all = torch.cat([x_h, x_v], dim=1)

        out = self.linear_1(x_all)
        out = self.relu(out)
        q = self.linear_2(out)

        return q


class SAQL(nn.Module):

    def __init__(self):
        super(SAQL, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.PReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class GRU(nn.Module):
    # input: n c T 1 1
    # output: n c

    def __init__(self, input_size, hidden_size, device="cuda:0", batch_first=True):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):  # n c T 1 1
        t = torch.squeeze(x, dim=3)
        t = torch.squeeze(t, dim=3)
        t = t.permute([0, 2, 1])
        r, h1 = self.rnn(t, self._get_initial_state(t.size(0), self.device))
        r = r.permute([0, 2, 1])
        f = self.pool(r).squeeze(2)
        return f

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


class Trans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(Trans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


class TransGRU(nn.Module):

    def __init__(self):
        super(TransGRU, self).__init__()
        self.avg_pooling = nn.AvgPool3d(kernel_size=(1, 16, 16), stride=1, padding=0)
        self.gru = GRU(256, 256, batch_first=True)
        self.angRes = 5
        self.channels = 256
        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.trans = Trans(self.channels, self.angRes, self.MHSA_params)

    def forward(self, x):
        for m in self.modules():
            m.h = x.size(-2)
            m.w = x.size(-1)
        ang_position = self.pos_encoding(x, dim=[2], token_dim=self.channels)
        for m in self.modules():
            m.ang_position = ang_position
        x = self.trans(x)
        x = self.avg_pooling(x)
        x = self.gru(x)

        return x


if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile

    input1 = torch.randn(2, 1, 5, 32, 160).cuda()
    input2 = torch.randn(2, 1, 5, 160, 32).cuda()
    flops, params = profile(net, inputs=(input1, input2,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
