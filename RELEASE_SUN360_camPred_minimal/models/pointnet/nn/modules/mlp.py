from torch import nn
import torch.nn.functional as F

from .conv import Conv1d, Conv2d
from .linear import FC


class MLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 dropout_prob=0.0,
                 bn=True,
                 bn_momentum=0.1):
        """Multilayer perceptron

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            dropout_prob (float or None): dropout probability
            bn (bool): whether to use batch normalization
            bn_momentum (float)

        """
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        for ind, out_channels in enumerate(mlp_channels):
            self.append(FC(in_channels, out_channels, relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        # Do not use modules due to ModuleList.
        assert dropout_prob >= 0.0
        self.dropout_prob = dropout_prob

    def forward(self, x):
        for module in self:
            assert isinstance(module, FC)
            x = module(x)
            if self.training and self.dropout_prob > 0.0:
                x = F.dropout(x, p=self.dropout_prob, training=True)
        return x

    def init_weights(self, init_fn=None):
        for module in self:
            assert isinstance(module, FC)
            module.init_weights(init_fn)

    def extra_repr(self):
        return 'dropout_prob={}'.format(self.dropout_prob) if self.dropout_prob > 0.0 else ''


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 dropout_prob=0.0,
                 bn=True,
                 bn_momentum=0.1):
        """Multilayer perceptron shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            dropout_prob (float or None): dropout ratio
            bn (bool): whether to use batch normalization
            bn_momentum (float)

        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.ndim = ndim

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv2d
        else:
            raise ValueError('SharedMLP only supports ndim=(1, 2).')

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1, relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        # Do not use modules due to ModuleList.
        assert dropout_prob >= 0.0
        self.dropout_prob = dropout_prob

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1d, Conv2d))
            x = module(x)
            if self.training and self.dropout_prob > 0.0:
                if self.ndim == 1:
                    x = F.dropout(x, p=self.dropout_prob, training=True)
                elif self.ndim == 2:
                    x = F.dropout2d(x, p=self.dropout_prob, training=True)
                else:
                    raise ValueError('SharedMLP only supports ndim=(1, 2).')
        return x

    def init_weights(self, init_fn=None):
        for module in self:
            assert isinstance(module, (Conv1d, Conv2d))
            module.init_weights(init_fn)
