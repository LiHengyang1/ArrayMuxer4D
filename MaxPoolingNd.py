import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class MaxPool4d(nn.Module):
    """
    MaxPool4d
    """
    def __init__(self, ker_shape, stride):
        super(MaxPool4d, self).__init__()
        self.ker_shape = ker_shape
        self.stride = stride

    def forward(self, x):
        # input = [n, in_ch, in_dim1, in_dim2, in_dim3, in_dim4]
        # kernel = [ker_dim1, ker_dim2, ker_dim3, ker_dim4]
        # output = [n, in_ch, out_dim1, out_dim2, out_dim3, out_dim4]

        # str_shape = [n, inc, ker_dim, out_dim]
        str_shape = (x.shape[0],
            x.shape[1],
            self.ker_shape[0],
            self.ker_shape[1],
            self.ker_shape[2],
            self.ker_shape[3],
            int((x.shape[2] - self.ker_shape[0])/self.stride + 1),
            int((x.shape[3] - self.ker_shape[1])/self.stride + 1),
            int((x.shape[4] - self.ker_shape[2])/self.stride + 1),
            int((x.shape[5] - self.ker_shape[3])/self.stride + 1))

        # x_strides = [n, inc, in_dim, in_stride=in_dim]
        x_strides = (x.stride()[0],
            x.stride()[1],
            x.stride()[2] * 2,
            x.stride()[3] * 2,
            x.stride()[4] * 2,
            x.stride()[5] * 2,
            x.stride()[2],
            x.stride()[3],
            x.stride()[4],
            x.stride()[5])
        # print(x_strides)
        # print(str_shape)
        x = tc.as_strided(x,
            size=str_shape,
            stride=x_strides)




        return tc.amax(x, dim=(2,3,4,5))