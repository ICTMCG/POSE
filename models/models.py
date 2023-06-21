import torch.nn as nn
import torch_dct as dct

def get_input_data(input_img, data='dct'):
    if data == 'dct':
        return dct.dct_2d(input_img)
    elif data == 'img':
        return input_img

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

class Simple_CNN(nn.Module):
    def __init__(self, class_num, out_feature_result=False):
        super(Simple_CNN, self).__init__()
        nf = 64
        nc = 3
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )

        self.fc = nn.Linear(nf * 8 * 8 * 8, nf * 8, bias=True)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, class_num, bias=True)
        )
        self.out_feature_result=out_feature_result

    def forward(self, input, data='dct'):
        input = get_input_data(input, data)
        embedding = self.main(input)
        feature = embedding.view(embedding.shape[0], -1)
        feature = self.fc(feature)
        cls_output = self.classification_head(feature)

        if self.out_feature_result:
            return cls_output, feature
        else:
            return cls_output



