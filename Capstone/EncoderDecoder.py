from MiDaS.midas.midas_net import MidasNet
import torch
import torch.nn as nn
from collections import OrderedDict
class YOLOLayers(nn.Module):
    def __init__(self, config, is_training=True):
        super(YOLOLayers, self).__init__()
        self.config = config
        self.training = is_training
        
        _out_filters = [256, 512, 1024, 2048]
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.embedding0 = self._make_embedding([512, 2048], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 1024], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 512], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

class MidasYoloModel(nn.Module):

    def __init__(self, config, midas_path):
        super(MidasYoloModel, self).__init__()
        self.midas = MidasNet(midas_path, non_negative = True)
        self.pretrained = self.midas.pretrained
        self.scratch = self.midas.scratch
        self.yolo = YOLOLayers(config, is_training = False)
        print('Loading yolo pretrained')
        state_dict = torch.load('trained_yolo_model.pth', map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.yolo.load_state_dict(state_dict, strict = False)


    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        midas_out = self.scratch.output_conv(path_1)

        x2, x1, x0 = layer_2, layer_3, layer_4

        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        
        #  yolo branch 0
        out0, out0_branch = _branch(self.yolo.embedding0, x0)
        #  yolo branch 1
        x1_in = self.yolo.embedding1_cbl(out0_branch)
        x1_in = self.yolo.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.yolo.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.yolo.embedding2_cbl(out1_branch)
        x2_in = self.yolo.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.yolo.embedding2, x2_in)

        return (out0, out1, out2), torch.squeeze(midas_out, dim=1)
