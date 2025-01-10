from thop import profile
from defaults import get_default_cfg
from models.seqnet_da import SeqNetDa

# from torchstat import stat
import torch

# from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ptflops import get_model_complexity_info

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model.to(device)
cfg = get_default_cfg()
model = SeqNetDa(cfg).to(device)
input = torch.zeros((1, 3, 900, 1500)).to(device)
flops, prams = profile(model, inputs=(input,))
print("GFLOPS:", flops / 1000000000.0)
print("参数:", prams)
# stat(model, (3, 900, 1500))  # CPU统计
# summary(model,input_size=(3,900,1500),batch_size=1)  # GPU统计

print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))


macs, params = get_model_complexity_info(
    model, (3, 900, 1500), as_strings=True, print_per_layer_stat=True, verbose=True
)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))
