import torch 
import torch.nn.functional as F

class DCS_multi_lay_resolution(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 128,
        output_channels: int = 10,
        num_blocks: int = 3,
        dilations: list = [3, 6, 9],
        kernel_size:int = 3,
        mode = "concat"
        
    ):
        """Initilize Dilated Convolutions Stack for multi layer resolution block module.
        Args:
            input_channels: int = The number of input channels for the DCS block
            output_channels: int = The number of output channels of the individual dcs block
            num_blocks:int = Number of Conv dilations blocks in the DCS 
            dilations:list = List of dilation value for each of the block
            kernel_size:int = Size of the kernel in the convolutional blocks
            mode:str = concat or add up 
        """
        super(DCS_multi_lay_resolution, self).__init__()
        self.conv_list = torch.nn.ModuleList()
        self.num_blocks = num_blocks
        self.mode = mode
        assert num_blocks == len(dilations)
        for i in range(0, num_blocks):
            if i == 0:
                self.conv_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(input_channels, output_channels, kernel_size, dilation=(dilations[i]), padding = 'same'),
                    torch.nn.LeakyReLU()
                ))
            else:
                self.conv_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(input_channels + i*output_channels, output_channels, kernel_size, dilation=(dilations[i]), padding = 'same'),
                    torch.nn.LeakyReLU(),
                ))
                
    def forward(self, xs: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, In_chn, Fbin, Tdim).
        Returns:
            Tensor: Batch of predicted (B, In_chn + num_blocks*output_channels, Fbin, Tdim).
        """
        pred = []
        pred.append(xs)
        for i in range(0, self.num_blocks):
            prev_inputs_list = pred
            model = self.conv_list[i] #.cuda()
            out = model(torch.concat(prev_inputs_list, axis = 1))
            pred.append(out)
        
        if self.mode == "concat":
            dcs_output = torch.concat(pred, axis = 1)
        return dcs_output
