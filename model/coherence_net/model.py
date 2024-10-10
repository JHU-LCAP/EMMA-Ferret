import torch 
from .layer import Layer

class CoherenceNet(torch.nn.Module):
    def __init__(
        self,
        config_lay1,
        config_lay2
    ):
        """Initilize Energy predictor module.
        Args:
        
        """
        super(CoherenceNet, self).__init__()
        self.config_lay1 = config_lay1
        self.config_lay2 = config_lay2
        
        self.lay1 = Layer(self.config_lay1)
        self.lay2 = Layer(self.config_lay2)
                
    def forward(self, xs: torch.Tensor, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (List of Tensors):List of Tensor outputs from different streams - Batch of input sequences (B, In_chn, Fbin, Tdim).
            x_indicator:  
            stage:
        Returns:
            Tensor: Batch of predicted (B, In_chn + num_blocks*output_channels, Fbin, Tdim).
        """
        predictions_lay1, embeddings_lay1, memory_lay1 = self.lay1(xs, x_indicator)
        
        pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
        embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1
        m_stream1_lay1, m_stream2_lay1, m_stream3_lay1, m_layer1comb_lay1 = memory_lay1
        
        
        x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
        
        predictions_lay2, embeddings_lay2, memory_lay2 = self.lay2(x_in_lay2, x_indicator)
        
        pred_stream1_lay2, pred_stream2_lay2, pred_stream3_lay2, pred_layer1comb_lay2 = predictions_lay2
        embedding_stream1_lay2, embedding_stream2_lay2, embedding_stream3_lay2, embedding_layer1comb_lay2 = embeddings_lay2
        m_stream1_lay2, m_stream2_lay2, m_stream3_lay2, m_layer1comb_lay2 = memory_lay2
        
        return {
            "layer1_stream_predictions": [pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1],
            "layer1_stream_integerator": pred_layer1comb_lay1,
            "layer1_embeddings": [embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1],
            "layer1_memory": [m_stream1_lay1, m_stream2_lay1, m_stream3_lay1, m_layer1comb_lay1],
            
            "layer2_stream_predictions": [pred_stream1_lay2, pred_stream2_lay2, pred_stream3_lay2],
            "layer2_stream_integerator": pred_layer1comb_lay2,
            "layer2_embeddings": [embedding_stream1_lay2, embedding_stream2_lay2, embedding_stream3_lay2],
            "layer2_memory" : [m_stream1_lay2, m_stream2_lay2, m_stream3_lay2, m_layer1comb_lay2] 
        }



