import torch 

class CoherenceNetLoss():
    # Loss function module for Coherence Net
    
    def __init__(self):
        self.criterion = torch.nn.L1Loss()
        
    def stream_loss(self, outputs, targets):
        """Calculate Loss for the streams and stream 1 integerator
        Args:
            outputs: Tensor = Batch of predictions (B, 1, F, T)
            targets: LongTensor = Batch of groundtruth (B, 1, F, T)
        Returns:
            Tensor: L1 loss value
        """
        loss = self.criterion(outputs, targets)
        return loss
    
    def lay2_loss(self,  targets, outputs_lay1, outputs_lay2, alpha = 0.2):
        """Calculate Loss for the stage 2 stream_integeration
        Args:
            outputs_lay1: Tensor = Batch of prediction lay1 (B, 1, F, T)
            outputs_lay2: Tensor = Batch of prediction lay2 (B, 1, F, T)
            target: LongTensor = Batch of groundtruth (B, 1, F, T)
        Returns:
            Tensor: L1 loss value
        """
        loss = self.criterion(targets, outputs_lay2) + alpha*self.criterion(outputs_lay1, outputs_lay2)
        return loss
    
    def tuning_loss(self, lay1_stream1, lay1_stream2, lay1_stream3, lay1_stream0, targets):
        loss = 0
        for item in (lay1_stream1, lay1_stream2, lay1_stream3, lay1_stream0):
            loss += self.criterion(item, targets)
        return loss
    
