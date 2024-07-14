import torch
import torch.nn as nn
from .typenet import TypeNet

class CustomTypeNet(nn.Module) :
  def __init__(self,sequence_length,in_dim,hidden_dim_1,hidden_dim_2,output_dim,dropout) :
    super(CustomTypeNet,self).__init__()
    self.tn1 = TypeNet(sequence_length,in_dim,hidden_dim_1,hidden_dim_2,output_dim,dropout)
    self.tn2 = TypeNet(sequence_length,in_dim,hidden_dim_1,hidden_dim_2,output_dim,dropout)

  def forward(self,x1,x2,length1,length2) :
    x1 = self.tn1(x1,length1)
    x2 = self.tn2(x2,length2)

    return x1,x2
