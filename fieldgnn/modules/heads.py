# MOFTransformer version 2.0.0
import torch.nn as nn

# from transformers.models.bert.modeling_bert import (
#     BertConfig,
#     BertPredictionHeadTransform,
# )


class Pooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
