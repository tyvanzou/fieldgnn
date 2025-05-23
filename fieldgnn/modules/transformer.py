import torch.nn as nn



class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    # self-attention block
    def _sa_block_attn(
        self, x, attn_mask, key_padding_mask
    ):
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout1(x), attn

    # multihead attention block
    def _mha_block_attn(
        self,
        x,
        mem,
        attn_mask,
        key_padding_mask,
    ):
        x, attn = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout2(x), attn
