# ------------------------------------------------------------------------------------------
# Implementation derived from: 
# ⚬ litgpt (https://github.com/Lightning-AI/litgpt), License: Apache License 2.0
# ------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import lit_gpt
from lit_gpt.config import Config as BaseConfig
from lit_gpt.model import GPT as BaseModel
from lit_gpt.model import Block as BaseBlock
from lit_gpt.model import CausalSelfAttention as BaseCausalSelfAttention
from lit_gpt.model import KVCache
from lit_gpt.utils import map_old_state_dict_weights


class ETHERLayer(nn.Module):
    def __init__(self, nb: int, 
                 Htype: str,
                 ether_dropout: float,
                 ):
        """Store ETHER specific attributes in a class.

        Args:
            nb: number of diagonal blocks
            ether_dropout: dropout that is applied on the input in the ETHER branch
        """
        super().__init__()
        assert nb >= 0
        self.nb = nb
        self.Htype = Htype
        # Optional dropout
        if ether_dropout > 0.0:
            self.ether_dropout = nn.Dropout(p=ether_dropout)
        else:
            self.ether_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class ETHERLinear(ETHERLayer):
    # ETHER implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for ETHER
        nb: int = 0,
        Htype: str = 'ether',
        ether_dropout: float = 0.0,
        flip_side: bool = False,
        **kwargs,
    ):
        """ETHER wrapper around linear class.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            nb: number of diagonal blocks
            Htype: type of transformation
            ether_dropout: dropout that is applied on the input in the ETHER branch
            flip_side: apply ETHER on the other (smaller) side to reduce computational overhead
        """
        super().__init__(nb=nb, Htype=Htype, ether_dropout=ether_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.Htype = Htype
        if 'HH' in self.Htype:
            self.is_HtransposeH = True
        else:
            self.is_HtransposeH = False
        self.flip_side = flip_side and not self.is_HtransposeH


        if nb>0:
            # get R
            self.nb = nb

            if self.flip_side:
                tmp_features = in_features
                in_features = out_features
                out_features = tmp_features
                
            if self.Htype == 'ether':
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * self.nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
            elif self.Htype == 'etherplus':
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
                ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                self.ether_R2 = nn.Parameter(ether_R2)
            elif self.Htype == 'oft':
                R_shape = [nb, in_features // nb, in_features // nb]
                ether_R = torch.zeros(R_shape[-1], R_shape[-1])
                ether_R = torch.stack([ether_R] * self.nb)
                self.ether_R = nn.Parameter(ether_R)
            # HH models
            elif self.Htype == 'etherplusHH':
                # front
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
                ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                self.ether_R2 = nn.Parameter(ether_R2)
                # back
                R34_shape = [nb, out_features // nb]
                ether_R3 = torch.rand(R34_shape[-1])
                ether_R3 = torch.stack([ether_R3] * nb)
                self.ether_R3 = nn.Parameter(ether_R3)
                nn.init.kaiming_uniform_(self.ether_R3, a=math.sqrt(5))
                ether_R4 = - torch.empty_like(ether_R3).copy_(ether_R3)
                self.ether_R4 = nn.Parameter(ether_R4)


    def reset_parameters(self):
        """Reset ETHER weights"""
        if hasattr(self, "ether_R"):
            nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
            if hasattr(self, "ether_R2"):
                self.ether_R2.data = - torch.empty_like(self.ether_R).copy_(self.ether_R)
        if hasattr(self, "ether_R3"):
            nn.init.kaiming_uniform_(self.ether_R3, a=math.sqrt(5))
            if hasattr(self, "ether_R4"):
                self.ether_R4.data = - torch.empty_like(self.ether_R3).copy_(self.ether_R3)


    def get_H(self):
        if self.Htype == 'ether':
            H = self.ether(self.ether_R)
        elif self.Htype == 'etherplus':
            H = self.etherplus(self.ether_R, self.ether_R2)
        elif self.Htype == 'oft':
            H = self.oft(self.ether_R)
        # or get HH
        elif self.Htype == 'etherplusHH':
            H = self.etherplus(self.ether_R, self.ether_R2)
            H2 = self.etherplus(self.ether_R3, self.ether_R4)

        if self.is_HtransposeH:
            return H, H2
        else:
            return H, None


    def forward(self, x: torch.Tensor):
        # if weights are merged or number of diagonal blocks is less or equal to zero (ETHER is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with the pretrained weights multiplied by the ETHER weights
        
        if self.nb == 0 or self.merged:
            pretrained = self.linear(x)
            return pretrained

        # - weights
        # get H
        H, H2 = self.get_H()
        
        # pretrained weights
        filt = self.linear.weight.data

        # - shapes
        nb,m,n = H.shape  #> [4,512,512]
        f,d = filt.shape  #> [8192,2048] or [2048,2048]

        # - direct transformation
        if not self.flip_side:
            # split in nb blocks
            filt = filt.reshape(nb, f, d//nb)

            # multiply
            filt = torch.einsum('rfm,rmn->rfn', filt, H)

            # rebuild in one block
            filt = filt.reshape(f, d)

        # - transposed transformation
        if self.flip_side or self.is_HtransposeH:
            # split in nb blocks
            filt = filt.reshape(nb, f//nb, d)

            # multiply
            if self.is_HtransposeH:
                filt = torch.einsum('rnm,rmd->rnd', H2, filt)
            else:
                filt = torch.einsum('rnm,rmd->rnd', H, filt)

            # rebuild in one block
            filt = filt.reshape(f, d)

        # - bias
        bias_term = self.linear.bias.data if self.linear.bias is not None else None

        # Apply the trainable identity matrix
        ether = nn.functional.linear(input=self.ether_dropout(x), weight=filt, bias=bias_term)
        return ether


    def merge(self):
        """Merges the ETHER weights to the pretrained weights (W = HW)."""
        if self.nb > 0 and not self.merged:
            #! copied exactly from forward()
            # - weights
            # get H
            H, H2 = self.get_H()
            
            # pretrained weights
            filt = self.linear.weight.data

            # - shapes
            nb,m,n = H.shape
            f,d = filt.shape

            # - direct transformation
            if not self.flip_side:
                # split in nb blocks
                filt = filt.reshape(nb, f, d//nb)

                # multiply
                filt = torch.einsum('rfm,rmn->rfn', filt, H)

                # rebuild in one block
                filt = filt.reshape(f, d)

            # - transposed transformation
            if self.flip_side or self.is_HtransposeH:
                # split in nb blocks
                filt = filt.reshape(nb, f//nb, d)

                # multiply
                if self.is_HtransposeH:
                    filt = torch.einsum('rnm,rmd->rnd', H2, filt)
                else:
                    filt = torch.einsum('rnm,rmd->rnd', H, filt)

                # rebuild in one block
                filt = filt.reshape(f, d)
            #! copied exactly from forward()
                
            # - merge
            self.linear.weight.data = filt
            self.merged = True
            

    def ether(self, R):
        nb, r = R.shape
        I = torch.eye(r, device=R.device, dtype=R.dtype).unsqueeze(0).expand(nb, r, r)
        R = R.unsqueeze(1)
        H = I - 2 * torch.bmm(R.transpose(1,2), R) / torch.bmm(R, R.transpose(1,2))
        return H

    def etherplus(self, R1, R2):
        nb, r = R1.shape
        I = torch.eye(r, device=R1.device).unsqueeze(0).expand(nb, r, r)
        R1 = R1.unsqueeze(1)
        R2 = R2.unsqueeze(1)
        H = I - torch.bmm(R1.transpose(1,2), R1) / torch.bmm(R1, R1.transpose(1,2)) +  torch.bmm(R2.transpose(1,2), R2) / torch.bmm(R2, R2.transpose(1,2))
        return H
    
    def oft(self, R):
        nb, r, c = R.shape
        skew = 0.5 * (R - R.transpose(1, 2))
        I = torch.eye(r, device=R.device).unsqueeze(0).expand(nb, r, c)
        H = torch.bmm(I + skew, torch.inverse(I - skew))
        return H
    

class ETHERQKVLinear(ETHERLinear):
    # ETHER implemented in QKV layers
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for ETHER
        n_head: int,
        n_query_groups: int,
        nb: int = 0,
        Htype: str = 'ether',
        ether_dropout: float = 0.0,
        enable_ether: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs,
    ):
        super(ETHERLinear, self).__init__(nb=nb, Htype=Htype, ether_dropout=ether_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_ether, bool):
            enable_ether = [enable_ether] * 3
        assert len(enable_ether) == 3
        self.enable_ether = enable_ether
        self.Htype = Htype
        if 'HH' in self.Htype:
            self.is_HtransposeH = True
        else:
            self.is_HtransposeH = False
        

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ nb: 2
        # ⚬ enable_ether: [True, False, True]

        if nb > 0 and any(enable_ether):
            # - q,k,v
            self.enable_q, self.enable_k, self.enable_v = enable_ether

            # consider separately for each head
            head_embd = in_features // n_head  #> if MQA/GQA, q_head_embd != kv_head_embd
            self.head_embd = head_embd
            self.q_per_kv = n_head // n_query_groups
            #total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            pretrained_qkv_emb = (
                in_features,
                in_features // self.q_per_kv,
                in_features // self.q_per_kv,
            )
            pretrained_qkv_nheads = (
                pretrained_qkv_emb[0]//head_embd, 
                pretrained_qkv_emb[1]//head_embd, 
                pretrained_qkv_emb[2]//head_embd
                )
            ether_qkv_embd = (
                pretrained_qkv_emb[0] * self.enable_q,
                pretrained_qkv_emb[1] * self.enable_k,
                pretrained_qkv_emb[2] * self.enable_v,
            )
            ether_qkv_nheads = (
                pretrained_qkv_nheads[0] * self.enable_q,
                pretrained_qkv_nheads[1] * self.enable_k,
                pretrained_qkv_nheads[2] * self.enable_v,
            )
            effective_ether_qkv_nheads = [s for s in ether_qkv_nheads if s > 0]
            self.same_ether_qkv_nheads = len(set(effective_ether_qkv_nheads)) == 1

            # indices for filt
            total_qkv = self.q_per_kv + 2
            ind = range(out_features)
            self.q_ind = [x for x in ind if (x // self.head_embd) % total_qkv < total_qkv - 2] if self.enable_q else []
            self.k_ind = [x for x in ind if (x // self.head_embd) % total_qkv == total_qkv - 2] if self.enable_k else []
            self.v_ind = [x for x in ind if (x // self.head_embd) % total_qkv == total_qkv - 1] if self.enable_v else []

            # - get R
            for idx_qkv, embd_qkv in enumerate(ether_qkv_embd):
                # init
                ether_R, ether_R2, ether_R3, ether_R4 = None, None, None, None

                # only if enabled
                if embd_qkv > 0:
                    # associate R
                    if self.Htype == 'ether':
                        R_shape = [1, nb, embd_qkv // nb]
                        ether_R = torch.rand(R_shape[-1])
                        ether_R = torch.stack([ether_R] * self.nb)
                        ether_R = torch.stack([ether_R] * R_shape[0]) #> qkv
                        nn.init.kaiming_uniform_(ether_R, a=math.sqrt(5))
                    elif self.Htype == 'etherplus':
                        R_shape = [1, nb, embd_qkv // nb]
                        ether_R = torch.rand(R_shape[-1])
                        ether_R = torch.stack([ether_R] * self.nb)
                        ether_R = torch.stack([ether_R] * R_shape[0]) #> qkv
                        nn.init.kaiming_uniform_(ether_R, a=math.sqrt(5))
                        ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                    elif self.Htype == 'oft':
                        R_shape = [1, nb, embd_qkv // nb, embd_qkv // nb]
                        ether_R = torch.zeros(R_shape[-1], R_shape[-1])
                        ether_R = torch.stack([ether_R] * self.nb)
                        ether_R = torch.stack([ether_R] * R_shape[0]) #> qkv
                    # HH models
                    elif self.Htype in ['etherplusHH']:
                        # front
                        R_shape = [1, nb, embd_qkv // nb]
                        ether_R = torch.rand(R_shape[-1])
                        ether_R = torch.stack([ether_R] * self.nb)
                        ether_R = torch.stack([ether_R] * R_shape[0])
                        nn.init.kaiming_uniform_(ether_R, a=math.sqrt(5))
                        ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                        # back
                        R34_shape = [1, nb, embd_qkv // nb]
                        ether_R3 = torch.rand(R34_shape[-1])
                        ether_R3 = torch.stack([ether_R3] * self.nb)
                        ether_R3 = torch.stack([ether_R3] * R34_shape[0])
                        nn.init.kaiming_uniform_(ether_R3, a=math.sqrt(5))
                        ether_R4 = - torch.empty_like(ether_R3).copy_(ether_R3)

                # store parameters indicating if they belong to q,k,v
                if idx_qkv == 0:
                    self.ether_Rq  = nn.Parameter(ether_R)  if ether_R  is not None else None
                    self.ether_R2q = nn.Parameter(ether_R2) if ether_R2 is not None else None
                    self.ether_R3q = nn.Parameter(ether_R3) if ether_R3 is not None else None
                    self.ether_R4q = nn.Parameter(ether_R4) if ether_R4 is not None else None
                elif idx_qkv == 1:
                    self.ether_Rk  = nn.Parameter(ether_R)  if ether_R  is not None else None
                    self.ether_R2k = nn.Parameter(ether_R2) if ether_R2 is not None else None
                    self.ether_R3k = nn.Parameter(ether_R3) if ether_R3 is not None else None
                    self.ether_R4k = nn.Parameter(ether_R4) if ether_R4 is not None else None
                elif idx_qkv == 2:
                    self.ether_Rv  = nn.Parameter(ether_R)  if ether_R  is not None else None
                    self.ether_R2v = nn.Parameter(ether_R2) if ether_R2 is not None else None
                    self.ether_R3v = nn.Parameter(ether_R3) if ether_R3 is not None else None
                    self.ether_R4v = nn.Parameter(ether_R4) if ether_R4 is not None else None


    def get_H_qkv(self, ether_R, ether_R2=None, ether_R3=None, ether_R4=None):
        if ether_R is None:
            return None, None

        if self.Htype == 'ether':
            H = self.ether_qkv(ether_R)
        elif self.Htype == 'etherplus':
            H = self.etherplus_qkv(ether_R, ether_R2)
        elif self.Htype == 'oft':
            H = self.oft_qkv(ether_R)
        # or get HH
        elif self.Htype == 'etherplusHH':
            H = self.etherplus_qkv(ether_R, ether_R2)
            H2 = self.etherplus_qkv(ether_R3, ether_R4)

        if self.is_HtransposeH:
            return H, H2
        else:
            return H, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        if self.nb == 0 or not any(self.enable_ether) or self.merged:
            pretrained = self.linear(x)
            return pretrained
        
        if not self.same_ether_qkv_nheads:
            raise NotImplementedError("Not implemented yet") #TODO:
        
        # - Hs
        # concatenate Rq, Rk, Rv
        R_list  = [R  for R  in [self.ether_Rq,  self.ether_Rk,  self.ether_Rv]  if R  is not None]
        R2_list = [R2 for R2 in [self.ether_R2q, self.ether_R2k, self.ether_R2v] if R2 is not None]
        R3_list = [R3 for R3 in [self.ether_R3q, self.ether_R3k, self.ether_R3v] if R3 is not None]
        R4_list = [R4 for R4 in [self.ether_R4q, self.ether_R4k, self.ether_R4v] if R4 is not None]

        # get H
        H, H2 = self.get_H_qkv(
            torch.cat(R_list,  dim=0),
            torch.cat(R2_list, dim=0) if R2_list else None,
            torch.cat(R3_list, dim=0) if R3_list else None,
            torch.cat(R4_list, dim=0) if R4_list else None,
        )

        # - weights
        new_filt = self.linear.weight.data.clone()
        filt = self.linear.weight.data[self.q_ind + self.k_ind + self.v_ind]

        # - shapes
        qkv,nb,_,_ = H.shape
        f,d = filt.shape

        # - transposed transformation
        # split in qkv blocks
        filt = filt.reshape(qkv, f//qkv, d)

        # split in nb blocks
        filt = filt.reshape(qkv, nb, f//qkv//nb, d) 

        # multiply
        filt = torch.einsum('qrmn,qrnd->qrmd', H, filt)

        # rebuild in one block
        filt = filt.reshape(qkv, f//qkv, d)

        # - direct transformation
        if self.is_HtransposeH:
            # split in nb blocks
            filt = filt.reshape(qkv, nb, f//qkv, d//nb)

            # multiply
            filt = torch.einsum('qrfm,qrmn->qrfn', filt, H2)

            # rebuild in one block
            filt = filt.reshape(qkv, f//qkv, d)

        # - map filt to weights
        # rebuild in one block
        filt = filt.reshape(f, d)

        new_filt[self.q_ind + self.k_ind + self.v_ind] = filt

        # - bias
        bias_term = self.linear.bias.data if self.linear.bias is not None else None

        # Apply the trainable identity matrix
        ether = nn.functional.linear(input=self.ether_dropout(x), weight=new_filt, bias=bias_term)
        return ether
    

    def merge(self):
        if self.nb > 0 and any(self.enable_ether) and not self.merged:
            #! copied exactly from forward()
            # - Hs
            # concatenate Rq, Rk, Rv
            R_list  = [R  for R  in [self.ether_Rq,  self.ether_Rk,  self.ether_Rv]  if R  is not None]
            R2_list = [R2 for R2 in [self.ether_R2q, self.ether_R2k, self.ether_R2v] if R2 is not None]
            R3_list = [R3 for R3 in [self.ether_R3q, self.ether_R3k, self.ether_R3v] if R3 is not None]
            R4_list = [R4 for R4 in [self.ether_R4q, self.ether_R4k, self.ether_R4v] if R4 is not None]

            # get H
            H, H2 = self.get_H_qkv(
                torch.cat(R_list,  dim=0),
                torch.cat(R2_list, dim=0) if R2_list else None,
                torch.cat(R3_list, dim=0) if R3_list else None,
                torch.cat(R4_list, dim=0) if R4_list else None,
            )

            # - weights
            new_filt = self.linear.weight.data.clone()
            filt = self.linear.weight.data[self.q_ind + self.k_ind + self.v_ind]

            # - shapes
            qkv,nb,_,_ = H.shape
            f,d = filt.shape

            # - transposed transformation
            # split in qkv blocks
            filt = filt.reshape(qkv, f//qkv, d)

            # split in nb blocks
            filt = filt.reshape(qkv, nb, f//qkv//nb, d) 

            # multiply
            filt = torch.einsum('qrmn,qrnd->qrmd', H, filt)

            # rebuild in one block
            filt = filt.reshape(qkv, f//qkv, d)

            # - direct transformation
            if self.is_HtransposeH:
                # split in nb blocks
                filt = filt.reshape(qkv, nb, f//qkv, d//nb)

                # multiply
                filt = torch.einsum('qrfm,qrmn->qrfn', filt, H2)

                # rebuild in one block
                filt = filt.reshape(qkv, f//qkv, d)

            # - map filt to weights
            # rebuild in one block
            filt = filt.reshape(f, d)

            new_filt[self.q_ind + self.k_ind + self.v_ind] = filt
            #! copied exactly from forward()

            # - merge
            self.linear.weight.data = new_filt
            self.merged = True
    

    def ether_qkv(self, R):
        qkv, nb, r = R.shape
        I = torch.eye(r, device=R.device, dtype=R.dtype).unsqueeze(0).expand(qkv, nb, r, r)
        RR = torch.einsum('ijk, ijl -> ijkl', R, R)
        rr = torch.einsum('ijk, ijk -> ij', R, R)
        RRrr = torch.einsum('ijkl, ij -> ijkl', RR, 1/rr)
        H = I - 2 * RRrr
        return H
    
    def etherplus_qkv(self, R1, R2):
        qkv, nb, r = R1.shape
        I = torch.eye(r, device=R1.device, dtype=R1.dtype).unsqueeze(0).expand(qkv, nb, r, r)
        RR1 = torch.einsum('ijk, ijl -> ijkl', R1, R1)
        rr1 = torch.einsum('ijk, ijk -> ij', R1, R1)
        RRrr1 = torch.einsum('ijkl, ij -> ijkl', RR1, 1/rr1)
        RR2 = torch.einsum('ijk, ijl -> ijkl', R2, R2)
        rr2 = torch.einsum('ijk, ijk -> ij', R2, R2)
        RRrr2 = torch.einsum('ijkl, ij -> ijkl', RR2, 1/rr2)
        H = I - RRrr1 + RRrr2
        return H
    
    def oft_qkv(self, R):
        qkv, nb, r, c = R.shape
        skew = 0.5 * (R - R.transpose(-2, -1))
        I = torch.eye(r, device=R.device, dtype=R.dtype).unsqueeze(0).expand(qkv, nb, r, c)
        Iskew = I - skew
        invIskew = torch.inverse(Iskew.float()).bfloat16()
        H = torch.einsum('ijkl, ijlm -> ijkm', I + skew, invIskew)
        return H


def mark_only_ether_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except ETHER's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with ETHER layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"ether_only"``: only bias weight for ETHER layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "ether_only", "all"]
    """
    # freeze all layers except ETHER's
    for n, p in model.named_parameters():
        if "ether_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "ether_only":
        for m in model.modules():
            if isinstance(m, ETHERLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def ether_filter(key: str, value: Any) -> bool:
    return "ether_" in key


@dataclass
class Config(BaseConfig):
    """
    Args:
        nb: number of diagonal blocks
        dropout: dropout that is applied on the input in the ETHER branch
        to_*: either apply ETHER to the specified weights or not
    """
    nb: int = 0
    Htype: str = 'ether'
    dropout: float = 0.0
    flip_side: bool = False
    to_query: bool = False
    to_key: bool = False
    to_value: bool = False
    to_projection: bool = False
    to_mlp: bool = False
    to_head: bool = False

    @property
    def mlp_class(self) -> Type:
        return getattr(lit_gpt.ether, self._mlp_class)


class GPT(BaseModel):  #> wrapping GPT model with ETHER
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = ETHERLinear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            nb=(config.nb if config.to_head else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, lm_head_chunk_size: int = 0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)  # (B, T, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, ETHERLinear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping ={"lm_head.weight": "lm_head.linear.weight", "lm_head.bias": "lm_head.linear.bias"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = ETHERQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            nb=config.nb,
            Htype=config.Htype,
            ether_dropout=config.dropout,
            enable_ether=(config.to_query, config.to_key, config.to_value),
            bias=config.bias,
            # for MQA/GQA support
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.proj = ETHERLinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            nb=(config.nb if config.to_projection else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(lit_gpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = ETHERLinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            nb=(config.nb if config.to_mlp else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
        )
        self.proj = ETHERLinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            nb=(config.nb if config.to_mlp else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
            flip_side=config.flip_side,
        )

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(lit_gpt.model.LLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = ETHERLinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            nb=(config.nb if config.to_mlp else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
        )
        self.fc_2 = ETHERLinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            nb=(config.nb if config.to_mlp else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
        )
        self.proj = ETHERLinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            nb=(config.nb if config.to_mlp else 0),
            Htype=config.Htype,
            ether_dropout=config.dropout,
            flip_side=config.flip_side,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1) * x_fc_2
        return self.proj(x)


class LLaMAMoE(lit_gpt.model.LLaMAMoE):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.gate = ETHERLinear(
            config.n_embd,
            config.n_expert,
            bias=False,
            nb=(config.nb if config.to_mlp else 0),
            ether_dropout=config.dropout,
        )
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"gate.weight": "gate.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_ether_weights(model: GPT) -> None:
    """Merge ETHER weights into the pretrained weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, ETHERLinear):
            module.merge()
