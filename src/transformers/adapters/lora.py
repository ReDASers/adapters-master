import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import PretrainedConfig
from .composition import AdapterCompositionBlock
from .configuration import LoRAConfig
from .layer import AdapterLayerBase
from .modeling import Activation_Function_Class


class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
    ):
        super().__init__()
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.is_dora = config.is_dora
        
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # Initialize scaling based on config type
        if config.scaling is None:  # Changed from config["scaling"]
            self.scaling = torch.tensor(1.0)
        elif isinstance(config.scaling, float):  # Changed from config["scaling"]
            self.scaling = torch.tensor(max(config.scaling, 1.0))  # Ensure scaling is positive
        elif config.scaling == "learnable":  # Changed from config["scaling"]
            self.scaling = nn.Parameter(torch.ones(1))
            nn.init.ones_(self.scaling.data)
        else:
            raise ValueError(f"Unknown scaling type: {config.scaling}")  # Changed from config["scaling"]
        
        # Validate composition mode and r
        if self.r > 1 and self.composition_mode == "scale":
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        
        # Initialize trainable parameters
        if self.r > 0:
            std_dev = 1 / torch.sqrt(torch.tensor(self.r).float())
       
            if self.lora_alpha is None or self.lora_alpha == 0:
                self.lora_alpha = 1.0
            self.f = nn.Sequential(
                    nn.Linear(lora_A_shape[1], self.r),
                    Activation_Function_Class(config.non_linearity.lower()),
                    nn.Linear(self.r, self.r),
                    Activation_Function_Class(config.non_linearity.lower()),
                    nn.Linear(self.r, lora_A_shape[1]),
                )
            if self.composition_mode == "add":
                self.lora_A = nn.Parameter(torch.randn(lora_A_shape) * std_dev)
                print("A ",self.lora_A.shape)
            self.lora_alpha = self.lora_alpha/math.sqrt(self.r) 
            self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
            print("B ", self.lora_B.shape)
            self.lora_C = nn.Parameter(torch.ones((lora_B_shape[0], 1)))
            print("C ", self.lora_C.shape)
            #nn.init.normal_(self.lora_C, mean=1.0, std=math.sqrt(2.0 / self.lora_C.shape[0]))
            nn.init.zeros_(self.lora_B)
            nn.init.ones_(self.lora_C)
            if self.use_gating:
                self.gate = nn.Linear(lora_A_shape[1], gating_heads)
                self.gate.weight.data.normal_(mean=1.0, std=0.02)
                #nn.init.ones_(self.gate.weight)
            
            self.m = nn.Parameter(torch.ones(1, lora_B_shape[0])) 
            nn.init.ones_(self.m)
            #nn.init.normal_(self.m, mean=1.0, std=0.02)
                

            # Initialize weights
            if config.init_weights == "lora":
                if self.composition_mode == "add":
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                nn.init.ones_(self.lora_C)
            elif config.init_weights == "bert":
                if self.composition_mode == "add":
                    nn.init.normal_(self.lora_A, std=0.02)
                nn.init.normal_(self.lora_B, std=0.02)
                nn.init.normal_(self.lora_C, mean=1.0, std=0.02)
            elif config.init_weights == "ia3":
                if self.composition_mode == "add":
                    nn.init.normal_(self.lora_A, mean=1.0, std=0.02)
                nn.init.ones_(self.lora_B)
                nn.init.uniform_(self.lora_C, a=0.95, b=1.05)
            elif config.init_weights == "xavier":
                if self.composition_mode == "add":
                    nn.init.xavier_uniform_(self.lora_A)
                nn.init.zeros_(self.lora_B)
                nn.init.ones_(self.lora_C)
            elif config.init_weights == "prexia":
                nn.init.zeros_(self.lora_B)
                nn.init.ones_(self.lora_C)
            else:
                raise ValueError(f"Unknown init_weights type: {config.init_weights}")

    def com(self, weights: torch.Tensor, added: torch.Tensor, gating=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if gating is None:
            gating = 1.0
        
        if self.composition_mode == "add":
            return weights + added * gating * self.scaling
        elif self.composition_mode == "scale":
            return weights * (added * gating * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        if self.composition_mode == "add":
            return weights - added * self.scaling
        elif self.composition_mode == "scale":
            return weights / (added * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")


class LoRALayer(AdapterLayerBase):
    def __init__(self, location_key: str, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.config = config
        self.loras = nn.ModuleDict(dict())
        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora = LoRA(
                *self._get_lora_shapes(lora_config),
                lora_config,
                gating_heads=self.get_n_heads(lora_config),
            )
            lora.train(self.training)
            self.loras[adapter_name] = lora
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> nn.Module:
        return self.loras.get(adapter_name, None)


class Linear(LoRALayer, nn.Linear):
    """
    LoRA implementation for Linear layer.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)
        self.n_passes = 0
        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def reset_adapter(self):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w
        
        if self.merged:
            raise NotImplementedError()
            lora = self.loras[self.merged]
            if lora.r > 0:
                if lora.composition_mode == "scale":
                    delta_w = T(lora.lora_B)
                    if lora.is_dora:
                        delta_w = delta_w * lora.lora_alpha
                        norm_w = delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9
                        unit_w = delta_w / norm_w
                        delta_w = unit_w * lora.m * (lora.scaling + 1e-9)
                else:
                    delta_w = T(lora.lora_B @ lora.lora_A)
                    if lora.is_dora:
                        delta_w = lora.lora_alpha * delta_w
                        mult = T(lora.lora_C)
                        X_plus_AB = self.weight.data + delta_w
                        X_plus_AB_times_C = X_plus_AB * mult
                        result = X_plus_AB_times_C * lora.scaling
                        return result
                self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def _compute_adapted_weight(self, lora, scaling=None):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w
        raise NotImplementedError()
        weight = self.weight
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = T(lora.lora_B)
                if lora.is_dora:
                    delta_w = delta_w * lora.lora_alpha
                    norm_w = delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9
                    unit_w = delta_w / norm_w
                    delta_w = unit_w * lora.m * (lora.scaling + 1e-9)
            else:
                delta_w = T(lora.lora_B @ lora.lora_A)
                if lora.is_dora:
                    delta_w = lora.lora_alpha * delta_w
                    mult = T(lora.lora_C)
                    X_plus_AB = weight + delta_w
                    X_plus_AB_times_C = X_plus_AB * mult
                    result = X_plus_AB_times_C * lora.scaling
                    return result
            weight = lora.com(weight, delta_w, gating=scaling)

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.transpose(w, -2, -1) if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    lora = self.loras[adapter_setup[0]]
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    if lora.r > 0:
                        if lora.use_gating:
                            gate = torch.tanh(lora.gate(x))
                            gate = torch.mean(gate, dim=1).unsqueeze(-1)
                            gate = gate + 1.0
                            self._store_gating_score(adapter_setup[0], gate)
                        else:
                            gate = 1.0
                        if lora.composition_mode == "scale":
                            if lora.is_dora:
                                delta_w = lora.lora_alpha * T(lora.lora_B)
                                norm_w = delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9
                                unit_w = delta_w / norm_w
                                delta_w = unit_w * lora.m * (lora.scaling + 1e-9)
                            else:
                                delta_w = T(lora.lora_B)
                            delta_w = delta_w.view(1, 1, -1)
                        else:
                            #fx = lora.lora_alpha * lora.scaling * lora.f(lora.lora_dropout(x))
                            mult = lora.lora_C.view(1, 1, -1)
                            fx = lora.f(lora.lora_dropout(x))
                            #print(x.shape, fx.shape, lora.lora_A.shape, lora.lora_B.shape, mult.shape)
                            delta_w = lora.scaling * lora.lora_alpha * (fx @ torch.t(lora.lora_A) @ torch.t(lora.lora_B))
                            dora = delta_w/ (delta_w.norm(p=2, dim=1, keepdim=True) + 1e-9)
                            
                            if lora.is_dora:
                                # result = result * mult
                                if lora.lora_A.shape[1] == lora.lora_B.shape[0]:
                                    result = result + dora
                                else:
                                    result = result * mult
                                #result = result * gate
                                return result*gate
                            else:
                                #xA = lora.lora_dropout(x) @ torch.t(lora.lora_A)
                                
                                #xAB = xA @ torch.t(lora.lora_B)
                                #fxAB = lora.f(lora.lora_alpha * lora.m * xAB)
                                if lora.lora_A.shape[1] == lora.lora_B.shape[0]:
                                    result = (result * mult + dora * lora.m)*lora.scaling
                                else:
                                    result = result * mult
                                #result = result * lora.scaling * gate
                                return result*gate
                        result = lora.com(result, delta_w, gating=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")
        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(LoRALayer, nn.Linear):
    """
    LoRA implementation for merged attention layer.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return len(set(lora.attn_matrices))

    def _get_lora_shapes(self, config: LoRAConfig):
        n_heads = self.get_n_heads(config)
        return (config.r * n_heads, self.in_features), (
            self.out_features // 3 * n_heads,
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx)
        if is_added:
            lora_config = self.config.adapters.match(
                adapter_name,
                config_type=LoRAConfig,
                layer_idx=self.layer_idx,
                location_key=self.location_key,
            )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                lora.lora_ind = self.weight.new_zeros((self.out_features,), dtype=torch.bool).view(
                    len(lora.enable_lora), -1
                )
                lora.lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora.lora_ind.view(-1)

    def pad(self, x, lora, fill_value=None):
        if fill_value is None:
            if lora.composition_mode == "add":
                fill_value = 0
            else:
                fill_value = 1
        result = x.new_full((*x.shape[:-1], self.out_features), fill_value)
        result = result.view(-1, self.out_features)
        result[:, lora.lora_ind] = x.reshape(-1, self.out_features // 3 * self.get_n_heads(lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_adapter(self):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.composition_mode == "scale":
                    delta_w = lora.lora_B
                    if lora.is_dora:
                        delta_w = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                else:
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
         
                    if lora.is_dora:
                        direction = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                        delta_w = direction * lora.m
                delta_w = delta_w.transpose(-2, -1)
                self.weight.data = lora.com_inv(self.weight.data, T(self.pad(delta_w, lora)))
            self.merged = None

    def _compute_adapted_weight(self, name, lora):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        weight = self.weight
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = lora.lora_B
                if lora.is_dora:
                    delta_w = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
            else:
                delta_w = F.conv1d(
                    lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                ).squeeze(0)
            if lora.is_dora:
                direction = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                delta_w = direction * lora.m

            # shape after transpose: <head_dim> x <head_dim * n_heads>
            delta_w = delta_w.transpose(-2, -1)

            weight = lora.com(weight, T(self.pad(delta_w, lora)))

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(name, lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        if not self.merged:
            raise NotImplementedError()
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.composition_mode == "scale":
                            delta_w = lora.lora_B
                            if lora.is_dora:
                                delta_w = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                            delta_w = delta_w.view(1, 1, -1)
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                            delta_w = F.conv1d(
                                after_A.transpose(-2, -1), lora.lora_B.unsqueeze(-1), groups=sum(lora.enable_lora)
                            ).transpose(-2, -1)
                            if lora.is_dora:
                                direction = delta_w / (delta_w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                                delta_w = direction * lora.m
                        if lora.use_gating:
                            gate = 1 + torch.tanh(lora.gate(x))
                            gate = torch.mean(gate, dim=1)
                            self._store_gating_score(adapter_setup[0], gate)
                            gate = self.pad(
                                gate.repeat_interleave(self.out_features // 3, dim=-1), lora, fill_value=1
                            ).unsqueeze(1)
                        else:
                            gate = None
                        # result = (batch_size, seq_len, head_dim * 3)
                        
                        result = lora.com(result, self.pad(delta_w, lora), gating=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
