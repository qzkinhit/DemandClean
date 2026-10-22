"""Train-only inference statistics for the unchanged author DAGPipe operators.

During gradient training each wrapped operator calls the author's forward.
After the topology and routing are fixed, operators are calibrated layer by
layer on dirty training features.  Validation and test never fit statistics.
"""
from __future__ import annotations

import torch
from torch import nn


class TrainingStatisticsOperation(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.kind = type(original).__name__
        self.frozen = False
        self.statistics = {}

    def fit_training_statistics(self, x):
        x = x.detach()
        state = {}
        eps = float(getattr(self.original, "eps", 1e-8))
        missing = ((x + 99999).abs() < 1e-3)
        filled = torch.where(missing, torch.zeros_like(x), x)
        if self.kind == "MeanImputer":
            state["mean"] = filled.sum(0, keepdim=True) / (~missing).sum(0, keepdim=True).clamp(min=1)
        elif self.kind == "MedianImputer":
            sorted_x = torch.sort(filled, dim=0).values
            positions = torch.arange(len(x), dtype=x.dtype, device=x.device)[:, None]
            weights = torch.softmax(-abs(positions - (len(x) - 1) / 2) / self.original.temperature, dim=0)
            state["median"] = (weights * sorted_x).sum(0, keepdim=True)
        elif self.kind == "KNNImputer":
            state["reference"] = filled
            state["std"] = filled.std(0, keepdim=True) + 1e-8
        elif self.kind == "StandardScaler":
            state["mean"] = x.mean(0, keepdim=True)
            state["std"] = x.std(0, keepdim=True) + eps
        elif self.kind == "RobustScaler":
            state["median"] = x.median(0, keepdim=True).values
            state["iqr"] = torch.quantile(x, .75, dim=0, keepdim=True) - torch.quantile(x, .25, dim=0, keepdim=True) + eps
        elif self.kind == "MinMaxScaler":
            state["minimum"] = x.min(0, keepdim=True).values
            state["range"] = x.max(0, keepdim=True).values - state["minimum"] + eps
        elif self.kind == "QuantileScaler":
            state["sorted"] = torch.sort(x, dim=0).values.contiguous()
        elif self.kind == "LogScaler":
            state["minimum"] = x.min(0, keepdim=True).values
        elif self.kind == "FrequencyEncoder":
            values, counts = torch.unique(x, sorted=True, return_counts=True)
            state["values"] = values
            state["frequencies"] = counts.to(x.dtype) / len(x)
        elif self.kind not in {"IdentityScaler", "TargetEncoder", "PolynomialTransform"}:
            raise NotImplementedError(f"No documented train-statistics inference adaptation for {self.kind}")
        self.statistics = {k: v.detach() for k, v in state.items()}
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return self.original(x)
        s = self.statistics
        k = self.kind
        eps = float(getattr(self.original, "eps", 1e-8))
        if k in {"MeanImputer", "MedianImputer", "KNNImputer"}:
            missing = ((x + 99999).abs() < 1e-3)
            filled = torch.where(missing, torch.zeros_like(x), x)
            if not bool(missing.any()):
                return filled
            if k == "MeanImputer":
                values = s["mean"]
            elif k == "MedianImputer":
                values = s["median"]
            else:
                # Only the dirty-training reference pool is allowed in KNN.
                values = filled.clone()
                rows = missing.any(1)
                distance = torch.cdist(filled[rows] / s["std"], s["reference"] / s["std"])
                weights = torch.softmax(-distance / self.original.temperature, dim=1)
                values[rows] = weights @ s["reference"]
            return torch.where(missing, values.expand_as(x), filled)
        if k == "StandardScaler":
            return (x - s["mean"]) / s["std"]
        if k == "RobustScaler":
            return (x - s["median"]) / s["iqr"]
        if k == "MinMaxScaler":
            return (x - s["minimum"]) / s["range"]
        if k == "QuantileScaler":
            n = len(s["sorted"])
            columns = []
            for column in range(x.shape[1]):
                rank = torch.searchsorted(s["sorted"][:, column].contiguous(), x[:, column].contiguous(), right=False)
                columns.append(rank.to(x.dtype).clamp(max=n-1) / (max(1, n-1) + eps))
            return torch.stack(columns, dim=1)
        if k == "LogScaler":
            # Domain clipping is fixed and does not fit on held-out values.
            return torch.log((x - s["minimum"] + 1 + eps).clamp(min=eps))
        if k == "FrequencyEncoder":
            flat = x.reshape(-1).contiguous()
            indices = torch.searchsorted(s["values"], flat)
            bounded = indices.clamp(max=len(s["values"])-1)
            values = torch.where((indices < len(s["values"])) & (s["values"][bounded] == flat), s["frequencies"][bounded], 0.)
            return values.view_as(x)
        return self.original(x)


@torch.no_grad()
def freeze_on_training_data(model, x):
    """Exact hierarchical forward order, fitting each op on training inputs."""
    model.eval()
    layer_adjacency = torch.sigmoid(model.layer_adj_logits)
    layer_outputs = []
    current = x
    for layer_index, layer_ops in enumerate(model.op_layers):
        routing = model.routers[layer_index]()
        intra = torch.sigmoid(model.intra_adj_logits[layer_index])
        outputs = []
        for operation_index, operation in enumerate(layer_ops):
            weighted = current * routing[:, operation_index][None, :]
            operation.fit_training_statistics(weighted)
            # No missing values makes KNN's result exactly its input, avoiding
            # the author's unnecessary quadratic distance allocation.
            output = operation(weighted)
            for previous_index in range(operation_index):
                output = output + intra[previous_index, operation_index] * outputs[previous_index]
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"Non-finite frozen representation at layer {layer_index}, op {operation_index}")
            outputs.append(output)
        layer_output = torch.stack(outputs, dim=0).mean(0)
        if layer_index:
            previous_outputs = torch.stack(layer_outputs, dim=0)
            previous_weights = layer_adjacency[:layer_index, layer_index].view(-1, 1, 1)
            layer_output = layer_output + (previous_outputs * previous_weights).sum(0)
        layer_outputs.append(layer_output)
        current = layer_output
    return current


def inference_statistics_manifest(model):
    return [
        {"layer": li, "operation": oi, "kind": op.kind, "frozen": op.frozen,
         "statistics_shapes": {key: list(value.shape) for key, value in op.statistics.items()}}
        for li, layer in enumerate(model.op_layers) for oi, op in enumerate(layer)
    ]
