"""
Model I/O Utilities (PyTorch)
=============================

Save/load helpers for DQN agents and detectors.
"""

import os
import pickle
from typing import Any, Optional, Type
import warnings
import torch


class ModelIO:
    """Model save/load utilities."""

    @staticmethod
    def agent_model_exists(path: str) -> bool:
        """Check whether an agent model file exists (supports both single-stage and two-stage).

        Single-stage agent: file stored at {base}.pt.
        Two-stage agent: files stored at {base}_stage1.pt + {base}_stage2.pt.

        Args:
            path: Model path (the caller's base path)

        Returns:
            True if a single-stage .pt file or a two-stage _stage1.pt file is found.
        """
        pt_path = path.replace('.h5', '.pt')
        # Single-stage: check the .pt file directly
        if os.path.exists(pt_path):
            return True
        # Two-stage: check for _stage1.pt
        base = pt_path.replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        return os.path.exists(stage1_path)

    @staticmethod
    def is_two_stage_model(path: str) -> bool:
        """Determine whether the model at the given path is two-stage (inferred from file presence).

        Returns:
            True if _stage1.pt exists and the base .pt does not.
        """
        pt_path = path.replace('.h5', '.pt')
        if os.path.exists(pt_path):
            return False  # base .pt present -> single-stage
        base = pt_path.replace('.pt', '')
        return os.path.exists(base + '_stage1.pt')

    @staticmethod
    def save_agent(agent: Any, path: str) -> None:
        """
        Save a DQN agent (.pt format).

        Args:
            agent: DQN agent instance
            path: Destination path
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        agent.save(path)

    @staticmethod
    def load_agent(agent_class: Type, path: str, **kwargs) -> Any:
        """
        Load a DQN agent.

        Automatically extracts state_size / action_size and other initialization
        arguments from the checkpoint. External kwargs take precedence.

        Two-stage models are supported: when the base .pt is missing but
        _stage1.pt exists, metadata is read from _stage1.pt.

        Args:
            agent_class: Agent class
            path: Model path
            **kwargs: Agent initialization arguments (overrides values saved in the checkpoint)
        """
        pt_path = path.replace('.h5', '.pt')
        base = pt_path.replace('.pt', '')

        # Determine which checkpoint to read for initialization parameters
        if os.path.exists(pt_path):
            ckpt_path = pt_path
        elif os.path.exists(base + '_stage1.pt'):
            ckpt_path = base + '_stage1.pt'
        else:
            raise FileNotFoundError(
                f"Agent model file not found: {pt_path} or {base}_stage1.pt")

        # Read the checkpoint to obtain construction arguments
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        init_kwargs = {}
        for key in ('state_size', 'action_size'):
            if key in checkpoint:
                init_kwargs[key] = checkpoint[key]
        if 'state_size' not in init_kwargs:
            # Older checkpoints store only the weights; the input width of the first
            # linear layer is the state size the agent was trained with.
            state = checkpoint.get('model_state', checkpoint) if isinstance(checkpoint, dict) else {}
            for name, tensor in state.items():
                if name.endswith('weight') and getattr(tensor, 'dim', lambda: 0)() == 2:
                    init_kwargs['state_size'] = int(tensor.shape[1])
                    break
        init_kwargs.update(kwargs)

        agent = agent_class(**init_kwargs)
        agent.load(pt_path)
        print(f"Agent loaded: {pt_path}")
        return agent

    @staticmethod
    def save_detector(detector: Any, path: str) -> None:
        """
        Save an error detector.

        Args:
            detector: Detector instance
            path: Destination path (.pkl)
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(detector, f)
        print(f"Detector saved: {path}")

    @staticmethod
    def load_detector(path: str) -> Any:
        """
        Load an error detector.

        Args:
            path: Detector file path (.pkl)

        Returns:
            Detector instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detector file not found: {path}")

        with open(path, 'rb') as f:
            detector = pickle.load(f)
        print(f"Detector loaded: {path}")
        return detector

    @staticmethod
    def save_config(config: Any, path: str) -> None:
        """
        Save a configuration object.

        Args:
            config: Configuration object
            path: Destination path (.json)
        """
        import json
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved: {path}")

    @staticmethod
    def load_config(config_class: Type, path: str) -> Any:
        """
        Load a configuration object.

        Args:
            config_class: Configuration class
            path: Configuration file path (.json)

        Returns:
            Configuration object
        """
        import json

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        if hasattr(config_class, 'from_dict'):
            return config_class.from_dict(config_dict)
        return config_class(**config_dict)

    @staticmethod
    def exists(path: str) -> bool:
        """Check whether a file exists (supports two-stage models)."""
        return ModelIO.agent_model_exists(path)

    @staticmethod
    def ensure_dir(path: str) -> str:
        """Ensure the containing directory exists, and return it."""
        dir_path = os.path.dirname(path) or '.'
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
