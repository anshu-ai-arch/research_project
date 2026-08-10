import torch
from typing import Dict, Type
from src.models.base_model import BaseECGModel


class ModelFactory:
    """
    Central registry factory for registering and instantiating deep learning model architectures.
    Allows easy model swapping by name string without changing training or evaluation code.
    """

    _registry: Dict[str, Type[BaseECGModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under a given name string."""
        def decorator(subclass: Type[BaseECGModel]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, model_name: str = None, **kwargs) -> BaseECGModel:
        """Instantiates a model registered with `model_name` using `kwargs` parameters."""
        target_name = model_name or kwargs.pop("name", None)
        if "name" in kwargs:
            kwargs.pop("name")

        if not target_name:
            raise ValueError("Model name must be provided to ModelFactory.create()")

        name_key = target_name.lower()
        if name_key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Model '{target_name}' is not registered in ModelFactory. Available models: {available}"
            )
        return cls._registry[name_key](**kwargs)

    @classmethod
    def list_available_models(cls) -> list:
        """Returns list of all registered model names."""
        return list(cls._registry.keys())
