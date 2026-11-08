"""NeuroFusion AI - Enhancement Models Package"""

from backend.models.enhancement.base import BaseEnhancer
from backend.models.enhancement.clahe_enhancer import CLAHEEnhancer
from backend.models.enhancement.restormer import Restormer
from backend.models.enhancement.swinir import SwinIR
from backend.models.enhancement.mirnet import MIRNet

ENHANCEMENT_MODELS = {
    "clahe": CLAHEEnhancer,
    "restormer": Restormer,
    "swinir": SwinIR,
    "mirnet": MIRNet,
}


def get_enhancement_model(model_name: str, **kwargs) -> BaseEnhancer:
    """
    Factory function to get an enhancement model by name.
    
    Args:
        model_name: Name of the model ('clahe', 'restormer', 'swinir', 'mirnet')
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized enhancement model
    """
    model_name = model_name.lower()
    if model_name not in ENHANCEMENT_MODELS:
        raise ValueError(
            f"Unknown enhancement model: {model_name}. "
            f"Available: {list(ENHANCEMENT_MODELS.keys())}"
        )
    return ENHANCEMENT_MODELS[model_name](**kwargs)


__all__ = [
    "BaseEnhancer",
    "CLAHEEnhancer",
    "Restormer",
    "SwinIR",
    "MIRNet",
    "ENHANCEMENT_MODELS",
    "get_enhancement_model",
]
