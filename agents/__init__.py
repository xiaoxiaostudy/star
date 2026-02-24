"""
Agents Module
Contains specialized agents:
- BenchmarkAgent: Collects benchmark information
- ModelAgent: Collects model information
"""


from .benchmark_agent import BenchmarkAgent
from .model_agent import ModelAgent
__all__ = [
    "BenchmarkAgent",
    "ModelAgent",
]

