"""cheatdetect: behavioural cheat-detection pipeline for CS view-angle telemetry.

Week 1-2 scope: data loading + light cleaning, EDA, and behavioural feature
engineering. Modelling, the stratified split, and imbalance handling (SMOTE +
class weights) are deliberately left for Week 3+.
"""
from . import config, data_loading, features  # noqa: F401

__all__ = ["config", "data_loading", "features"]
__version__ = "0.1.0"
