"""
Baseline methods of the DemandClean experiments.

1. Data-oriented cleaning: HoloClean, Baran_Raha, Horizon, UniClean, Lopster
2. Model-oriented cleaning: ActiveClean, BoostClean
3. Data preparation: ctxpipe, MLImputer, SimpleImputer
"""

# Methods with a complete implementation
# from .Horizon import *
# from .Baran_Raha import *

# Methods that need extra dependencies
# from .HoloClean import HoloCleanWrapper
# from .UniClean import UniCleanWrapper
# from .Lopster import LopsterWrapper
# from .ActiveClean import ActiveCleanWrapper
# from .BoostClean import BoostCleanWrapper
# from .MLImputer import MLImputerWrapper
# from .SimpleImputer import SimpleImputerWrapper

__all__ = [
    # Data-oriented cleaning methods
    'HoloClean',
    'Baran_Raha',
    'Horizon',
    'UniClean',
    'Lopster',
    # Model-oriented cleaning methods
    'ActiveClean',
    'BoostClean',
    # Data preparation methods
    'ctxpipe',
    'MLImputer',
    'SimpleImputer',
]

__version__ = '1.0.0'
