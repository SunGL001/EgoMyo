"""MyoBridge reference implementation."""

from .components import (CausalChannelEMGEncoder, ContactSupportedForceField,
                         GranularityMatchedEncoder, RegionalVertexLifting,
                         VisionAnchoredRouting)
from .data import TrialBlockBatches, TrialDataset
from .losses import augment, objective
from .mano import EmbeddedMANO
from .metrics import Metrics, geodesic_neighborhood
from .model import FeatureAdapter, MyoBridge, build_model
from .splits import build_splits

__all__ = [
    'CausalChannelEMGEncoder', 'ContactSupportedForceField', 'GranularityMatchedEncoder',
    'RegionalVertexLifting', 'VisionAnchoredRouting', 'TrialBlockBatches', 'TrialDataset',
    'augment', 'objective', 'EmbeddedMANO', 'Metrics', 'geodesic_neighborhood',
    'FeatureAdapter', 'MyoBridge', 'build_model', 'build_splits',
]
__version__ = '1.0.0'
