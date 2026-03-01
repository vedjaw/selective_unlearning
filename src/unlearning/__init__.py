"""
Unlearning methods initialization.
"""

from .base import BaseUnlearner
from .fine_tune import FineTuneUnlearner
from .gradient_ascent import GradientAscentUnlearner
from .ssd import SSDUnlearner
from .pgu import PGUUnlearner
from .igtu import IGTUUnlearner
from .iweup import IWEUPUnlearner
from .iweup_v2 import IWEUPv2Unlearner
from .scrub import SCRUBUnlearner
from .retrain import RetrainUnlearner
from .bad_teacher import BadTeacherUnlearner
from .amnesiac import AmnesiacUnlearner
from .salun import SalUnUnlearner

__all__ = [
    'BaseUnlearner',
    'FineTuneUnlearner',
    'GradientAscentUnlearner',
    'SSDUnlearner',
    'PGUUnlearner',
    'IGTUUnlearner',
    'IWEUPUnlearner',
    'IWEUPv2Unlearner',
    'SCRUBUnlearner',
    'RetrainUnlearner',
    'BadTeacherUnlearner',
    'AmnesiacUnlearner',
    'SalUnUnlearner',
]


