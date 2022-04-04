__all__ = []

import dl.nn
import dl.Layers
import dl.dataset
import dl.metrics
import dl.optimizer
import dl.utils

from dl.graph.variable import *
from dl.graph.op import *


import numpy
numpy.seterr(all='ignore')
dtype = numpy.float64
