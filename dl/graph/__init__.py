from .op import *
from .variable import *

# Layers created will hold variable of weight and bias
# Create placeholder for input values
# Compute to create the final y
# go backward to compute gradient for each variables
#   if variable is not constant, gradient ++
#   if not, recurrent over
