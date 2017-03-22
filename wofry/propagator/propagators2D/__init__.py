
from wofry.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofry.propagator.propagators2D.fresnel import Fresnel2D, FresnelConvolution2D
from wofry.propagator.propagators2D.integral import Integral2D
from wofry.propagator.propagator import PropagationManager

def initialize_default_propagator_2D():
    propagator = PropagationManager.Instance()

    propagator.add_propagator(Fraunhofer2D())
    propagator.add_propagator(Fresnel2D())
    propagator.add_propagator(FresnelConvolution2D())
    propagator.add_propagator(Integral2D())
