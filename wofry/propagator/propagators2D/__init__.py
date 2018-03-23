
from wofry.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofry.propagator.propagators2D.fresnel import Fresnel2D
from wofry.propagator.propagators2D.fresnel_convolution import FresnelConvolution2D
from wofry.propagator.propagators2D.integral import Integral2D
from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofry.propagator.propagator import PropagationManager

def initialize_default_propagator_2D():
    propagator = PropagationManager.Instance()

    propagator.add_propagator(Fraunhofer2D())
    propagator.add_propagator(Fresnel2D())
    propagator.add_propagator(FresnelConvolution2D())
    propagator.add_propagator(Integral2D())
    propagator.add_propagator(FresnelZoomXY2D())
