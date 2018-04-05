
from wofry.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofry.propagator.propagators1D.fresnel import Fresnel1D
from wofry.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
from wofry.propagator.propagators1D.integral import Integral1D
from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofry.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D
from wofry.propagator.propagator import PropagationManager

def initialize_default_propagator_1D():
    propagator = PropagationManager.Instance()

    propagator.add_propagator(Fraunhofer1D())
    propagator.add_propagator(Fresnel1D())
    propagator.add_propagator(FresnelConvolution1D())
    propagator.add_propagator(Integral1D())
    propagator.add_propagator(FresnelZoom1D())
    propagator.add_propagator(FresnelZoomScaling1D())
