
from syned.beamline.optical_element import OpticalElement

from wofry.propagator.wavefront import Wavefront

class WOOpticalElement(OpticalElement):

    def __init__(self, name, boundary_shape=None):
        super().__init__(name, boundary_shape)

    def applyOpticalElement(self, wavefront=Wavefront()):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\naccepts " + Wavefront.__module__ + "." + Wavefront.__name__ +
                                  "\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)
