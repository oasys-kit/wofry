
from wofry.propagator.wavefront import Wavefront

class LightSourceDecorator():

    def get_wavefront(self, wavefront_parameters):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)


class OpticalElementDecorator(object):

    def __init__(self):
        super().__init__()

    def applyOpticalElement(self, wavefront=Wavefront(), parameters=None):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\naccepts " + Wavefront.__module__ + "." + Wavefront.__name__ +
                                  "\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)
