
from wofry.propagator.generic_wavefront import GenericWavefront

class WOLightSourceDecorator():

    def get_wavefront(self):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\nreturns " + GenericWavefront.__module__ + "." + GenericWavefront.__name__)


class WOOpticalElementDecorator(object):

    def __init__(self):
        super().__init__()

    def applyOpticalElement(self, wavefront=GenericWavefront()):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\naccepts " + GenericWavefront.__module__ + "." + GenericWavefront.__name__ +
                                  "\nreturns " + GenericWavefront.__module__ + "." + GenericWavefront.__name__)
