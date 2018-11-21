"""
Represents an ideal lens.
"""
from syned.beamline.optical_elements.ideal_elements.screen import Screen
from wofry.beamline.decorators import OpticalElementDecorator

class WOScreen(Screen, OpticalElementDecorator):
    def __init__(self, name="Undefined"):
        Screen.__init__(self, name=name)

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):
        return wavefront

class WOScreen1D(Screen, OpticalElementDecorator):
    def __init__(self, name="Undefined"):
        Screen.__init__(self, name=name)

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):
        return wavefront
