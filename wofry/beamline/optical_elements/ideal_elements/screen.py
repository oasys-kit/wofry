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

    def to_python_code(self):
        txt  = ""
        txt += "\nfrom wofry.beamline.optical_elements.ideal_elements.screen import WOScreen"
        txt += "\n"
        txt += "\noptical_element = WOScreen()"
        txt += "\n"
        return txt

class WOScreen1D(Screen, OpticalElementDecorator):
    def __init__(self, name="Undefined"):
        Screen.__init__(self, name=name)

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):
        return wavefront

    def to_python_code(self):
        txt  = ""
        txt += "\nfrom wofry.beamline.optical_elements.ideal_elements.screen import WOScreen1D"
        txt += "\n"
        txt += "\noptical_element = WOScreen1D()"
        txt += "\n"
        return txt


