"""
Represents an ideal lens.
"""

from syned.beamline.optical_elements.ideal_elements.lens import IdealLens
from wofry.elements.decorators import WOOpticalElementDecorator

class WOIdealLens(IdealLens, WOOpticalElementDecorator):
    def __init__(self, name, focal_x, focal_y):
        IdealLens.__init__(self, name, focal_x, focal_y)

    def applyOpticalElement(self, wavefront):
        wavefront.add_phase_shifts( -1.0  * wavefront.get_wavenumber() *
                ( (wavefront.get_mesh_x()**2/self.focal_x() + wavefront.get_mesh_y()**2/self.focal_y()) / 2))

        return wavefront

class WOIdealLens1D(IdealLens, WOOpticalElementDecorator):
    def __init__(self, name, focal_length):
        IdealLens.__init__(self, name, focal_length, focal_length)

    def applyOpticalElement(self, wavefront):
        wavefront.add_phase_shift((-1.0) * wavefront.get_wavenumber() * (wavefront.get_abscissas() ** 2 / self.focal_x()) / 2)

        return wavefront