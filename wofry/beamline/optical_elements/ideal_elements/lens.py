"""
Represents an ideal lens.
"""

from syned.beamline.optical_elements.ideal_elements.lens import IdealLens
from wofry.beamline.decorators import OpticalElementDecorator

class WOIdealLens(IdealLens, OpticalElementDecorator):
    def __init__(self, name, focal_x, focal_y):
        IdealLens.__init__(self, name, focal_x, focal_y)

    def applyOpticalElement(self, wavefront, parameters=None):

        focal_term = 0.0

        if self.focal_x() is not None:
            if self.focal_x() != 0.0:
                focal_term += wavefront.get_mesh_x()**2/self.focal_x()

        if self.focal_y() is not None:
            if self.focal_y() != 0.0:
                focal_term += wavefront.get_mesh_y()**2/self.focal_y()

        wavefront.add_phase_shifts( -1.0  * wavefront.get_wavenumber() * ( focal_term / 2))

        return wavefront

class WOIdealLens1D(WOIdealLens):
    def __init__(self, name, focal_length, plane='horizontal'):
        if plane == 'horizontal':
            focal_length_x = focal_length
            focal_length_y = None
        elif plane == 'vertical':
            focal_length_x = None
            focal_length_y = focal_length
        else:
            raise Exception("invalid focusing plane: plane must be horizontal or vertical")

        WOIdealLens.__init__(self, name, focal_length_x, focal_length_y)
