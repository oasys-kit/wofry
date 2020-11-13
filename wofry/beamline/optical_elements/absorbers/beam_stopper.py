
from syned.beamline.optical_elements.absorbers.beam_stopper import BeamStopper
from syned.beamline.shape import BoundaryShape, Rectangle, Circle, Ellipse

from wofry.beamline.decorators import OpticalElementDecorator

class WOBeamStopper(BeamStopper, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        BeamStopper.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip_square(boundaries[0], boundaries[1], boundaries[2], boundaries[3], negative=True)
        elif isinstance(self._boundary_shape, Circle):
            wavefront.clip_circle(boundaries[0], boundaries[1], boundaries[2], negative=True)
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

    def to_python_code(self):
        boundary_shape = self.get_boundary_shape()
        txt = "\nfrom syned.beamline.shape import *"
        if isinstance(boundary_shape, Rectangle):
            txt += "\nboundary_shape=Rectangle(%g, %g, %g, %g)" % boundary_shape.get_boundaries()
        elif isinstance(boundary_shape, Circle):
            txt += "\nboundary_shape=Circle(%g, %g, %g)" % boundary_shape.get_boundaries()
        elif isinstance(boundary_shape, Ellipse):
            txt += "\nboundary_shape=Ellipse(%g, %g, %g, %g)" % boundary_shape.get_boundaries()
        txt += "\n"
        txt += "from wofry.beamline.optical_elements.absorbers.beam_stopper import WOBeamStopper"
        txt += "\n"
        txt += "optical_element = WOBeamStopper(boundary_shape=boundary_shape)"
        txt += "\n"
        return txt


class WOBeamStopper1D(BeamStopper, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        BeamStopper.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip(boundaries[0], boundaries[1], negative=True)
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

    def to_python_code(self):
        boundary_shape = self.get_boundary_shape()
        txt = "\nfrom syned.beamline.shape import *"
        if isinstance(boundary_shape, Rectangle):
            txt += "\nboundary_shape=Rectangle(%g, %g, %g, %g)" % boundary_shape.get_boundaries()
        else:
            txt += "\n# ERROR getting boundary shape..."
        txt += "\n"
        txt += "from wofry.beamline.optical_elements.absorbers.beam_stopper import WOBeamStopper1D"
        txt += "\n"
        txt += "optical_element = WOBeamStopper1D(boundary_shape=boundary_shape)"
        txt += "\n"
        return txt


