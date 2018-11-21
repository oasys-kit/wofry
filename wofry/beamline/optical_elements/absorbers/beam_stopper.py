
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



