
from syned.beamline.optical_elements.absorbers.slit import Slit
from syned.beamline.optical_elements.shape import BoundaryShape, Rectangle, Ellipse

from wofry.elements.decorators import WOOpticalElementDecorator

class WOSlit(Slit, WOOpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip_square(boundaries[0], boundaries[1], boundaries[2], boundaries[3])
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

class WOSlit1D(Slit, WOOpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip(boundaries[0], boundaries[1])
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

