
import numpy

from syned.beamline.optical_elements.absorbers.slit import Slit
from syned.beamline.shape import BoundaryShape, Rectangle, Circle, Ellipse

from wofry.beamline.decorators import OpticalElementDecorator

class WOSlit(Slit, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip_square(boundaries[0], boundaries[1], boundaries[2], boundaries[3])
        elif isinstance(self._boundary_shape, Circle):
            wavefront.clip_circle(boundaries[0], boundaries[1], boundaries[2])
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

class WOGaussianSlit(Slit, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None):
        boundaries = self._boundary_shape.get_boundaries()
        aperture_diameter_x =  numpy.abs(boundaries[1] - boundaries[0])
        aperture_diameter_y =  numpy.abs(boundaries[2] - boundaries[3])
        X = wavefront.get_mesh_x()
        Y = wavefront.get_mesh_y()

        wavefront.rescale_amplitudes(numpy.exp(- (((X*X)/2/(aperture_diameter_x/2.35)**2) + \
                                                  ((Y*Y)/2/(aperture_diameter_y/2.35)**2) )))

        return wavefront

class WOSlit1D(Slit, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None):
        boundaries = self._boundary_shape.get_boundaries()

        if isinstance(self._boundary_shape, Rectangle):
            wavefront.clip(boundaries[0], boundaries[1])
        else:
            raise NotImplementedError("to be implemented")

        return wavefront

class WOGaussianSlit1D(Slit, OpticalElementDecorator):
    def __init__(self, name="Undefined", boundary_shape=BoundaryShape()):
        Slit.__init__(self, name=name, boundary_shape=boundary_shape)

    def applyOpticalElement(self, wavefront, parameters=None):
        boundaries = self._boundary_shape.get_boundaries()
        aperture_diameter =  numpy.abs(boundaries[1] - boundaries[0])
        X = wavefront.get_abscissas()

        window = numpy.exp(-(X*X)/2/(aperture_diameter/2.35)**2)
        wavefront.rescale_amplitudes(window)

        return wavefront
