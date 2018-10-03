
import numpy

from syned.beamline.optical_elements.absorbers.slit import Slit
from syned.beamline.shape import BoundaryShape, Rectangle, Circle, Ellipse, MultiplePatch

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
        elif isinstance(self._boundary_shape, Ellipse):
            #clip_ellipse expects axes and center
            a_axis = boundaries[1] - boundaries[0]
            b_axis = boundaries[3] - boundaries[2]
            x_center = 0.5 * (boundaries[0] + boundaries[1])
            y_center = 0.5 * (boundaries[2] + boundaries[3])
            wavefront.clip_ellipse(a_axis, b_axis, x_center, y_center)
        elif isinstance(self._boundary_shape, MultiplePatch):
            windows = []
            for i in range(self._boundary_shape.get_number_of_patches()):
                patch = self._boundary_shape.get_patch(i)
                bd = patch.get_boundaries()
                if self._boundary_shape.get_name_of_patch(i) == "Rectangle":
                    windows.append( wavefront.clip_square(bd[0],bd[1],bd[2],bd[3],apply_to_wavefront=False) )
                elif self._boundary_shape.get_name_of_patch(i) == "Circle":
                    windows.append( wavefront.clip_circle(bd[0],bd[1],bd[2],apply_to_wavefront=False) )
                elif self._boundary_shape.get_name_of_patch(i) == "Ellipse":
                    print(bd)
                    a_axis = bd[1] - bd[0]
                    b_axis = bd[3] - bd[2]
                    x_center = 0.5 * (bd[0] + bd[1])
                    y_center = 0.5 * (bd[2] + bd[3])
                    windows.append( wavefront.clip_ellipse(a_axis,b_axis,x_center,y_center,apply_to_wavefront=False) )
                else:
                    raise Exception(NotImplementedError)

            final_window = numpy.zeros_like(windows[0])

            for i in range(len(windows)):
                final_window += windows[i]

            final_window[ final_window > 0 ] = 1.0 # renormalize

            wavefront.clip_window(final_window)
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
