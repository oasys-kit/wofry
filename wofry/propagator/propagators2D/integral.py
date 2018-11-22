#   propagate_2D_integral: Simplification of the Kirchhoff-Fresnel integral. TODO: Very slow and give some problems

import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

# TODO: check resulting amplitude normalization (fft and srw likely agree, convolution gives too high amplitudes, so needs normalization)

class Integral2D(Propagator2D):

    HANDLER_NAME = "INTEGRAL_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    """
    2D Fresnel-Kirchhoff propagator via simplified integral

    NOTE: this propagator is experimental and much less performant than the ones using Fourier Optics
          Therefore, it is not recommended to use.

    :param wavefront:
    :param propagation_distance: propagation distance
    :param shuffle_interval: it is known that this method replicates the central diffraction spot
                            The distace of the replica is proportional to 1/pixelsize
                            To avoid that, it is possible to change a bit (randomly) the coordinates
                            of the wavefront. shuffle_interval controls this shift: 0=No shift. A typical
                             value can be 1e5.
                             The result shows a diffraction pattern without replica but with much noise.
    :param calculate_grid_only: if set, it calculates only the horizontal and vertical profiles, but returns the
                             full image with the other pixels to zero. This is useful when calculating large arrays,
                             so it is set as the default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):

        shuffle_interval = self.get_additional_parameter("shuffle_interval",False,parameters,element_index=element_index)
        calculate_grid_only = self.get_additional_parameter("calculate_grid_only",True,parameters,element_index=element_index)

        return self.propagate_wavefront(wavefront,propagation_distance,shuffle_interval=shuffle_interval,
                                 calculate_grid_only=calculate_grid_only)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shuffle_interval=False,calculate_grid_only=True):
        #
        # Fresnel-Kirchhoff integral (neglecting inclination factor)
        #

        if not calculate_grid_only:
            #
            # calculation over the whole detector area
            #
            p_x = wavefront.get_coordinate_x()
            p_y = wavefront.get_coordinate_y()
            wavelength = wavefront.get_wavelength()
            amplitude = wavefront.get_complex_amplitude()

            det_x = p_x.copy()
            det_y = p_y.copy()

            p_X = wavefront.get_mesh_x()
            p_Y = wavefront.get_mesh_y()

            det_X = p_X
            det_Y = p_Y


            amplitude_propagated = numpy.zeros_like(amplitude,dtype='complex')

            wavenumber = 2 * numpy.pi / wavelength

            for i in range(det_x.size):
                for j in range(det_y.size):
                    if not shuffle_interval:
                        rd_x = 0.0
                        rd_y = 0.0
                    else:
                        rd_x = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval
                        rd_y = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval

                    r = numpy.sqrt( numpy.power(p_X + rd_x - det_X[i,j],2) +
                                    numpy.power(p_Y + rd_y - det_Y[i,j],2) +
                                    numpy.power(propagation_distance,2) )

                    amplitude_propagated[i,j] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()

            output_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(det_x,det_y,amplitude_propagated)

        else:
            x = wavefront.get_coordinate_x()
            y = wavefront.get_coordinate_y()
            X = wavefront.get_mesh_x()
            Y = wavefront.get_mesh_y()
            wavenumber = 2 * numpy.pi / wavefront.get_wavelength()
            amplitude = wavefront.get_complex_amplitude()

            used_indices = wavefront.get_mask_grid(width_in_pixels=(1,1),number_of_lines=(1,1))
            indices_x = wavefront.get_mesh_indices_x()
            indices_y = wavefront.get_mesh_indices_y()

            indices_x_flatten = indices_x[numpy.where(used_indices == 1)].flatten()
            indices_y_flatten = indices_y[numpy.where(used_indices == 1)].flatten()
            X_flatten         =         X[numpy.where(used_indices == 1)].flatten()
            Y_flatten         =         Y[numpy.where(used_indices == 1)].flatten()
            complex_amplitude_propagated = amplitude*0

            print("propagate_2D_integral: Calculating %d points from a total of %d x %d = %d"%(
                X_flatten.size,amplitude.shape[0],amplitude.shape[1],amplitude.shape[0]*amplitude.shape[1]))

            for i in range(X_flatten.size):
                r = numpy.sqrt( numpy.power(wavefront.get_mesh_x() - X_flatten[i],2) +
                                numpy.power(wavefront.get_mesh_y() - Y_flatten[i],2) +
                                numpy.power(propagation_distance,2) )

                complex_amplitude_propagated[int(indices_x_flatten[i]),int(indices_y_flatten[i])] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()

            output_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=x,
                                                                                   y_array=y,
                                                                                   z_array=complex_amplitude_propagated,
                                                                                   wavelength=wavefront.get_wavelength())

        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        output_wavefront.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    output_wavefront.get_intensity().sum()))

        return output_wavefront
