#
#   propagate_2D_fraunhofer: Far field Fraunhofer propagator. TODO: Check phases, not to be used for downstream propagation
#
#    The fraunhoffer method cannot be used in a compound system (more than one element) and in connection with lenses
#

import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class Fraunhofer2D(Propagator2D):

    HANDLER_NAME = "FRAUNHOFER_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)


    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):

        shift_half_pixel = self.get_additional_parameter("shift_half_pixel",False,parameters,element_index=element_index)

        return self.propagate_wavefront(wavefront,propagation_distance,shift_half_pixel=shift_half_pixel)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shift_half_pixel=False):
        wavelength = wavefront.get_wavelength()

        #
        # check validity
        #
        x =  wavefront.get_coordinate_x()
        y =  wavefront.get_coordinate_y()
        half_max_aperture = 0.5 * numpy.array((x[-1]-x[0], y[-1]-y[0])).max()
        far_field_distance = half_max_aperture**2/wavelength
        if propagation_distance < far_field_distance:
            print("WARNING: Fraunhoffer diffraction valid for distances > > half_max_aperture^2/lambda = %f m (propagating at %4.1f)"%
                        (far_field_distance,propagation_distance))
        #
        #compute Fourier transform
        #
        F1 = numpy.fft.fft2(wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = numpy.fft.fftshift( F1 )

        # frequency for axis 1
        shape = wavefront.size()
        delta = wavefront.delta()
        wavenumber = wavefront.get_wavenumber()

        pixelsize = delta[0]  # p_x[1] - p_x[0]
        npixels = shape[0]
        fft_scale = numpy.fft.fftfreq(npixels, d=pixelsize)
        fft_scale = numpy.fft.fftshift(fft_scale)
        x2 = fft_scale * propagation_distance * wavelength

        # frequency for axis 2
        pixelsize = delta[1]
        npixels = shape[1]
        fft_scale = numpy.fft.fftfreq(npixels, d=pixelsize)
        fft_scale = numpy.fft.fftshift(fft_scale)
        y2 = fft_scale * propagation_distance * wavelength

        f_x, f_y = numpy.meshgrid(x2, y2, indexing='ij')
        fsq = numpy.fft.fftshift(f_x ** 2 + f_y ** 2)

        P1 = numpy.exp(1.0j * wavenumber * propagation_distance)
        P2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * fsq)
        P3 = 1.0j * wavelength * propagation_distance

        F1 = numpy.fft.fft2(wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
        #  Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F1 *= P1
        F1 *= P2
        F1 /= P3
        F2 = numpy.fft.fftshift(F1)

        if shift_half_pixel:
            x2 = x2 - 0.5 * numpy.abs(x2[1] - x2[0])
            y2 = y2 - 0.5 * numpy.abs(y2[1] - y2[0])

        wavefront_out = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=x2,
                                                                            y_array=y2,
                                                                            z_array=F2,
                                                                            wavelength=wavelength)


        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out