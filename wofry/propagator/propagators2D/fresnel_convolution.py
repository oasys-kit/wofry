#   propagate_2D_fresnel               \
#   propagate_2D_fresnel_convolution   | Near field Fresnel propagators via convolution in Fourier space. Three methods
#   propagate_2D_fresnel_srw          /
#
#          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
#                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
#                                   'srw': use the SRW package
#
#
#
# *********************************** IMPORTANT *******************************************
#                RECOMMENDATIONS:
#
#     >>> Prefer propagate_2D_fresnel <<<
#       Prefer EVEN number of bins.
#       Set shift_half_pixel=1 (now the default)
#    Under these circumstances, the results agree very well with SRW
#
#
#


import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class FresnelConvolution2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME


    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):

        shift_half_pixel = self.get_additional_parameter("shift_half_pixel",False,parameters,element_index=element_index)

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            pass
        else:
            wavefront = wavefront.toGenericWavefront()

        return self.propagate_wavefront(wavefront,propagation_distance,shift_half_pixel=shift_half_pixel)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shift_half_pixel=False):

        from scipy.signal import fftconvolve

        wavelength = wavefront.get_wavelength()

        X = wavefront.get_mesh_x()
        Y = wavefront.get_mesh_y()

        if shift_half_pixel:
            x = wavefront.get_coordinate_x()
            y = wavefront.get_coordinate_y()
            X += 0.5 * numpy.abs( x[0] - x[1] )
            Y += 0.5 * numpy.abs( y[0] - y[1] )

        kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() *
                           (X**2 + Y**2) / 2 / propagation_distance)
        kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
        kernel /=  1j * wavefront.get_wavelength() * propagation_distance

        wavefront_out = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x(),
                                                                            y_array=wavefront.get_coordinate_y(),
                                                                            z_array=fftconvolve(wavefront.get_complex_amplitude(),
                                                                                                kernel,
                                                                                                mode='same'),
                                                                            wavelength=wavelength)
        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out
