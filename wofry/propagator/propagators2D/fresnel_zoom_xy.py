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
import scipy.constants as codata

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class FresnelZoomXY2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_ZOOM_XY_2D"

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

    def do_specific_progation(self, wavefront1, propagation_distance, parameters, element_index=None):

        shift_half_pixel = self.get_additional_parameter("shift_half_pixel",False,parameters,element_index=element_index)
        m_x = self.get_additional_parameter("magnification_x",1.0,parameters,element_index=element_index)
        m_y = self.get_additional_parameter("magnification_y",1.0,parameters,element_index=element_index)
        return self.propagate_wavefront(wavefront1,propagation_distance, magnification_x=m_x, magnification_y=m_y,shift_half_pixel=shift_half_pixel)


    @classmethod
    def propagate_wavefront(cls,wavefront1,propagation_distance,magnification_x=1.0,magnification_y=1.0,shift_half_pixel=False):


        wavefront = wavefront1.duplicate()
        wavelength = wavefront.get_wavelength()
        wavenumber = wavefront.get_wavenumber()

        shape = wavefront.size()
        delta = wavefront.delta()

        pixelsize = delta[0]
        npixels = shape[0]
        freq_nyquist = 0.5 / pixelsize
        freq_n = numpy.linspace(-1.0, 1.0, npixels)
        freq_x = freq_n * freq_nyquist

        # frequency for axis 2
        pixelsize = delta[1]
        npixels = shape[1]
        freq_nyquist = 0.5 / pixelsize
        freq_n = numpy.linspace(-1.0, 1.0, npixels)
        freq_y = freq_n * freq_nyquist

        if shift_half_pixel:
            freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
            freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

        f_x, f_y = numpy.meshgrid(freq_x, freq_y, indexing='ij')
        fsq = numpy.fft.fftshift(f_x ** 2 / magnification_x + f_y ** 2 / magnification_y)

        x = wavefront.get_mesh_x()
        y = wavefront.get_mesh_y()

        x_rescaling = wavefront.get_mesh_x() * magnification_x
        y_rescaling = wavefront.get_mesh_y() * magnification_y

        r1sq = x ** 2 * (1 - magnification_x) + y ** 2 * (1 - magnification_y)
        r2sq = x_rescaling ** 2 * ((magnification_x - 1) / magnification_x) + y_rescaling ** 2 * ((magnification_y - 1) / magnification_y)

        Q1 = wavenumber / 2 / propagation_distance * r1sq
        Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
        Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

        wavefront.add_phase_shift(Q1)

        fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

        ifft = numpy.fft.ifft2(fft * Q2) * Q3 / numpy.sqrt(magnification_x * magnification_y)

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x()*magnification_x,
                                                                            y_array=wavefront.get_coordinate_y()*magnification_y,
                                                                            z_array=ifft,
                                                                            wavelength=wavelength)
        return wf_propagated

