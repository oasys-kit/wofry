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

angstroms_to_eV = codata.h*codata.c/codata.e*1e10

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class FresnelZoomXY2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_ZOOM_XY_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront1, propagation_distance, parameters):
        if not parameters.has_additional_parameter("shift_half_pixel"):
            shift_half_pixel = True
        else:
            shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        wavefront = wavefront1.duplicate()
        shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        wavelength = wavefront.get_wavelength()


        wavenumber = wavefront.get_wavenumber()

        m_x = parameters.get_additional_parameter("magnification_x")
        m_y = parameters.get_additional_parameter("magnification_y")

        shape = wavefront.size()
        delta = wavefront.delta()

        pixelsize = delta[0]  # p_x[1] - p_x[0]
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
        fsq = numpy.fft.fftshift(f_x ** 2 / m_x + f_y ** 2 / m_y)

        x = wavefront.get_mesh_x()
        y = wavefront.get_mesh_y()

        x_rescaling = wavefront.get_mesh_x() * m_x
        y_rescaling = wavefront.get_mesh_y() * m_y

        r1sq = x ** 2 * (1 - m_x) + y ** 2 * (1 - m_y)
        r2sq = x_rescaling ** 2 * ((m_x - 1) / m_x) + y_rescaling ** 2 * ((m_y - 1) / m_y)

        Q1 = wavenumber / 2 / propagation_distance * r1sq
        Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
        Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

        wavefront.add_phase_shift(Q1)

        fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

        ifft = numpy.fft.ifft2(fft * Q2) * Q3 / numpy.sqrt(m_x * m_y)

        # wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x() * m_x,
        #                                                              ####################NON SONO SICURO CHE SIA GIUSTO QUELLO CHE RETURN
        #                                                              wavefront.get_coordinate_y() * m_y,
        #                                                              ifft,
        #                                                              wavelength=wavelength)
        #===============
        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x()*m_x,
                                                                            y_array=wavefront.get_coordinate_y()*m_y,
                                                                            z_array=ifft,
                                                                            wavelength=wavelength)

        return wf_propagated

class FresnelConvolution2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME


    def do_specific_progation_after(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            pass
        else:
            wavefront_original = wavefront
            wavefront = wavefront.toGenericWavefront()

        if not parameters.has_additional_parameter("shift_half_pixel"):
            shift_half_pixel = True
        else:
            shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

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

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x(),
                                                                            y_array=wavefront.get_coordinate_y(),
                                                                            z_array=fftconvolve(wavefront.get_complex_amplitude(),
                                                                                                kernel,
                                                                                                mode='same'),
                                                                            wavelength=wavelength)
        return wf_propagated
