
import numpy

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D, PropagationParameters, PropagationElements

class Fraunhofer1D(Propagator1D):

    HANDLER_NAME = "FRAUNHOFER_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    """
    1D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :return: a new 1D wavefront object with propagated wavefront
    """

    # TODO: check resulting amplitude normalization
    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        shape = wavefront.size()
        delta = wavefront.delta()
        wavelength = wavefront.get_wavelength()
        wavenumber = wavefront.get_wavenumber()
        fft_scale = numpy.fft.fftfreq(shape, d=delta)
        fft_scale = numpy.fft.fftshift(fft_scale)
        x2 = fft_scale * propagation_distance * wavelength

        try:
            if parameters.get_additional_parameter("shift_half_pixel"):
                x2 = x2 - 0.5 * numpy.abs(x2[1] - x2[0])
        except:
            pass


        p1 = numpy.exp(1.0j * wavenumber * propagation_distance)
        p2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * x2**2)
        p3 = 1.0j*wavelength*propagation_distance

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        fft = fft * p1 * p2 / p3
        fft2 = numpy.fft.fftshift(fft)

        wavefront_out =  GenericWavefront1D.initialize_wavefront_from_arrays(x2, fft2, wavelength=wavefront.get_wavelength())


        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out