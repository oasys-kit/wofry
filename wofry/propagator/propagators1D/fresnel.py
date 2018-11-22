import numpy

from srxraylib.util.data_structures import ScaledArray
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class Fresnel1D(Propagator1D):

    HANDLER_NAME = "FRESNEL_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    """
    1D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :return: a new 1D wavefront object with propagated wavefront
    """
    def do_specific_progation(self, wavefront, propagation_distance, parameters=None, element_index=None):

        return self.propagate_wavefront(wavefront,propagation_distance)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance):

        fft_scale = numpy.fft.fftfreq(wavefront.size())/wavefront.delta()

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        fft *= numpy.exp((-1.0j) * numpy.pi * wavefront.get_wavelength() * propagation_distance * fft_scale**2)
        ifft = numpy.fft.ifft(fft)

        return GenericWavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(ifft, wavefront.offset(), wavefront.delta()))
