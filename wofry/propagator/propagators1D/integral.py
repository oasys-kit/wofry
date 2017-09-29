import numpy

from srxraylib.util.data_structures import ScaledArray
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class Integral1D(Propagator1D):

    HANDLER_NAME = "INTEGRAL_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    """
    1D Fresnel-Kirchhoff propagator via simplified integral
    :param wavefront:
    :param propagation_distance: propagation distance
    :param detector_abscissas: a numpy array with the anscissas at the image position. If undefined ([None])
                            it uses the same abscissas present in input wavefront.
    :return: a new 1D wavefront object with propagated wavefront
    """
    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not parameters.has_additional_parameter("detector_abscissas"):
            detector_abscissas = [None]
        else:
            detector_abscissas = parameters.get_additional_parameter("detector_abscissas")

        if detector_abscissas[0] == None:
            detector_abscissas = wavefront.get_abscissas()

        # calculate via outer product, it spreads over a lot of memory, but it is OK for 1D
        x1 = numpy.outer(wavefront.get_abscissas(),numpy.ones(detector_abscissas.size))
        x2 = numpy.outer(numpy.ones(wavefront.size()),detector_abscissas)
        r = numpy.sqrt( numpy.power(x1-x2,2) + numpy.power(propagation_distance,2) )
        wavenumber = numpy.pi*2/wavefront.get_wavelength()
        distances_matrix  = numpy.exp(1.j * wavenumber *  r)


        fieldComplexAmplitude = numpy.dot(wavefront.get_complex_amplitude(),distances_matrix)

        return GenericWavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fieldComplexAmplitude,
                                                                                                detector_abscissas[0],
                                                                                                detector_abscissas[1]-detector_abscissas[0]))
