import numpy

from srxraylib.util.data_structures import ScaledArray
from wofry.propagator.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Generic1DPropagator, PropagationParameters, PropagationElements

class Integral1DPropagationParameters(PropagationParameters):
    def __init__(self,
                 wavefront = GenericWavefront1D(),
                 propagation_elements = PropagationElements(),
                 detector_abscissas=[None]):
        PropagationParameters.__init__(wavefront, propagation_elements)

        self._detector_abscissas=detector_abscissas

    def detector_abscissas(self):
        return self._detector_abscissas

class Integral1D(Generic1DPropagator):

    def get_handler_name(self):
        return "INTEGRAL_1D"

    """
    1D Fresnel-Kirchhoff propagator via simplified integral
    :param wavefront:
    :param propagation_distance: propagation distance
    :param detector_abscissas: a numpy array with the anscissas at the image position. If undefined ([None])
                            it uses the same abscissas present in input wavefront.
    :return: a new 1D wavefront object with propagated wavefront
    """
    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not isinstance(parameters, Integral1DPropagationParameters):
            raise ValueError("parameters are not " + Integral1DPropagationParameters.__name__)

        detector_abscissas = parameters.get_detector_abscissas()


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
