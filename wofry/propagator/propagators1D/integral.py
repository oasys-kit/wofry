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

    # 1D Fresnel-Kirchhoff propagator via simplified integral
    def do_specific_progation(self, wavefront, propagation_distance, parameters):

        try:
            mX = parameters.get_additional_parameter("magnification_x")
        except:
            mX = 1.0

        try:
            mN = parameters.get_additional_parameter("magnification_N")
        except:
            mN = 1.0

        method = 0

        wavenumber = numpy.pi*2/wavefront.get_wavelength()

        x = wavefront.get_abscissas()


        if mN != 1.0:
            npoints_exit = int(mN * x.size)
        else:
            npoints_exit = x.size

        detector_abscissas = numpy.linspace(mX*x[0],mX*x[-1],npoints_exit)

        if method == 0:
            # calculate via loop pver detector coordinates
            x1 = wavefront.get_abscissas()
            x2 = detector_abscissas
            fieldComplexAmplitude = numpy.zeros_like(x2,dtype=complex)
            for ix,x in enumerate(x2):
                r = numpy.sqrt( numpy.power(x1-x,2) + numpy.power(propagation_distance,2) )
                distances_array  = numpy.exp(1.j * wavenumber *  r)
                fieldComplexAmplitude[ix] = (wavefront.get_complex_amplitude() * distances_array).sum()
        elif method == 1:
            # calculate via outer product, it spreads over a lot of memory, but it is OK for 1D
            x1 = numpy.outer(wavefront.get_abscissas(),numpy.ones(detector_abscissas.size))
            x2 = numpy.outer(numpy.ones(wavefront.size()),detector_abscissas)
            r = numpy.sqrt( numpy.power(x1-x2,2) + numpy.power(propagation_distance,2) )

            distances_matrix  = numpy.exp(1.j * wavenumber *  r)
            fieldComplexAmplitude = numpy.dot(wavefront.get_complex_amplitude(),distances_matrix)


        wavefront_out =  GenericWavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fieldComplexAmplitude,
                                                                                                detector_abscissas[0],
                                                                                                detector_abscissas[1]-detector_abscissas[0]))

        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out
