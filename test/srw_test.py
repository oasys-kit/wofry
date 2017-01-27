

import wpg.srwlib as srw

from wofry.propagator.wavefront import GenericWavefront1D, GenericWavefront2D, WavefrontDimension
from wofry.propagator.decorators import WavefrontDecorator
from wofry.propagator.propagator import AbstractPropagator, PropagationParameters

class WavefrontSRW(GenericWavefront2D, WavefrontDecorator):

    def __init__(self, parameteres=None):

        self.__srwwf =  srw.SRWLWfr()

        # do somethiing with paramaters

    def toGenericWavefront(self):
        # from SRW to Generic
        return GenericWavefront2D()

    def fromGenericWavefront(self, wavefront):
        # modify or replace self.__srwwf
        pass



class PropagatorSRW(AbstractPropagator):

    HANDLER = "SRW"

    def get_dimension(self):
        WavefrontDimension.TWO

    def get_handler_name(self):
        return self.HANDLER

    def do_propagation(self, parameters=PropagationParameters()):

        # build SRW Beamline and its parameters from list of elements

        # do calculation
        return WavefrontSRW()


class WavefrontWISE(GenericWavefront1D, WavefrontDecorator):

    def toGenericWavefront(self):
        # from SRW to Generic
        return GenericWavefront1D()

    def fromGenericWavefront(self, wavefront):
        # modify or replace self.__srwwf
        pass


# source = UndulatorSRW()
# srwwf = source.getWavefront()

srwwf = WavefrontSRW()

# do propagation stuff

#switch to WISE


wisewf = WavefrontWISE()
wisewf.fromGenericWavefront(srwwf.toGenericWavefront().get_Wavefront1D_from_histogram(axis=1))

