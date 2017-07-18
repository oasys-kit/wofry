
from syned.storage_ring.light_source import LightSource
from syned.storage_ring.electron_beam import ElectronBeam
from syned.storage_ring.magnetic_structures.undulator import Undulator
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.optical_elements.absorbers.slit import Slit
from syned.beamline.optical_elements.mirrors.mirror import Mirror
from syned.beamline.shape import Rectangle, Ellipsoid

from wofry.beamline.decorators import WOLightSourceDecorator, WOOpticalElementDecorator
from wofry.propagator.decorators import WavefrontDecorator
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D, WavefrontDimension
from wofry.propagator.propagator import PropagationManager, AbstractPropagator, PropagationParameters, PropagationElements

# ----------------------------------------------
# ENGINE 1 (es. SRW)

class WavefrontEngine1(GenericWavefront2D, WavefrontDecorator):

    def __init__(self):
        self._test_parameter = 0

    def set_test_parameter(self, value):
        self._test_parameter = value

    def get_test_parameter(self):
        return self._test_parameter

    def toGenericWavefront(self):
        return self

    def fromGenericWavefront(self, wavefront):
        self.set_test_parameter(0)

class UndulatorEngine1(LightSource, WOLightSourceDecorator):

    def __init__(self):
        super().__init__(name="UndulatorTest",
                         electron_beam=ElectronBeam.initialize_as_pencil_beam(energy_in_GeV=2.0,
                                                                              energy_spread=0.0008,
                                                                              current=0.3),
                         magnetic_structure=Undulator.initialize_as_vertical_undulator(K=1.0223,
                                                                                       period_length=0.01,
                                                                                       periods_number=22))

    def get_wavefront(self):
        wavefront = WavefrontEngine1()
        wavefront.set_test_parameter(100)

        return wavefront


class SlitEngine1(Slit, WOOpticalElementDecorator):

    def __init__(self):
        Slit.__init__(self,
                      name="SlitTest",
                      boundary_shape=Rectangle(x_left=-0.01, x_right=0.01, y_bottom=-0.01, y_top=0.01))
        WOOpticalElementDecorator.__init__(self)

    def applyOpticalElement(self, wavefront):
        wavefront.set_test_parameter(10)

        return wavefront

class PropagatorEngine1(AbstractPropagator):

    HANDLER_NAME = "TEST1"

    def get_dimension(self):
        return WavefrontDimension.TWO

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_propagation(self, parameters=PropagationParameters()):
        wavefront = parameters.get_wavefront()

        if not isinstance(wavefront, WavefrontEngine1): raise Exception("wrong wavefront! it is not WavefrontTest")

        for element in parameters.get_PropagationElements().get_propagation_elements():
            wavefront = element.get_optical_element().applyOpticalElement(wavefront)

        return wavefront

# -------------------------------------------------------
# ENGINE 2

class WavefrontEngine2(GenericWavefront1D, WavefrontDecorator):

    def __init__(self):
        self._test_parameter_1 = 10
        self._test_parameter_2 = 20

    def set_test_parameter_1(self, value):
        self._test_parameter_1 = value

    def get_test_parameter_1(self):
        return self._test_parameter_1

    def set_test_parameter_2(self, value):
        self._test_parameter_2 = value

    def get_test_parameter_2(self):
        return self._test_parameter_2

    def toGenericWavefront(self):
        return self

    def fromGenericWavefront(self, wavefront):
        self.set_test_parameter_1(10)
        self.set_test_parameter_2(20)

class MirrorEngine2(Mirror, WOOpticalElementDecorator):

    def __init__(self):
        Mirror.__init__(self,
                        name="MirrorEngine2",
                        boundary_shape=Rectangle(x_left=-10.0, x_right=10.0, y_bottom=-10.0, y_top=10.0),
                        surface_shape=Ellipsoid())
        WOOpticalElementDecorator.__init__(self)

    def applyOpticalElement(self, wavefront):
        wavefront.set_test_parameter_1(30)
        wavefront.set_test_parameter_2(40)

        return wavefront

class PropagatorEngine2(AbstractPropagator):

    HANDLER_NAME = "TEST2"

    def get_dimension(self):
        return WavefrontDimension.ONE

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_propagation(self, parameters=PropagationParameters()):
        wavefront = parameters.get_wavefront()

        if not isinstance(wavefront, WavefrontEngine2): raise Exception("wrong wavefront!  it is not WavefrontEngine2")

        for element in parameters.get_PropagationElements().get_propagation_elements():
            wavefront = element.get_optical_element().applyOpticalElement(wavefront)

        return wavefront

if __name__=="__main__":

    # -------------- INITIALIZATION OF CHAIN OF RESPONSIBILITY
    #
    # MUST BE DONE ONCE

    propagator = PropagationManager.Instance()
    propagator.add_propagator(PropagatorEngine1())
    propagator.add_propagator(PropagatorEngine2())

    #---------------------------

    light_source = UndulatorEngine1()
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(coordinates=ElementCoordinates(), 
                                                              optical_element=SlitEngine1()))

    initial_wavefront = light_source.get_wavefront()

    print("TEST1 initial WF:           test parameter = ", initial_wavefront.get_test_parameter())

    wavefront1 = propagator.do_propagation(PropagationParameters(wavefront=light_source.get_wavefront(),
                                                                propagation_elements=propagation_elements),
                                          handler_name="TEST1")

    print("TEST1 : propagate from source to Slit")

    print("TEST1 after WF propagation: test parameter = ", wavefront1.get_test_parameter())

    propagation_elements2 = PropagationElements()
    propagation_elements2.add_beamline_element(BeamlineElement(coordinates=ElementCoordinates(), optical_element=MirrorEngine2()))

    print("switch to TEST2 from TEST1 - > propagate from slit to mirror")

    initial_wavefront2 = WavefrontEngine2()
    initial_wavefront2.fromGenericWavefront(wavefront1.toGenericWavefront().get_Wavefront1D_from_histogram(axis=1))

    print("TEST2 initial WF:           test parameters = ", initial_wavefront2.get_test_parameter_1(), ", ", initial_wavefront2.get_test_parameter_2())

    final_wavefront = propagator.do_propagation(PropagationParameters(wavefront=initial_wavefront2,
                                                                      propagation_elements=propagation_elements2),
                                                handler_name="TEST2")

    print("TEST2 after WF propagation: test parameters = ", final_wavefront.get_test_parameter_1(), ", ", final_wavefront.get_test_parameter_2())