
from syned.storage_ring.light_source import LightSource
from syned.storage_ring.electron_beam import ElectronBeam
from syned.storage_ring.magnetic_structures.undulator import Undulator
from syned.beamline.beamline import Beamline
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.optical_elements.absorbers.slit import Slit
from syned.beamline.optical_elements.shape import Rectangle

from wofry.elements.decorators import WOLightSourceDecorator, WOOpticalElementDecorator
from wofry.propagator.wavefront import Wavefront
from wofry.propagator.propagator_chain import PropagatorsChain, AbstractPropagator, PropagationParameters

class WavefrontTest(Wavefront):

    def __init__(self):
        self._test_parameter = 0

    def set_test_parameter(self, value):
        self._test_parameter = value

    def get_test_parameter(self):
        return self._test_parameter

class UndulatorTest(LightSource, WOLightSourceDecorator):

    def __init__(self):
        super().__init__(name="UndulatorTest",
                         electron_beam=ElectronBeam.initialize_as_pencil_beam(energy_in_GeV=2.0,
                                                                              energy_spread=0.0008,
                                                                              current=0.3),
                         magnetic_structure=Undulator.initialize_as_vertical_undulator(K=1.0223,
                                                                                       period_length=0.01,
                                                                                       periods_number=22))

    def get_wavefront(self):
        return WavefrontTest()

class SlitTest(Slit, WOOpticalElementDecorator):

    def __init__(self):

        Slit.__init__(self, name="SlitTest", boundary_shape=Rectangle(x_left=-0.01, x_right=0.01, y_bottom=0.01, y_top=0.01))
        WOOpticalElementDecorator.__init__(self)


    def applyOpticalElement(self, wavefront):
        wavefront.set_test_parameter(10)

        return wavefront


class PropagatorTest(AbstractPropagator):

    HANDLER_NAME = "TEST"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_propagation(self, parameters=PropagationParameters()):
        beamline = parameters.get_beamline()

        source = beamline.get_light_source()
        wavefront = source.get_wavefront()

        for element in beamline.get_beamline_elements():
            wavefront = element.get_optical_element().applyOpticalElement(wavefront)

        return wavefront



propagator = PropagatorsChain.Instance()
propagator.add_propagator(PropagatorTest())

light_source = UndulatorTest()

beamline = Beamline()
beamline.set_light_source(light_source)
beamline.append_beamline_element(BeamlineElement(coordinates=ElementCoordinates(), optical_element=SlitTest()))

wavefront = propagator.do_propagation(PropagationParameters(wavefront=None, beamline=beamline), handler_name=PropagatorTest.HANDLER_NAME)

print (wavefront.get_test_parameter())