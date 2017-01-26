
from syned.storage_ring.electron_beam import ElectronBeam
from syned.storage_ring.magnetic_structure import MagneticStructure
from syned.storage_ring.light_source import LightSource

from wofry.propagator.wavefront import Wavefront

class WOLightSource(LightSource):

    def __init__(self, name="Undefined", electron_beam=ElectronBeam(), magnetic_structure=MagneticStructure()):
        super().__init__(name, electron_beam, magnetic_structure)

    def get_wavefront(self):
        raise NotImplementedError("This method should be specialized by specific implementors" +
                                  "\n\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)