# TODO: REMOVE THIS!!!!
try:
    from srwlib import *
    SRWLIB_AVAILABLE = True
except:
    try:
        from wpg.srwlib import *
        SRWLIB_AVAILABLE = True
    except:
        SRWLIB_AVAILABLE = False
        print("SRW is not available")

import numpy
import scipy.constants as codata
angstroms_to_eV = codata.h*codata.c/codata.e*1e10

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.decorators import WavefrontDecorator

class WOSRWWavefront(SRWLWfr, WavefrontDecorator):

    def __init__(self,
                 _arEx=None,
                 _arEy=None,
                 _typeE='f',
                 _eStart=0,
                 _eFin=0,
                 _ne=0,
                 _xStart=0,
                 _xFin=0,
                 _nx=0,
                 _yStart=0,
                 _yFin=0,
                 _ny=0,
                 _zStart=0,
                 _partBeam=None):
        SRWLWfr.__init__(self,
                         _arEx=_arEx,
                         _arEy=_arEy,
                         _typeE=_typeE,
                         _eStart=_eStart,
                         _eFin=_eFin,
                         _ne=_ne,
                         _xStart=_xStart,
                         _xFin=_xFin,
                         _nx=_nx,
                         _yStart=_yStart,
                         _yFin=_yFin,
                         _ny=_ny,
                         _zStart=_zStart,
                         _partBeam=_partBeam)
        if _eFin == 0 or _eStart == 0:
            raise ValueError("Energy has not set!")

        self._wavelength = angstroms_to_eV/((_eFin+_eStart)*0.5*1e10)

    def toGenericWavefront(self):
        wavefront = GenericWavefront2D.initialize_wavefront_from_range(self.mesh.xStart,
                                                                       self.mesh.xFin,
                                                                       self.mesh.yStart,
                                                                       self.mesh.yFin,
                                                                       number_of_points=(self.mesh.nx,self.mesh.ny),
                                                                       wavelength=self._wavelength)

        amplitude = SRWEFieldAsNumpy(self)
        amplitude = amplitude[0,:,:,0]
        wavefront.set_complex_amplitude(amplitude)

        return wavefront

    @classmethod
    def fromGenericWavefront(cls, wavefront):
        return SRWWavefrontFromElectricField(wavefront.get_coordinate_x()[0],
                                             wavefront.get_coordinate_x()[-1],
                                             wavefront.get_complex_amplitude(),
                                             wavefront.get_coordinate_y()[0],
                                             wavefront.get_coordinate_y()[-1],
                                             numpy.zeros_like(wavefront.get_complex_amplitude()),
                                             angstroms_to_eV/(wavefront.get_wavelength()*1e10),
                                             1.0,
                                             1.0,
                                             1e-3,
                                             1.0,
                                             1e-3)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ACCESSORIES
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

def SRWEFieldAsNumpy(swrwf):
    """
    Extracts electrical field from a SRWWavefront
    :param srw_wavefront: SRWWavefront to extract electrical field from.
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """

    dim_x = swrwf.mesh.nx
    dim_y = swrwf.mesh.ny
    number_energies = swrwf.mesh.ne

    x_polarization = SRWArrayToNumpy(swrwf.arEx, dim_x, dim_y, number_energies)
    y_polarization = SRWArrayToNumpy(swrwf.arEy, dim_x, dim_y, number_energies)

    e_field = numpy.concatenate((x_polarization,y_polarization), 3)

    return e_field

def SRWWavefrontFromElectricField(horizontal_start, horizontal_end, horizontal_efield,
                                  vertical_start, vertical_end, vertical_efield,
                                  energy, z, Rx, dRx, Ry, dRy):
    """
    Creates a SRWWavefront from pi and sigma components of the electrical field.
    :param horizontal_start: Horizontal start position of the grid in m
    :param horizontal_end: Horizontal end position of the grid in m
    :param horizontal_efield: The pi component of the complex electrical field
    :param vertical_start: Vertical start position of the grid in m
    :param vertical_end: Vertical end position of the grid in m
    :param vertical_efield: The sigma component of the complex electrical field
    :param energy: Energy in eV
    :param z: z position of the wavefront in m
    :param Rx: Instantaneous horizontal wavefront radius
    :param dRx: Error in instantaneous horizontal wavefront radius
    :param Ry: Instantaneous vertical wavefront radius
    :param dRy: Error in instantaneous vertical wavefront radius
    :return: A wavefront usable with SRW.
    """

    horizontal_size = horizontal_efield.shape[0]
    vertical_size = horizontal_efield.shape[1]

    if horizontal_size % 2 == 1 or \
       vertical_size % 2 == 1:
        # raise Exception("Both horizontal and vertical grid must have even number of points")
        print("NumpyToSRW: WARNING: Both horizontal and vertical grid must have even number of points")

    horizontal_field = numpyArrayToSRWArray(horizontal_efield)
    vertical_field = numpyArrayToSRWArray(vertical_efield)

    srwwf = WOSRWWavefront(_arEx=horizontal_field,
                           _arEy=vertical_field,
                           _typeE='f',
                           _eStart=energy,
                           _eFin=energy,
                           _ne=1,
                           _xStart=horizontal_start,
                           _xFin=horizontal_end,
                           _nx=horizontal_size,
                           _yStart=vertical_start,
                           _yFin=vertical_end,
                           _ny=vertical_size,
                           _zStart=z)

    srwwf.Rx = Rx
    srwwf.Ry = Ry
    srwwf.dRx = dRx
    srwwf.dRy = dRy

    return srwwf

def numpyArrayToSRWArray(numpy_array):
    """
    Converts a numpy.array to an array usable by SRW.
    :param numpy_array: a 2D numpy array
    :return: a 2D complex SRW array
    """
    elements_size = numpy_array.size

    r_horizontal_field = numpy_array[:, :].real.transpose().flatten().astype(numpy.float)
    i_horizontal_field = numpy_array[:, :].imag.transpose().flatten().astype(numpy.float)

    tmp = numpy.zeros(elements_size * 2, dtype=numpy.float32)
    for i in range(elements_size):
        tmp[2*i] = r_horizontal_field[i]
        tmp[2*i+1] = i_horizontal_field[i]

    return array('f', tmp)

def SRWArrayToNumpy(srw_array, dim_x, dim_y, number_energies):
    """
    Converts a SRW array to a numpy.array.
    :param srw_array: SRW array
    :param dim_x: size of horizontal dimension
    :param dim_y: size of vertical dimension
    :param number_energies: Size of energy dimension
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """
    re = numpy.array(srw_array[::2], dtype=numpy.float)
    im = numpy.array(srw_array[1::2], dtype=numpy.float)

    e = re + 1j * im
    e = e.reshape((dim_y,
                   dim_x,
                   number_energies,
                   1)
                  )

    e = e.swapaxes(0, 2)

    return e.copy()


