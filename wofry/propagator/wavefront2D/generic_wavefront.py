import numpy
import scipy.constants as codata

from srxraylib.util.data_structures import ScaledMatrix, ScaledArray

from wofry.propagator.wavefront import Wavefront, WavefrontDimension
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

from wofry.propagator.util.gaussian_schell_model import GaussianSchellModel2D


import copy

# needed for h5 i/o
import os
import sys
import time
try:
    import h5py
except:
    raise ImportError("h5py not available: input/output to files not working")
# --------------------------------------------------
# Wavefront 2D
# --------------------------------------------------




class GenericWavefront2D(Wavefront):
    XX = 0
    YY = 1


    def __init__(self, wavelength=1e-10, electric_field_matrix=None):
        self._wavelength = wavelength
        self._electric_field_matrix = electric_field_matrix

    def get_dimension(self):
        return WavefrontDimension.TWO


    def duplicate(self):
        return GenericWavefront2D(wavelength=self._wavelength,
                                  electric_field_matrix=ScaledMatrix(x_coord=copy.copy(self._electric_field_matrix.x_coord),
                                                                     y_coord=copy.copy(self._electric_field_matrix.y_coord),
                                                                     z_values=copy.copy(self._electric_field_matrix.z_values),
                                                                     interpolator=self._electric_field_matrix.interpolator))

    @classmethod
    def initialize_wavefront(cls, number_of_points=(100,100), wavelength=1e-10):
        return GenericWavefront2D(wavelength,
                                  ScaledMatrix.initialize(np_array_z=numpy.full(number_of_points, (1.0 + 0.0j),
                                                                              dtype=complex),
                                                          interpolator=False))

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=0.0, x_step=0.0, y_start=0.0, y_step=0.0,
                                        number_of_points=(100,100), wavelength=1e-10):
        sM = ScaledMatrix.initialize_from_steps(numpy.full(number_of_points,(1.0 + 0.0j),
                                                           dtype=complex),
                                                x_start,
                                                x_step,
                                                y_start,
                                                y_step,
                                                interpolator=False)

        return GenericWavefront2D(wavelength,sM)

    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0,
                                        number_of_points=(100,100), wavelength=1e-10 ):


        return GenericWavefront2D(wavelength, ScaledMatrix.initialize_from_range( \
                    numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                    x_min,x_max,y_min,y_max,interpolator=False))

    @classmethod
    def initialize_wavefront_from_arrays(cls,x_array, y_array,  z_array, wavelength=1e-10):
        sh = z_array.shape

        if sh[0] != x_array.size:
            raise Exception("Unmatched shapes for x")
        
        if sh[1] != y_array.size:
            raise Exception("Unmatched shapes for y")

        sM = ScaledMatrix.initialize_from_steps(
                    z_array,x_array[0],numpy.abs(x_array[1]-x_array[0]),
                            y_array[0],numpy.abs(y_array[1]-y_array[0]),interpolator=False)

        return GenericWavefront2D(wavelength, sM)


    def get_Wavefront1D_from_profile(self, axis, coordinate):
        # swap axis - changed giovanni+manuel
        if axis == 1: # fixed X
            index = numpy.argmin(numpy.abs(self._electric_field_matrix.x_coord - coordinate))

            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(scale=self._electric_field_matrix.y_coord,
                                                                       np_array=self._electric_field_matrix.z_values[index, :]))
        elif axis == 0:
            index = numpy.argmin(numpy.abs(self._electric_field_matrix.y_coord - coordinate))

            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(scale=self._electric_field_matrix.x_coord,
                                                                       np_array=self._electric_field_matrix.z_values[:, index]))

    #TODO
    def get_Wavefront1D_from_histogram(self, axis):
        raise NotImplementedError("Not yet implemented!")


    @classmethod
    def combine_1D_wavefronts_into_2D(cls, wavefront_h, wavefront_v, normalize_to=0, wavelength=0.0):
        if not wavelength > 0.0:
            wavelength = (wavefront_h.get_wavelength() + wavefront_v.get_wavelength())/2

        if normalize_to == 0:
            normalization_factor = numpy.sqrt(numpy.sum(wavefront_h.get_intensity()))
        elif normalize_to == 1:
            normalization_factor = numpy.sqrt(numpy.sum(wavefront_v.get_intensity()))
        else:
            normalization_factor = 1.0

        wavefront_2D = GenericWavefront2D.initialize_wavefront_from_steps(x_start=wavefront_h.offset(),
                                                                          x_step=wavefront_h.delta(),
                                                                          y_start=wavefront_v.offset(),
                                                                          y_step=wavefront_v.delta(),
                                                                          number_of_points=(wavefront_h.size(), wavefront_v.size()))

        # complex_amplitude =  numpy.zeros((wavefront_h.size(), wavefront_v.size()), dtype=complex)
        #
        # for i in range (0, wavefront_h.size()):
        #     for j in range (0, wavefront_v.size()):
        #         complex_amplitude[i, j] = complex(wavefront_h.get_amplitude()[i]*wavefront_v.get_amplitude()[j],
        #                                           wavefront_h.get_phase()[i] + wavefront_v.get_phase()[j])

        complex_amplitude = numpy.outer(wavefront_h.get_complex_amplitude(), wavefront_v.get_complex_amplitude())

        normalization_factor /= numpy.sum(numpy.abs(complex_amplitude))

        wavefront_2D.set_complex_amplitude(complex_amplitude * normalization_factor)

        return wavefront_2D

    # main parameters

    def size(self):
        return self._electric_field_matrix.shape()

    def delta(self):
        x = self.get_coordinate_x()
        y = self.get_coordinate_y()
        return numpy.abs(x[1]-x[0]),numpy.abs(y[1]-y[0])
    #
    def offset(self):
        return self.get_coordinate_x()[0],self.get_coordinate_y()[0]

    def get_wavelength(self):
        return self._wavelength

    def get_wavenumber(self):
        return 2*numpy.pi/self._wavelength

    def get_photon_energy(self):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        return  m2ev / self._wavelength

    def get_coordinate_x(self):
        return self._electric_field_matrix.get_x_values()

    def get_coordinate_y(self):
        return self._electric_field_matrix.get_y_values()

    def get_complex_amplitude(self):
        return self._electric_field_matrix.get_z_values()

    def get_amplitude(self):
        return numpy.absolute(self.get_complex_amplitude())

    def get_phase(self,from_minimum_intensity=0.0,unwrap=0):
        """

        :param from_minimum_intensity: set to zero phase values at pixels where intensity
                                        is less than from_minimum_intensity threshold
        :param unwrap: Flag to unwrap the returned phase:
            0: No unwrap (default)
            1: Unwrap only in Horizontal axis.
            2: Unwrap only in Vertical axis.
            3: Unwrap first in H, then in V.
            4: Unwrap first in V, then in H.
        :return: the phase in a numpy array
        """
        # return numpy.arctan2(numpy.imag(self.get_complex_amplitude()), numpy.real(self.get_complex_amplitude()))
        phase = numpy.angle( self.get_complex_amplitude() )

        if (from_minimum_intensity > 0.0):
            intensity = self.get_intensity()
            intensity /= intensity.max()
            bad_indices = numpy.where(intensity < from_minimum_intensity )
            phase[bad_indices] = 0.0

        if unwrap > 0:
            if unwrap == 1: # x only
                phase = numpy.unwrap(phase,axis=0)
            elif unwrap == 2: # y only
                phase = numpy.unwrap(phase,axis=1)
            elif unwrap == 3: # x and y
                phase = numpy.unwrap(numpy.unwrap(phase,axis=0),axis=1)
            elif unwrap == 4: # y and x
                phase = numpy.unwrap(numpy.unwrap(phase,axis=1),axis=0)
            else:
                raise Exception(NotImplemented)

        return phase

    def get_intensity(self):
        return self.get_amplitude()**2

    def get_mesh_indices_x(self):
        return numpy.outer( numpy.arange(0,self.size()[0]), numpy.ones(self.size()[1]))

    def get_mesh_indices_y(self):
        return numpy.outer( numpy.ones(self.size()[0]), numpy.arange(0,self.size()[1]))

    def get_mask_grid(self,width_in_pixels=(1,1),number_of_lines=(1,1)):
        """

        :param width_in_pixels: (pixels_for_horizontal_lines,pixels_for_vertical_lines
        :param number_of_lines: (number_of_horizontal_lines, number_of_vertical_lines)
        :return:
        """

        indices_x = self.get_mesh_indices_x()
        indices_y = self.get_mesh_indices_y()

        used_indices = numpy.zeros( self.size(),dtype=int)

        for i in range(number_of_lines[1]):
            used_indices[ numpy.where( numpy.abs(indices_x - (i+1)*self.size()[0]/(1+number_of_lines[1])) <= (width_in_pixels[1]-1) )] = 1
        for i in range(number_of_lines[0]):
            used_indices[ numpy.where( numpy.abs(indices_y - (i+1)*self.size()[1]/(1+number_of_lines[0])) <= (width_in_pixels[0]-1) )] = 1

        return used_indices

    # interpolated values (a bit redundant, but kept the same interfacs as wavefront 1D)

    def get_interpolated(self,x_value,y_value,toreturn='complex_amplitude'):
        interpolated_values = self._electric_field_matrix.interpolate_value(x_value,y_value)
        if toreturn == 'complex_amplitude':
            return interpolated_values
        elif toreturn == 'amplitude':
            return numpy.abs(interpolated_values)
        elif toreturn == 'phase':
            return numpy.arctan2(numpy.imag(interpolated_values), numpy.real(interpolated_values))
        elif toreturn == 'intensity':
            return numpy.abs(interpolated_values)**2
        else:
            raise Exception('Unknown return string')

    def get_interpolated_complex_amplitude(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='complex_amplitude')

    def get_interpolated_complex_amplitudes(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='complex_amplitude')

    def get_interpolated_amplitude(self, x_value,y_value): # singular!
        return self.get_interpolated(x_value,y_value,toreturn='amplitude')

    def get_interpolated_amplitudes(self, x_value,y_value): # plural!
        return self.get_interpolated(x_value,y_value,toreturn='amplitude')
    #
    def get_interpolated_phase(self, x_value,y_value): # singular!
        return self.get_interpolated(x_value,y_value,toreturn='phase')

    def get_interpolated_phases(self, x_value,y_value): # plural!
        return self.get_interpolated(x_value,y_value,toreturn='phase')

    def get_interpolated_intensity(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='intensity')

    def get_interpolated_intensities(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='intensity')

    # only for 2D
    def get_mesh_x(self):
        XY = numpy.meshgrid(self.get_coordinate_x(),self.get_coordinate_y())
        return XY[0].T

    def get_mesh_y(self):
        XY = numpy.meshgrid(self.get_coordinate_x(),self.get_coordinate_y())
        return XY[1].T

    # modifiers

    def set_wavelength(self,wavelength):
        self._wavelength = wavelength

    def set_wavenumber(self,wavenumber):
        self._wavelength = 2*numpy.pi / wavenumber

    def set_photon_energy(self,photon_energy):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        self._wavelength = m2ev / photon_energy

    def set_complex_amplitude(self,complex_amplitude):
        if self._electric_field_matrix.shape() != complex_amplitude.shape:
            raise Exception("Incompatible shape")
        self._electric_field_matrix.set_z_values(complex_amplitude)

    # TODO: add inclination like for 1D
    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= 0.0
        new_value += complex_amplitude
        self._electric_field_matrix.set_z_values(new_value)

    # TODO: add inclination like for 1D
    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))

    # TODO: add center like for 1D
    def set_spherical_wave(self,  radius=1.0, complex_amplitude=1.0,):
        """

        :param radius:  Positive radius is divergent wavefront, negative radius is convergent
        :param complex_amplitude:
        :return:
        """
        if radius == 0:
            raise Exception("Radius cannot be zero")
        new_value = complex_amplitude * numpy.exp( -1.0j * self.get_wavenumber() *
                                (self.get_mesh_x()**2 + self.get_mesh_y()**2) / (-2*radius) )

        # new_value = numpy.exp(-1.0j * self.get_wavenumber() *
        #                         (self.get_mesh_x()**2+self.get_mesh_y()**2)/(-2*radius))
        self._electric_field_matrix.set_z_values(new_value)

    def set_gaussian_hermite_mode(self, sigma_x, sigma_y, nx, ny, amplitude=1.0, center_x=0.0, center_y=0.0):
        x = self.get_coordinate_x()
        y = self.get_coordinate_y()

        a2D = GaussianSchellModel2D(amplitude, sigma_x, 100.0*sigma_x, sigma_y, 100.0*sigma_y)
        Phi = a2D.phi_nm(nx, ny, x-center_x, y-center_y) + 0j

        self.set_complex_amplitude(Phi)

    # note that amplitude is for "amplitude" not for intensity!
    def set_gaussian(self, sigma_x, sigma_y, amplitude=1.0, center_x=0.0, center_y=0.0):
        self.set_gaussian_hermite_mode(sigma_x, sigma_y, 0, 0, amplitude=amplitude, center_x=center_x, center_y=center_y)


    def add_phase_shift(self, phase_shift):
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= numpy.exp(1.0j*phase_shift)
        self._electric_field_matrix.set_z_values(new_value)

    def add_phase_shifts(self, phase_shifts):
        if phase_shifts.shape != self._electric_field_matrix.shape():
            raise Exception("Phase Shifts array has different dimension")
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= numpy.exp(1.0j*phase_shifts)
        self._electric_field_matrix.set_z_values(new_value)

    def rescale_amplitude(self, factor):
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= factor
        self._electric_field_matrix.set_z_values(new_value)

    def rescale_amplitudes(self, factors):
        if factors.shape != self._electric_field_matrix.shape():
            raise Exception("Factors array has different dimension")
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= factors
        self._electric_field_matrix.set_z_values(new_value)

    def rebin(self,expansion_points_horizontal, expansion_points_vertical, expansion_range_horizontal, expansion_range_vertical,
              keep_the_same_intensity=0,set_extrapolation_to_zero=0):

        x0 = self.get_coordinate_x()
        y0 = self.get_coordinate_y()

        x1 = numpy.linspace(x0[0]*expansion_range_horizontal,x0[-1]*expansion_range_horizontal,int(x0.size*expansion_points_horizontal))
        y1 = numpy.linspace(y0[0]*expansion_range_vertical  ,y0[-1]*expansion_range_vertical,  int(y0.size*expansion_points_vertical))

        X1 = numpy.outer(x1,numpy.ones_like(y1))
        Y1 = numpy.outer(numpy.ones_like(x1),y1)
        z1 = self.get_interpolated_complex_amplitudes(X1,Y1)

        if set_extrapolation_to_zero:
            z1[numpy.where( X1 < x0[0])] = 0.0
            z1[numpy.where( X1 > x0[-1])] = 0.0
            z1[numpy.where( Y1 < y0[0])] = 0.0
            z1[numpy.where( Y1 > y0[-1])] = 0.0


        if keep_the_same_intensity:
            z1 /= numpy.sqrt( (numpy.abs(z1)**2).sum() / self.get_intensity().sum() )

        new_wf = GenericWavefront2D.initialize_wavefront_from_arrays(x1,y1,z1,wavelength=self.get_wavelength())

        return new_wf

    def clip_square(self, x_min, x_max, y_min, y_max, negative=False):


        if not negative:
            window = numpy.ones(self._electric_field_matrix.shape())
            lower_window_x = numpy.where(self.get_coordinate_x() < x_min)
            upper_window_x = numpy.where(self.get_coordinate_x() > x_max)
            lower_window_y = numpy.where(self.get_coordinate_y() < y_min)
            upper_window_y = numpy.where(self.get_coordinate_y() > y_max)

            if len(lower_window_x) > 0: window[lower_window_x,:] = 0
            if len(upper_window_x) > 0: window[upper_window_x,:] = 0
            if len(lower_window_y) > 0: window[:,lower_window_y] = 0
            if len(upper_window_y) > 0: window[:,upper_window_y] = 0
        else:
            window = numpy.ones(self._electric_field_matrix.shape())
            window2 = numpy.ones_like(window)
            window_x = numpy.where( (x_min <= self.get_coordinate_x()) & ( self.get_coordinate_x() <= x_max))
            window_y = numpy.where( (y_min <= self.get_coordinate_y()) & ( self.get_coordinate_y() <= y_max))

            if len(window_x) > 0: window[window_x,:] = 0.0
            if len(window_y) > 0: window2[:,window_y] = 0.0

            window += window2
            window_good = numpy.where( window > 0 )
            if len(window_good) > 0: window[window_good] = 1.0

        self.rescale_amplitudes(window)

    # new
    def clip_circle(self, radius, x_center=0.0, y_center=0.0, negative=False):
        window = numpy.zeros(self._electric_field_matrix.shape())
        X = self.get_mesh_x()
        Y = self.get_mesh_y()
        distance_to_center = numpy.sqrt( (X-x_center)**2 + (Y-y_center)**2 )
        if negative:
            indices_good = numpy.where(distance_to_center >= radius)
        else:
            indices_good = numpy.where(distance_to_center <= radius)
        window[indices_good] = 1.0

        self.rescale_amplitudes(window)

    def is_identical(self,wfr,decimal=7):
        from numpy.testing import assert_array_almost_equal
        try:
            assert_array_almost_equal(self.get_complex_amplitude(),wfr.get_complex_amplitude(),decimal)
            assert_array_almost_equal(self.get_coordinate_x(),wfr.get_coordinate_x(),decimal)
            assert_array_almost_equal(self.get_coordinate_y(),wfr.get_coordinate_y(),decimal)
            assert_array_almost_equal(self.get_photon_energy(),wfr.get_photon_energy(),decimal)
        except:
            return False

        return True

    # auxiliary function to dump h5 files
    def _dump_arr_2_hdf5(self,_arr,_calculation, _filename, _subgroupname):
        """
        Auxiliary routine to save_h5_file
        :param _arr: (usually 2D) array to be saved on the hdf5 file inside the _subgroupname
        :param _calculation
        :param _filename: path to file for saving the wavefront
        :param _subgroupname: container mechanism by which HDF5 files are organised
        """
        sys.stdout.flush()
        f = h5py.File(_filename, 'a')
        try:
            f1 = f.create_group(_subgroupname)
        except:
            f1 = f[_subgroupname]
        fdata = f1.create_dataset(_calculation, data=_arr)
        f.close()

    def save_h5_file(self,filename,subgroupname="wfr",intensity=False,phase=False,overwrite=True,verbose=False):
        """
        Auxiliary function to write wavefront data into a hdf5 generic file.
        When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
        issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
        :param self: input / output resulting Wavefront structure (instance of GenericWavefront2D);
        :param filename: path to file for saving the wavefront
        :param subgroupname: container mechanism by which HDF5 files are organised
        :param intensity: writes intensity for sigma and pi polarisation (default=False)
        :param amplitude:
        :param phase:
        :param overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
        :param verbose: if True, print some file i/o messages
        """
        try:
            if not os.path.isfile(filename):  # if file doesn't exist, create it.
                sys.stdout.flush()
                f = h5py.File(filename, 'w')
                # point to the default data to be plotted
                f.attrs['default']          = 'entry'
                # give the HDF5 root some more attributes
                f.attrs['file_name']        = filename
                f.attrs['file_time']        = time.time()
                f.attrs['creator']          = 'save_wofry_wavefront_to_hdf5'
                f.attrs['HDF5_Version']     = h5py.version.hdf5_version
                f.attrs['h5py_version']     = h5py.version.version
                f.close()

            # always writes complex amplitude
            x_polarization = self.get_complex_amplitude()       # sigma
            # TODO: implement polarization
            # y_polarization = self.get_complex_amplitude()*0.0   # pi

            self._dump_arr_2_hdf5(x_polarization.T, "wfr_complex_amplitude_s", filename, subgroupname)
            # self._dump_arr_2_hdf5(y_polarization.T, "wfr_complex_amplitude_p", filename, subgroupname)


            if intensity:
                self._dump_arr_2_hdf5(self.get_intensity().T,"intensity/wfr_intensity", filename, subgroupname)

            if phase:
                self._dump_arr_2_hdf5(self.get_phase().T,"phase/wfr_phase", filename, subgroupname)

            # add mesh and SRW information
            f = h5py.File(filename, 'a')
            f1 = f[subgroupname]

            # point to the default data to be plotted
            f1.attrs['NX_class'] = 'NXentry'
            f1.attrs['default'] = 'intensity'

            # TODO: add self interpreting decoder
            # f1["wfr_method"] = "WOFRY"
            f1["wfr_dimension"] = 2
            f1["wfr_photon_energy"] = self.get_photon_energy()
            x = self.get_coordinate_x()
            y = self.get_coordinate_y()

            f1["wfr_mesh_X"] =  numpy.array([x[0],x[-1],x.size])
            f1["wfr_mesh_Y"] =  numpy.array([y[0],y[-1],y.size])

            # Add NX plot attribites for automatic plot with silx view
            myflags = [intensity,phase]
            mylabels = ['intensity','phase']
            for i,label in enumerate(mylabels):
                if myflags[i]:
                    f2 = f1[mylabels[i]]
                    f2.attrs['NX_class'] = 'NXdata'
                    f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
                    f2.attrs['axes'] = [b'axis_y', b'axis_x']

                    f3 = f2["wfr_%s"%(mylabels[i])]
                    f3.attrs['interpretation'] = 'image'

                    # X axis data
                    ds = f2.create_dataset('axis_y', data=1e6*y)
                    ds.attrs['units'] = 'microns'
                    ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label

                    # Y axis data
                    ds = f2.create_dataset('axis_x', data=1e6*x)
                    ds.attrs['units'] = 'microns'
                    ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
            f.close()

        except:
            # TODO: check exit??
            if overwrite is not True:
                raise Exception("Bad input argument")
            os.remove(filename)
            if verbose: print("save_h5_file: file deleted %s"%filename)
            self.save_h5_file(filename,subgroupname, intensity=intensity, phase=phase, overwrite=False)

        if verbose: print("save_h5_file: witten/updated %s data in file: %s"%(subgroupname,filename))

    @classmethod
    def load_h5_file(cls,filename,filepath):

        try:
            f = h5py.File(filename, 'r')
            mesh_X = f[filepath+"/wfr_mesh_X"].value
            mesh_Y = f[filepath+"/wfr_mesh_Y"].value
            complex_amplitude_s = f[filepath+"/wfr_complex_amplitude_s"].value.T
            wfr = cls.initialize_wavefront_from_arrays(
                                x_array=numpy.linspace(mesh_X[0],mesh_X[1],mesh_X[2]),
                                y_array=numpy.linspace(mesh_Y[0],mesh_Y[1],mesh_Y[2]),
                                z_array=complex_amplitude_s)
            wfr.set_photon_energy(f[filepath+"/wfr_photon_energy"].value)
            f.close()
            return wfr
        except:
            raise Exception("Failed to load 2D wavefront to h5 file: "+filename)
