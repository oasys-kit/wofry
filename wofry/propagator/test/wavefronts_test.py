import unittest
import numpy
import os

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens, WOIdealLens1D

from wofry.propagator.polarization import Polarization

do_plot = False

#
# 1D tests
#
class GenericWavefront1DTest(unittest.TestCase):
    def test_initializers(self,do_plot=do_plot):

        print("#                                                             ")
        print("# Tests for initializars (1D)                                 ")
        print("#                                                             ")

        x = numpy.linspace(-100,100,50)
        y = numpy.abs(x)**1.5 +  1j*numpy.abs(x)**1.8



        wf0 = GenericWavefront1D.initialize_wavefront_from_steps(x[0],numpy.abs(x[1]-x[0]),y.size)
        wf0.set_complex_amplitude(y)

        wf1 = GenericWavefront1D.initialize_wavefront_from_range(x[0],x[-1],y.size)
        wf1.set_complex_amplitude(y)

        wf2 = GenericWavefront1D.initialize_wavefront_from_arrays(x,y)

        print("wavefront sizes: ",wf1.size(),wf1.size(),wf2.size())

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf0.get_abscissas(),wf0.get_intensity(),
                       title="initialize_wavefront_from_steps",show=0)
            plot(wf1.get_abscissas(),wf1.get_intensity(),
                       title="initialize_wavefront_from_range",show=0)
            plot(wf2.get_abscissas(),wf2.get_intensity(),
                       title="initialize_wavefront_from_arrays",show=1)

        numpy.testing.assert_almost_equal(wf0.get_intensity(),numpy.abs(y)**2,5)
        numpy.testing.assert_almost_equal(wf1.get_intensity(),numpy.abs(y)**2,5)

        numpy.testing.assert_almost_equal(x,wf1.get_abscissas(),11)
        numpy.testing.assert_almost_equal(x,wf2.get_abscissas(),11)

    def test_plane_wave(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 1D plane wave                                   ")
        print("#                                                             ")

        wavelength        = 1.24e-10

        wavefront_length_x = 400e-6

        npixels_x =  1024

        wavefront_x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)


        wavefront = GenericWavefront1D.initialize_wavefront_from_steps(
                        x_start=wavefront_x[0],x_step=numpy.abs(wavefront_x[1]-wavefront_x[0]),
                        number_of_points=npixels_x,wavelength=wavelength)

        numpy.testing.assert_almost_equal(wavefront_x,wavefront.get_abscissas(),9)

        # possible modifications

        wavefront.set_plane_wave_from_amplitude_and_phase(5.0,numpy.pi/2)
        numpy.testing.assert_almost_equal(wavefront.get_intensity(),25,5)

        wavefront.set_plane_wave_from_complex_amplitude(2.0+3j)
        numpy.testing.assert_almost_equal(wavefront.get_intensity(),13,5)

        phase_before = wavefront.get_phase()
        wavefront.add_phase_shift(numpy.pi/2)
        phase_after = wavefront.get_phase()
        numpy.testing.assert_almost_equal(phase_before+numpy.pi/2,phase_after,5)

        intensity_before = wavefront.get_intensity()
        wavefront.rescale_amplitude(10.0)
        intensity_after = wavefront.get_intensity()
        numpy.testing.assert_almost_equal(intensity_before*100,intensity_after,5)

        # interpolation

        wavefront.set_plane_wave_from_complex_amplitude(2.0+3j)
        test_value1 = wavefront.get_interpolated_complex_amplitude(0.01)
        self.assertAlmostEqual( (2.0+3j).real, test_value1.real, 5)
        self.assertAlmostEqual( (2.0+3j).imag, test_value1.imag, 5)


        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wavefront.get_abscissas(),wavefront.get_intensity(),title="Intensity (plane wave)",show=0)
            plot(wavefront.get_abscissas(),wavefront.get_phase(),title="Phase (plane wave)",show=1)


    def test_spherical_wave(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 1D spherical wave                               ")
        print("#                                                             ")

        wavelength        = 1.24e-10

        wavefront_length_x = 400e-6

        npixels_x =  1024

        wavefront_x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)



        wf1 = GenericWavefront1D.initialize_wavefront_from_steps(
                        x_start=wavefront_x[0],x_step=numpy.abs(wavefront_x[1]-wavefront_x[0]),
                        number_of_points=npixels_x,wavelength=wavelength)

        wf2 = GenericWavefront1D.initialize_wavefront_from_steps(
                        x_start=wavefront_x[0],x_step=numpy.abs(wavefront_x[1]-wavefront_x[0]),
                        number_of_points=npixels_x,wavelength=wavelength)

        # an spherical wavefront is obtained 1) by creation, 2) focusing a planewave

        wf1.set_spherical_wave(radius=-5.0, complex_amplitude=3+0j)
        wf1.clip(-50e-6,10e-6)

        wf2.set_plane_wave_from_complex_amplitude(3+0j)
        ideal_lens = WOIdealLens1D("test", 5.0)
        ideal_lens.applyOpticalElement(wf2)
        wf2.clip(-50e-6,10e-6)



        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_abscissas(),wf1.get_phase(),title="Phase of spherical wavefront",show=0)
            plot(wf2.get_abscissas(),wf2.get_phase(),title="Phase of focused plane wavefront",show=0)
            plot(wf1.get_abscissas(),wf1.get_phase(from_minimum_intensity=0.1),title="Phase of spherical wavefront (for intensity > 0.1)",show=0)
            plot(wf2.get_abscissas(),wf2.get_phase(from_minimum_intensity=0.1),title="Phase of focused plane wavefront (for intensity > 0.1)",show=1)


        numpy.testing.assert_almost_equal(wf1.get_phase(),wf2.get_phase(),5)

    def test_gaussianhermite_mode(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 1D Gaussian Hermite mode                        ")
        print("#                                                             ")

        wavelength        = 1.24e-10


        # 2D
        sigma_x = 100e-6
        mode_x = 0
        npixels_x = 100


        wavefront_length_x = 10*sigma_x

        x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)


        wf1 = GenericWavefront1D.initialize_wavefront_from_steps(
                        x_start=x[0],x_step=numpy.abs(x[1]-x[0]),
                        number_of_points=npixels_x,wavelength=wavelength)


        wf1.set_gaussian_hermite_mode(sigma_x, mode_x, amplitude=1.0)


        numpy.testing.assert_almost_equal(wf1.get_amplitude()[30],23.9419082194,5)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_abscissas(),wf1.get_amplitude(),title="Amplitude of gaussianhermite",show=0)


    def test_interpolator(self,do_plot=do_plot):
        #
        # interpolator
        #
        print("#                                                             ")
        print("# Tests for 1D interpolator                                   ")
        print("#                                                             ")

        x = numpy.linspace(-10,10,100)

        sigma = 3.0
        Z = numpy.exp(-1.0*x**2/2/sigma**2)

        print("shape of Z",Z.shape)

        wf = GenericWavefront1D.initialize_wavefront_from_steps(x[0],numpy.abs(x[1]-x[0]),number_of_points=100)
        print("wf shape: ",wf.size())
        wf.set_complex_amplitude( Z )

        x1 = 3.2
        z1 = numpy.exp(x1**2/-2/sigma**2)
        print("complex ampl at (%g): %g+%gi (exact=%g)"%(x1,
                                                        wf.get_interpolated_complex_amplitude(x1).real,
                                                        wf.get_interpolated_complex_amplitude(x1).imag,
                                                        z1))
        self.assertAlmostEqual(wf.get_interpolated_complex_amplitude(x1).real,z1,4)

        print("intensity  at (%g):   %g (exact=%g)"%(x1,wf.get_interpolated_intensity(x1),z1**2))
        self.assertAlmostEqual(wf.get_interpolated_intensity(x1),z1**2,4)


        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf.get_abscissas(),wf.get_intensity(),title="Original",show=1)
            xx = wf.get_abscissas()
            yy = wf.get_interpolated_intensities(wf.get_abscissas()-1e-5)
            plot(xx,yy,title="interpolated on same grid",show=1)


    def test_save_load_h5_file(self):
        wfr = GenericWavefront1D.initialize_wavefront_from_range(-2.0,2.0,number_of_points=100)
        wfr.set_gaussian(.2,amplitude=5+8j)
        print("Saving 1D wavefront to file: tmp.h5")
        wfr.save_h5_file("tmp.h5","wfr1",intensity=True,phase=True)
        print("Reading 1D wavefront from file: tmp.h5")
        wfr2  = GenericWavefront1D.load_h5_file("tmp.h5","wfr1")
        print("Cleaning file tmp.h5")
        os.remove("tmp.h5")
        assert(wfr2.is_identical(wfr))

    def test_guess_wavefront_curvature(self):
        print("#                                                             ")
        print("# Tests guessing wavefront curvature                          ")
        print("#                                                             ")
        #
        # create source
        #
        wavefront = GenericWavefront1D.initialize_wavefront_from_range(x_min=-0.5e-3,
                                                                       x_max=0.5e-3,
                                                                       number_of_points=2048,
                                                                       wavelength=1.5e-10)
        radius = -50.0 # 50.
        wavefront.set_spherical_wave(radius=radius)
        if do_plot:
            from srxraylib.plot.gol import plot
            radii,fig_of_mer = wavefront.scan_wavefront_curvature(rmin=-1000,rmax=1000,rpoints=100)
            plot(radii,fig_of_mer)
        guess_radius = wavefront.guess_wavefront_curvature(rmin=-1000,rmax=1000,rpoints=100)
        assert(numpy.abs(radius - guess_radius) < 1e-3)

    def test_polarization(self):
        print("#                                                             ")
        print("# Tests polarization (1D)                                     ")
        print("#                                                             ")
        wavefront = GenericWavefront1D.initialize_wavefront_from_range(x_min=-0.5e-3,
                                                                       x_max=0.5e-3,
                                                                       number_of_points=2048,
                                                                       wavelength=1.5e-10,
                                                                       polarization=Polarization.TOTAL)

        ca = numpy.zeros(wavefront.size())
        wavefront.set_complex_amplitude(ca+(1+0j),ca+(0+1j))

        numpy.testing.assert_almost_equal(wavefront.get_interpolated_phase(0.1e-3,polarization=Polarization.SIGMA),0.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_phase(-0.1e-3,polarization=Polarization.PI),numpy.pi/2 )

        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,polarization=Polarization.TOTAL),2.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,polarization=Polarization.SIGMA),1.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,polarization=Polarization.PI),1.0 )

        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.SIGMA),(ca+1)**2 )
        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.PI),   (ca+1)**2 )
        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.TOTAL),2*(ca+1)**2 )
#
# 2D tests
#

class GenericWavefront2DTest(unittest.TestCase):

    def test_initializers(self,do_plot=do_plot):

        print("#                                                             ")
        print("# Tests for initializars (2D)                                 ")
        print("#                                                             ")

        x = numpy.linspace(-100,100,50)
        y = numpy.linspace(-50,50,200)
        XY = numpy.meshgrid(x,y)
        X = XY[0].T
        Y = XY[1].T
        sigma = 10
        Z = numpy.exp(- (X**2 + Y**2)/2/sigma**2) * 1j
        print("Shapes x,y,z: ",x.shape,y.shape,Z.shape)

        wf0 = GenericWavefront2D.initialize_wavefront_from_steps(x[0],numpy.abs(x[1]-x[0]),y[0],numpy.abs(y[1]-y[0]),number_of_points=Z.shape)
        wf0.set_complex_amplitude(Z)

        wf1 = GenericWavefront2D.initialize_wavefront_from_range(x[0],x[-1],y[0],y[-1],number_of_points=Z.shape)
        wf1.set_complex_amplitude(Z)

        wf2 = GenericWavefront2D.initialize_wavefront_from_arrays(x,y,Z)

        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf0.get_intensity(),wf0.get_coordinate_x(),wf0.get_coordinate_y(),
                       title="initialize_wavefront_from_steps",show=0)
            plot_image(wf1.get_intensity(),wf1.get_coordinate_x(),wf1.get_coordinate_y(),
                       title="initialize_wavefront_from_range",show=0)
            plot_image(wf2.get_intensity(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),
                       title="initialize_wavefront_from_arrays",show=1)


        numpy.testing.assert_almost_equal(numpy.abs(Z)**2,wf0.get_intensity(),11)
        numpy.testing.assert_almost_equal(numpy.abs(Z)**2,wf1.get_intensity(),11)
        numpy.testing.assert_almost_equal(numpy.abs(Z)**2,wf2.get_intensity(),11)

        numpy.testing.assert_almost_equal(x,wf0.get_coordinate_x(),11)
        numpy.testing.assert_almost_equal(x,wf1.get_coordinate_x(),11)
        numpy.testing.assert_almost_equal(x,wf2.get_coordinate_x(),11)

        numpy.testing.assert_almost_equal(y,wf0.get_coordinate_y(),11)
        numpy.testing.assert_almost_equal(y,wf1.get_coordinate_y(),11)
        numpy.testing.assert_almost_equal(y,wf2.get_coordinate_y(),11)



    def test_plane_wave(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 2D plane wave                                      ")
        print("#                                                             ")

        wavelength        = 1.24e-10

        wavefront_length_x = 400e-6
        wavefront_length_y = wavefront_length_x

        npixels_x =  1024
        npixels_y =  npixels_x

        x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)
        y = numpy.linspace(-0.5*wavefront_length_y,0.5*wavefront_length_y,npixels_y)

        wavefront = GenericWavefront2D.initialize_wavefront_from_steps(
                        x_start=x[0],x_step=numpy.abs(x[1]-x[0]),
                        y_start=y[0],y_step=numpy.abs(y[1]-y[0]),
                        number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        # possible modifications

        wavefront.set_plane_wave_from_amplitude_and_phase(5.0,numpy.pi/2)
        numpy.testing.assert_almost_equal(wavefront.get_intensity(),25,5)

        wavefront.set_plane_wave_from_complex_amplitude(2.0+3j)
        numpy.testing.assert_almost_equal(wavefront.get_intensity(),13,5)

        phase_before = wavefront.get_phase()
        wavefront.add_phase_shift(numpy.pi/2)
        phase_after = wavefront.get_phase()
        numpy.testing.assert_almost_equal(phase_before+numpy.pi/2,phase_after,5)

        intensity_before = wavefront.get_intensity()
        wavefront.rescale_amplitude(10.0)
        intensity_after = wavefront.get_intensity()
        numpy.testing.assert_almost_equal(intensity_before*100,intensity_after,5)

        # interpolation

        wavefront.set_plane_wave_from_complex_amplitude(2.0+3j)
        test_value1 = wavefront.get_interpolated_complex_amplitude(0.01,1.3)
        self.assertAlmostEqual( (2.0+3j).real, test_value1.real, 5)
        self.assertAlmostEqual( (2.0+3j).imag, test_value1.imag, 5)


        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wavefront.get_intensity(),wavefront.get_coordinate_x(),wavefront.get_coordinate_y(),
                       title="Intensity (plane wave)",show=0)
            plot_image(wavefront.get_phase(),wavefront.get_coordinate_x(),wavefront.get_coordinate_y(),
                       title="Phase (plane wave)",show=1)



    def test_spherical_wave(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 2D spherical wave                               ")
        print("#                                                             ")

        wavelength        = 1.24e-10

        wavefront_length_x = 400e-6
        wavefront_length_y = wavefront_length_x

        npixels_x =  1024
        npixels_y =  npixels_x

        x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)
        y = numpy.linspace(-0.5*wavefront_length_y,0.5*wavefront_length_y,npixels_y)



        wf1 = GenericWavefront2D.initialize_wavefront_from_steps(
                        x_start=x[0],x_step=numpy.abs(x[1]-x[0]),
                        y_start=y[0],y_step=numpy.abs(y[1]-y[0]),
                        number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        numpy.testing.assert_almost_equal(x,wf1.get_coordinate_x(),9)
        numpy.testing.assert_almost_equal(y,wf1.get_coordinate_y(),9)

        wf2 = GenericWavefront2D.initialize_wavefront_from_steps(
                        x_start=x[0],x_step=numpy.abs(x[1]-x[0]),
                        y_start=y[0],y_step=numpy.abs(y[1]-y[0]),
                        number_of_points=(npixels_x,npixels_y),wavelength=wavelength)


        numpy.testing.assert_almost_equal(x,wf2.get_coordinate_x(),9)
        numpy.testing.assert_almost_equal(y,wf2.get_coordinate_y(),9)
        # an spherical wavefront is obtained 1) by creation, 2) focusing a planewave

        wf1.set_spherical_wave(-5.0, 3+0j)
        wf1.clip_square(-50e-6,10e-6,-20e-6,40e-6)

        wf2.set_plane_wave_from_complex_amplitude(3+0j)

        ideal_lens = WOIdealLens("test", 5.0, 5.0)
        ideal_lens.applyOpticalElement(wf2)

        wf2.clip_square(-50e-6,10e-6,-20e-6,40e-6)



        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf1.get_phase(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),
                       title="Phase of spherical wavefront",show=0)
            plot_image(wf2.get_phase(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),
                       title="Phase of focused plane wavefront",show=0)
            plot_image(wf1.get_phase(from_minimum_intensity=0.1),wf2.get_coordinate_x(),wf2.get_coordinate_y(),
                       title="Phase of spherical wavefront (for intensity > 0.1)",show=0)
            plot_image(wf2.get_phase(from_minimum_intensity=0.1),wf2.get_coordinate_x(),wf2.get_coordinate_y(),
                       title="Phase of focused plane wavefront (for intensity > 0.1)",show=1)


        numpy.testing.assert_almost_equal(wf1.get_phase(),wf2.get_phase(),5)

    def test_gaussianhermite_mode(self,do_plot=do_plot):
        #
        # plane wave
        #
        print("#                                                             ")
        print("# Tests for a 2D Gaussian Hermite mode                        ")
        print("#                                                             ")

        wavelength        = 1.24e-10


        # 2D
        sigma_x = 100e-6
        mode_x = 0
        npixels_x = 100
        sigma_y = 50e-6
        mode_y = 3
        npixels_y = 100


        wavefront_length_x = 10*sigma_x
        wavefront_length_y = 10*sigma_y

        x = numpy.linspace(-0.5*wavefront_length_x,0.5*wavefront_length_x,npixels_x)
        y = numpy.linspace(-0.5*wavefront_length_y,0.5*wavefront_length_y,npixels_y)



        wf1 = GenericWavefront2D.initialize_wavefront_from_steps(
                        x_start=x[0],x_step=numpy.abs(x[1]-x[0]),
                        y_start=y[0],y_step=numpy.abs(y[1]-y[0]),
                        number_of_points=(npixels_x,npixels_y),wavelength=wavelength)


        wf1.set_gaussian_hermite_mode(sigma_x, sigma_y, mode_x, mode_y, amplitude=1.0)


        numpy.testing.assert_almost_equal(wf1.get_amplitude()[30,40],1383.76448118,3)

        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf1.get_amplitude(),wf1.get_coordinate_x(),wf1.get_coordinate_y(),
                       title="Amplitude of gaussianhermite mode",show=1)



    def test_interpolator(self,do_plot=do_plot):
        #
        # interpolator
        #
        print("#                                                             ")
        print("# Tests for 2D interpolator                                   ")
        print("#                                                             ")

        x = numpy.linspace(-10,10,100)
        y = numpy.linspace(-20,20,50)

        xy = numpy.meshgrid(x,y)
        X = xy[0].T
        Y = xy[1].T
        sigma = 3.0
        Z = 3*numpy.exp(- (X**2+Y**2)/2/sigma**2) +4j

        print("shape of Z",Z.shape)

        wf = GenericWavefront2D.initialize_wavefront_from_steps(x[0],x[1]-x[0],y[0],y[1]-y[0],number_of_points=(100,50))
        print("wf shape: ",wf.size())
        print("wf polarized: ",wf.is_polarized())
        wf.set_complex_amplitude( Z )

        x1 = 3.2
        y1 = -2.5
        z1 = 3*numpy.exp(- (x1**2+y1**2)/2/sigma**2) + 4j
        print("complex ampl at (%g,%g): %g+%gi (exact=%g+%gi)"%(x1,y1,
                                                        wf.get_interpolated_complex_amplitude(x1,y1).real,
                                                        wf.get_interpolated_complex_amplitude(x1,y1).imag,
                                                        z1.real,z1.imag))
        self.assertAlmostEqual(wf.get_interpolated_complex_amplitude(x1,y1).real,z1.real,4)
        self.assertAlmostEqual(wf.get_interpolated_complex_amplitude(x1,y1).imag,z1.imag,4)
        #
        print("intensity  at (%g,%g):   %g (exact=%g)"%(x1,y1,wf.get_interpolated_intensity(x1,y1),numpy.abs(z1)**2))
        self.assertAlmostEqual(wf.get_interpolated_intensity(x1,y1),numpy.abs(z1)**2,3)

        # interpolate on same grid

        interpolated_complex_amplitude_on_same_grid = wf.get_interpolated_complex_amplitudes(X,Y)
        print("Shape interpolated at same grid: ",interpolated_complex_amplitude_on_same_grid.shape)

        numpy.testing.assert_array_almost_equal(wf.get_complex_amplitude(),interpolated_complex_amplitude_on_same_grid,4)

        print("Total intensity original wavefront: %g, interpolated on the same grid: %g"%
              (wf.get_intensity().sum(), (numpy.abs(interpolated_complex_amplitude_on_same_grid)**2).sum()))



        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title="Original",show=0)
            plot_image(wf.get_interpolated_intensity(X,Y),wf.get_coordinate_x(),wf.get_coordinate_y(),
                       title="interpolated on same grid",show=1)


        # rebin wavefront

        # wf.set_plane_wave_from_complex_amplitude(3+4j)
        wf_rebin = wf.rebin(2.0,5.0,0.5,0.8,keep_the_same_intensity=1,set_extrapolation_to_zero=1)
        print("Shape before rebinning: ",wf.size())
        print("Shape after rebinning: ",wf_rebin.size())

        print("Total intensity original wavefront: %g, rebinned: %g"%
              (wf.get_intensity().sum(), wf_rebin.get_intensity().sum() ))

        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title="BEFORE REBINNING",show=0)
            plot_image(wf_rebin.get_intensity(),wf_rebin.get_coordinate_x(),wf_rebin.get_coordinate_y(),
                       title="REBINNED",show=1)



    def test_save_load_h5_file(self):

        wfr = GenericWavefront2D.initialize_wavefront_from_range(-0.004,0.004,-0.001,0.001,(500,200))
        wfr.set_gaussian(0.002/6,0.001/12)
        wfr.save_h5_file("tmp_wofry.h5",subgroupname="wfr", intensity=True,phase=True,overwrite=True)

        wfr.set_gaussian(0.002/6/2,0.001/12/2)
        print("Writing file: tmp_wofry.h5")
        wfr.save_h5_file("tmp_wofry.h5",subgroupname="wfr2", intensity=True,phase=False,overwrite=False)

        # test same amplitudes:
        print("Accessing file, path: ","tmp_wofry.h5","wfr2")
        wfr2 = GenericWavefront2D.load_h5_file("tmp_wofry.h5","wfr2")
        print("Cleaning file tmp_wofry.h5")
        os.remove("tmp_wofry.h5")
        assert(wfr2.is_identical(wfr))

    def test_polarization(self):
        print("#                                                             ")
        print("# Tests polarization (2D)                                     ")
        print("#                                                             ")
        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-0.5e-3,
                                                                       x_max=0.5e-3,
                                                                       y_min=-0.5e-3,
                                                                       y_max=0.5e-3,
                                                                       number_of_points=(2048,1024),
                                                                       wavelength=1.5e-10,
                                                                       polarization=Polarization.TOTAL)

        ca = numpy.zeros(wavefront.size())
        wavefront.set_complex_amplitude(ca+(1+0j),ca+(0+1j))
        #
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_phase(0.1e-3,0.1e-3,polarization=Polarization.SIGMA),0.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_phase(-0.1e-3,0.1e-3,polarization=Polarization.PI),numpy.pi/2 )

        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,-0.111e-3,polarization=Polarization.TOTAL),2.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,-0.111e-3,polarization=Polarization.SIGMA),1.0 )
        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(-0.111e-3,-0.111e-3,polarization=Polarization.PI),1.0 )

        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.SIGMA),(ca+1)**2 )
        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.PI),   (ca+1)**2 )
        numpy.testing.assert_almost_equal(wavefront.get_intensity(polarization=Polarization.TOTAL),2*(ca+1)**2 )



    def test_multiple_slit(self):
        print("#                                                             ")
        print("# Tests multiple slit (2D)                                     ")
        print("#                                                             ")
        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-0.5e-3,
                                                                       x_max=0.5e-3,
                                                                       y_min=-0.5e-3,
                                                                       y_max=0.5e-3,
                                                                       number_of_points=(2048,1024),
                                                                       wavelength=1.5e-10,
                                                                       polarization=Polarization.TOTAL)

        ca = numpy.zeros(wavefront.size())
        wavefront.set_complex_amplitude(ca+(10+0j),ca+(0+1j))
        #

        window_circle = wavefront.clip_circle(50e-6,-2e-4,-2e-4,apply_to_wavefront=False)
        window_rectangle = wavefront.clip_square(2e-4,3e-4,2e-4,3e-4,apply_to_wavefront=False)
        window_ellipse = wavefront.clip_ellipse(50e-6,25e-6,-2e-4,2e-4,apply_to_wavefront=False)
        window_ellipse2 = wavefront.clip_ellipse(50e-6,100e-6,2e-4,-2e-4,apply_to_wavefront=False)

        wavefront.clip_window(window_circle+window_rectangle+window_ellipse+window_ellipse2)

        if True:
            from srxraylib.plot.gol import plot_image
            plot_image(wavefront.get_intensity(),1e6*wavefront.get_coordinate_x(),1e6*wavefront.get_coordinate_y())

        numpy.testing.assert_almost_equal(wavefront.get_interpolated_intensities(0,0,polarization=Polarization.TOTAL),0.0 )



