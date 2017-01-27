import unittest
import numpy

from srxraylib.waveoptics.wavefront import Wavefront1D
from srxraylib.waveoptics.wavefront2D import Wavefront2D

from srxraylib.waveoptics.propagator import propagate_1D_fraunhofer
from srxraylib.waveoptics.propagator import propagate_1D_integral
from srxraylib.waveoptics.propagator import propagate_1D_fresnel, propagate_1D_fresnel_convolution
from srxraylib.waveoptics.propagator2D import propagate_2D_fraunhofer
from srxraylib.waveoptics.propagator2D import propagate_2D_integral
from srxraylib.waveoptics.propagator2D import propagate_2D_fresnel, propagate_2D_fresnel_convolution, propagate_2D_fresnel_srw

do_plot = True

if do_plot:
    from srxraylib.plot.gol import plot,plot_image,plot_table

try:
    import srwlib
    SRWLIB_AVAILABLE = True
except:
    SRWLIB_AVAILABLE = False
    print("SRW is not available")

#
# some common tools
#
def get_theoretical_diffraction_pattern(angle_x,
                                        aperture_type='square',aperture_diameter=40e-6,
                                        wavelength=1.24e-10,normalization=True):

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':
        # remove x=0 (to avoid 0/0) #TODO: check this
        # indices_ok = numpy.where(angle_x != 0)
        # angle_x = angle_x[indices_ok]
        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'gaussian':
        sigma = aperture_diameter/2.35
        sigma_ft = 1.0 / sigma * wavelength / (2.0 * numpy.pi)
        # Factor 2.0 is because we wwant intensity (amplitude**2)
        intensity_theory = numpy.exp( -2.0*(angle_x**2/sigma_ft**2/2) )
    else:
        raise Exception("Undefined aperture type (accepted: circle, square, gaussian)")

    if normalization:
        intensity_theory /= intensity_theory.max()

    return intensity_theory

def line_image(image,horizontal_or_vertical='H'):
    if horizontal_or_vertical == "H":
        tmp = image[:,image.shape[1]/2]
    else:
        tmp = image[image.shape[0]/2,:]
    return tmp

def line_fwhm(line):
    #
    #CALCULATE fwhm in number of abscissas bins (supposed on a regular grid)
    #
    tt = numpy.where(line>=max(line)*0.5)
    if line[tt].size > 1:
        # binSize = x[1]-x[0]
        FWHM = (tt[0][-1]-tt[0][0])
        return FWHM
    else:
        return -1


class propagatorTest(unittest.TestCase):

    #
    # Common interface for all 1D methods :
    #                                   'fraunhofer':
    #                                   'fft': fft -> multiply by kernel in freq -> ifft
    #                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
    # valid apertute_type: square, gaussian

    def propagate_1D(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                wavefront_length=100e-6,npoints=500,
                                propagation_distance = 30.0,show=1):


        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")

        wf = Wavefront1D.initialize_wavefront_from_range(x_min=-wavefront_length/2, x_max=wavefront_length/2,
                                                                number_of_points=npoints,wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((2.0+1.0j)) # an arbitraty value



        if aperture_type == 'square':
            wf.apply_slit(-aperture_diameter/2, aperture_diameter/2)
        elif aperture_type == 'gaussian':
            X = wf.get_mesh_x()
            Y = wf.get_mesh_y()
            window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
            wf.rescale_amplitudes(window)
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")


        if method == 'fft':
            wf1 = propagate_1D_fresnel(wf, propagation_distance)
        elif method == 'convolution':
            wf1 = propagate_1D_fresnel_convolution(wf, propagation_distance)
        elif method == 'integral':
            wf1 = propagate_1D_integral(wf, propagation_distance)
        elif method == 'fraunhofer':
            wf1 = propagate_1D_fraunhofer(wf, propagation_distance)
        else:
            raise Exception("Not implemented method: %s"%method)

        # get the theoretical value
        angle_x = wf1.get_abscissas() / propagation_distance

        intensity_theory = get_theoretical_diffraction_pattern(angle_x,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)

        intensity_calculated =  wf1.get_intensity()
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_abscissas()*1e6/propagation_distance,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["%s "%method,"Theoretical (far field)"],
                 legend_position=(0.95, 0.95),
                 title="1D (%s) diffraction from a %s aperture of %3.1f um at wavelength of %3.1f A"%
                       (method,aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)

        return wf1.get_abscissas()/propagation_distance,intensity_calculated,intensity_theory


    #
    # tests/example cases for 1D propagators
    #

    def test_propagate_1D_fraunhofer_bis(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="fraunhofer",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = 1.0, show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    def test_propagate_1D_fft(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="fft",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance, show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


    def test_propagate_1D_fresnel_convolution(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="convolution",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance, show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    def test_propagate_1D_integral(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="integral",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance, show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


class propagator2DTest(unittest.TestCase):
    #
    # TOOLS
    #


    #
    # Common interface for all methods using fresnel, via convolution in FT space
    #          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
    #                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
    #                                   'srw': use the SRW package
    def propagate_2D_fresnel(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance = 30.0,show=1):


        method_label = "fresnel (%s)"%method
        print("\n#                                                             ")
        print("# 2D near field fresnel (%s) diffraction from a %s aperture  "%(method_label,aperture_type))
        print("#                                                             ")


        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))

        wf = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))

        if aperture_type == 'circle':
            wf.apply_pinhole(aperture_diameter/2)
        elif aperture_type == 'square':
            wf.apply_slit(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
        elif aperture_type == 'gaussian':
            X = wf.get_mesh_x()
            Y = wf.get_mesh_y()
            window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
            wf.rescale_amplitudes(window)
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")


        if method == 'fft':
            wf1 = propagate_2D_fresnel(wf, propagation_distance)
        elif method == 'convolution':
            wf1 = propagate_2D_fresnel_convolution(wf, propagation_distance)
        elif method == 'srw':
            wf1 = propagate_2D_fresnel_srw(wf, propagation_distance)
        elif method == 'integral':
            wf1 = propagate_2D_integral(wf, propagation_distance)
        else:
            raise Exception("Not implemented method: %s"%method)


        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1.get_intensity(),
                       1e6*wf1.get_coordinate_x()/propagation_distance,
                       1e6*wf1.get_coordinate_y()/propagation_distance,
                       title="Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,method_label,1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        # get the theoretical value
        angle_x = wf1.get_coordinate_x() / propagation_distance

        intensity_theory = get_theoretical_diffraction_pattern(angle_x,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)

        intensity_calculated =  wf1.get_intensity()[:,wf1.size()[1]/2]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["%s H profile"%method_label,"Theoretical (far field)"],
                 legend_position=(0.95, 0.95),
                 title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (method_label,aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)

        return wf1.get_coordinate_x()/propagation_distance,intensity_calculated,angle_x,intensity_theory

    #
    #
    #
    def propagation_with_lens(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,
                                pixelsize_x=1e-6,npixels_x=2000,pixelsize_y=1e-6,npixels_y=2000,
                                propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1,show=1):


        method_label = "fresnel (%s)"%method
        print("\n#                                                             ")
        print("# near field fresnel (%s) diffraction and focusing  "%(method_label))
        print("#                                                             ")

        #                               \ |  /
        #   *                           | | |                      *
        #                               / | \
        #   <-------    d  ---------------><---------   d   ------->
        #   d is propagation_distance

        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))

        wf = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        spherical_or_plane_and_lens = 0
        if spherical_or_plane_and_lens == 0:
            # set spherical wave at the lens entrance (radius=distance)
            wf.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)
        else:
            # apply lens that will focus at propagation_distance downstream the lens.
            # Note that the vertical is a bit defocused
            wf.set_plane_wave_from_complex_amplitude(1.0+0j)
            focal_length = propagation_distance # / 2
            wf.apply_ideal_lens(focal_length,focal_length)

        print("Incident intensity: ",wf.get_intensity().sum())

        # propagation downstream the lens to image plane
        for i in range(propagation_steps):
            if propagation_steps > 1:
                print(">>> Propagating step %d of %d; propagation_distance=%g m"%(i+1,propagation_steps,
                                                    propagation_distance*defocus_factor/propagation_steps))
            if method == 'fft':
                wf = propagate_2D_fresnel(wf, propagation_distance*defocus_factor/propagation_steps)
            elif method == 'convolution':
                wf = propagate_2D_fresnel_convolution(wf, propagation_distance*defocus_factor/propagation_steps)
            elif method == 'srw':
                wf = propagate_2D_fresnel_srw(wf, propagation_distance*defocus_factor/propagation_steps)
            elif method == 'fraunhofer':
                wf = propagate_2D_fraunhofer(wf, propagation_distance*defocus_factor/propagation_steps)
            else:
                raise Exception("Not implemented method: %s"%method)




        horizontal_profile = wf.get_intensity()[:,wf.size()[1]/2]
        horizontal_profile /= horizontal_profile.max()
        print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf.delta()[0]))
        vertical_profile = wf.get_intensity()[wf.size()[0]/2,:]
        vertical_profile /= vertical_profile.max()
        print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf.delta()[1]))

        if do_plot:
            from srxraylib.plot.gol import plot,plot_image
            plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='intensity (%s)'%method,show=0)
            # plot_image(wf.get_amplitude(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='amplitude (%s)'%method,show=0)
            plot_image(wf.get_phase(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='phase (%s)'%method,show=0)

            plot(wf.get_coordinate_x(),horizontal_profile,
                 wf.get_coordinate_y(),vertical_profile,
                 legend=['Horizontal profile','Vertical profile'],title="%s"%method,show=show)

        print("Output intensity: ",wf.get_intensity().sum())
        return wf.get_coordinate_x(),horizontal_profile

    #
    # TESTS
    #

    def test_propagate_2D_fraunhofer(self,do_plot=do_plot,aperture_type='square',aperture_diameter=40e-6,
                    pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,wavelength=1.24e-10):
        """

        :param do_plot: 0=No plot, 1=Do plot
        :param aperture_type: 'circle' 'square' 'gaussian' (Gaussian sigma = aperture_diameter/2.35)
        :param aperture_diameter:
        :param pixelsize_x:
        :param pixelsize_y:
        :param npixels_x:
        :param npixels_y:
        :param wavelength:
        :return:
        """

        print("\n#                                                            ")
        print("# far field 2D (fraunhofer) diffraction from a square aperture  ")
        print("#                                                            ")

        method = "fraunhofer"

        print("Fraunhoffer diffraction valid for distances > > a^2/lambda = %f m"%((aperture_diameter/2)**2/wavelength))

        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))
        wf = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))

        if aperture_type == 'circle':
            wf.apply_pinhole(aperture_diameter/2)
        elif aperture_type == 'square':
            wf.apply_slit(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
        elif aperture_type == 'gaussian':
            X = wf.get_mesh_x()
            Y = wf.get_mesh_y()
            window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
            wf.rescale_amplitudes(window)
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")



        wf1 = propagate_2D_fraunhofer(wf, propagation_distance=1.0) # propagating at 1 m means the result is like in angles

        if do_plot:
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                       title="2D Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,method,1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        angle_x = wf1.get_coordinate_x() # + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        intensity_theory = get_theoretical_diffraction_pattern(angle_x,
                                            aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)


        intensity_calculated =  wf1.get_intensity()[:,wf1.size()[1]/2]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            plot(wf1.get_coordinate_x()*1e6,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["Calculated (FT) H profile","Theoretical"],legend_position=(0.95, 0.95),
                 title="2D Fraunhofer Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-80,80])

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


    def test_propagate_2D_fresnel_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='fft',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    def test_propagate_2D_fresnel_convolution_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='convolution',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    def test_propagate_2D_fresnel_srw_square(self):

        if not SRWLIB_AVAILABLE:
            print("SRW not available, skipping test_propagate_2D_fresnel_srw_square")
            return

        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='srw',aperture_type='square',
                                aperture_diameter=40e-6,
                                #pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                pixelsize_x=1e-6*2,pixelsize_y=1e-6*4,npixels_x=1024/2,npixels_y=1024/4,
                                propagation_distance=30.0,wavelength=1.24e-10)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    def test_propagate_2D_fresnel_integral_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='integral',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6*2,pixelsize_y=1e-6*4,npixels_x=1024/2,npixels_y=1024/4,
                                propagation_distance=30.0,wavelength=1.24e-10)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)



    def test_propagate_2D_fresnel_all_circle(self):

        xcalc_fft, ycalc_fft, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=0,method='fft',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=5.0,wavelength=1.24e-10)

        if  SRWLIB_AVAILABLE:

            xcalc_srw, ycalc_srw, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=0,method='srw',aperture_type='square',
                                    aperture_diameter=40e-6,
                                    pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                    propagation_distance=5.0,wavelength=1.24e-10)

            xcalc_srw, ycalc_convolution, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=0,method='convolution',aperture_type='square',
                                    aperture_diameter=40e-6,
                                    pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                    propagation_distance=5.0,wavelength=1.24e-10)


            if do_plot:
               x = xcalc_srw
               y = numpy.vstack((ycalc_fft,ycalc_srw,ycalc_convolution))
               plot_table(1e6*x,y,legend=["fft","srw","convolution"],ytitle="Intensity",xtitle="x coodinate [um]",
                              title="Comparison circular aperture - near field")

            numpy.testing.assert_almost_equal(ycalc_fft,ycalc_srw,1)
            numpy.testing.assert_almost_equal(ycalc_convolution,ycalc_srw,1)

    def test_lens(self):

        lens_diameter = 0.002
        npixels_x = 2048
        pixelsize_x = lens_diameter / npixels_x
        print("pixelsize: ",pixelsize_x)




        pixelsize_y = pixelsize_x
        npixels_y = npixels_x

        wavelength = 1.24e-10
        propagation_distance = 30.0
        defocus_factor = 1.0 # 1.0 is at focus
        propagation_steps = 1

        x_fft, y_fft = self.propagation_with_lens(do_plot=0,method='fft',
                                propagation_steps=propagation_steps,
                                wavelength=wavelength,
                                pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)
        if SRWLIB_AVAILABLE:
            x_srw, y_srw = self.propagation_with_lens(do_plot=0,method='srw',
                                    propagation_steps=propagation_steps,
                                    wavelength=wavelength,
                                    pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                    propagation_distance = propagation_distance, defocus_factor=defocus_factor)


        x_convolution, y_convolution = self.propagation_with_lens(do_plot=0,method='convolution',
                                propagation_steps=propagation_steps,
                                wavelength=wavelength,
                                pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)

        if do_plot:
            if SRWLIB_AVAILABLE:
                x = x_fft
                y = numpy.vstack((y_fft,y_srw,y_convolution))

                plot_table(1e6*x,y,legend=["fft","srw","convolution"],ytitle="Intensity",xtitle="x coordinate [um]",
                           title="Comparison 1:1 focusing")
            else:
                x = x_fft
                y = numpy.vstack((y_fft,y_convolution))

                plot_table(1e6*x,y,legend=["fft","convolution"],ytitle="Intensity",xtitle="x coordinate [um]",
                           title="Comparison 1:1 focusing")

        numpy.testing.assert_almost_equal(y_fft,y_convolution,1)

        if SRWLIB_AVAILABLE:
            numpy.testing.assert_almost_equal(y_fft,y_srw,1)
            numpy.testing.assert_almost_equal(y_convolution,y_srw,1)

