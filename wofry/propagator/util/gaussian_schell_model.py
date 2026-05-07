"""
Gaussian-Schell model for partially coherent beams (1D and 2D).
"""
import numpy as np
import scipy.special as special

class GaussianSchellModel1D(object):
    """
    1-D Gaussian-Schell model for a partially coherent beam.

    Implements the cross-spectral density W(x1, x2) = sqrt(S(x1)) sqrt(S(x2)) g(x1-x2)
    following Mandel and Wolf, *Optical Coherence and Quantum Optics*, p. 253.
    """

    def __init__(self, A, sigma_s, sigma_g):
        """
        Parameters
        ----------
        A : float
            Amplitude of the spectral density S.
        sigma_s : float
            RMS width of the spectral density S [m].
        sigma_g : float
            RMS width of the spectral degree of coherence g [m].
        """
        self._A = A
        self._sigma_s = sigma_s
        self._sigma_g = sigma_g

    def S(self, x):
        """
        Spectral density (intensity profile).

        Parameters
        ----------
        x : array_like
            Transverse coordinate [m].

        Returns
        -------
        numpy.ndarray
            S(x) = A^2 exp(-x^2 / (2 sigma_s^2)).
        """
        S = (self._A ** 2) * np.exp(-(x**2)/(2*self._sigma_s**2))
        return S

    def g(self, x):
        """
        Spectral degree of coherence.

        Parameters
        ----------
        x : array_like
            Coordinate difference x1 - x2 [m].

        Returns
        -------
        numpy.ndarray
            g(x) = exp(-x^2 / (2 sigma_g^2)).
        """
        g = np.exp(-(x**2)/(2*self._sigma_g**2))
        return g

    def evaluate(self, x_1, x_2):
        """
        Evaluate the cross-spectral density W(x1, x2).

        Parameters
        ----------
        x_1 : array_like
            First transverse coordinate [m].
        x_2 : array_like
            Second transverse coordinate [m].

        Returns
        -------
        numpy.ndarray
            W(x1, x2) = sqrt(S(x1)) sqrt(S(x2)) g(x1-x2).
        """
        dr = x_1-x_2

        S_1 = self.S(x_1)
        S_2 = self.S(x_2)
        g = self.g(dr)

        result = np.sqrt(S_1) * np.sqrt(S_2) * g

        return result

    def a(self):
        """
        Auxiliary parameter a = 1 / (4 sigma_s^2).

        Returns
        -------
        float
        """
        a = 1.0/(4.0*self._sigma_s**2)
        return a

    def b(self):
        """
        Auxiliary parameter b = 1 / (2 sigma_g^2).

        Returns
        -------
        float
        """
        b = 1.0/(2.0*self._sigma_g**2)
        return b

    def c(self):
        """
        Auxiliary parameter c = sqrt(a^2 + 2ab).

        Returns
        -------
        float
        """
        a = self.a()
        b = self.b()

        res = np.sqrt(a**2 + 2 * a * b)
        return res

    def beta(self,n):
        """
        Eigenvalue of the n-th coherent mode.

        Parameters
        ----------
        n : int
            Mode order (0 = fundamental).

        Returns
        -------
        float
            beta_n = A^2 sqrt(pi/(a+b+c)) * (b/(a+b+c))^n.
        """
        a = self.a()
        b = self.b()
        c = self.c()

        res = self._A**2 * np.sqrt(np.pi/(a+b+c)) * (b/(a+b+c))**n
        return res

    def phi(self, n, x):
        """
        n-th coherent-mode eigenfunction (normalised Hermite-Gaussian).

        Parameters
        ----------
        n : int
            Mode order.
        x : array_like
            Transverse coordinate [m].

        Returns
        -------
        numpy.ndarray
            phi_n(x).
        """
        c = self.c()

        h_n = special.eval_hermite(n, x * np.sqrt(2*c))

        res = ((2*c/np.pi) ** 0.25) * np.exp(-c * x**2)
        res *= h_n * (1.0/np.sqrt(2**n * special.factorial(n) ))

        return res

class GaussianSchellModel2D(object):
    """
    2-D Gaussian-Schell model for a partially coherent beam.

    Separable product of two independent :class:`GaussianSchellModel1D`
    instances (one per transverse axis), following Mandel and Wolf,
    *Optical Coherence and Quantum Optics*, p. 253.
    """

    def __init__(self, A, sigma_s_x, sigma_g_x, sigma_s_y, sigma_g_y):
        """
        Parameters
        ----------
        A : float
            Overall amplitude of the spectral density S.
        sigma_s_x : float
            RMS beam size in x [m].
        sigma_g_x : float
            RMS coherence width in x [m].
        sigma_s_y : float
            RMS beam size in y [m].
        sigma_g_y : float
            RMS coherence width in y [m].
        """
        self._mode_x = GaussianSchellModel1D(A**0.5, sigma_s_x, sigma_g_x)
        self._mode_y = GaussianSchellModel1D(A**0.5, sigma_s_y, sigma_g_y)

        # For eigenvalue ordering.
        self._sorted_mode_indices = None

    def evaluate(self, r_1, r_2):
        """
        Evaluate the 2-D cross-spectral density W(r1, r2).

        Parameters
        ----------
        r_1 : array_like, shape (2,)
            First position vector [x1, y1] in metres.
        r_2 : array_like, shape (2,)
            Second position vector [x2, y2] in metres.

        Returns
        -------
        float or numpy.ndarray
            W(r1, r2) = W_x(x1, x2) * W_y(y1, y2).
        """
        x = self._mode_x.evaluate(r_1[0], r_2[0])
        y = self._mode_y.evaluate(r_1[1], r_2[1])

        result = x * y

        return result

    def beta(self, n_x, n_y):
        """
        Eigenvalue of the (n_x, n_y) coherent mode.

        Parameters
        ----------
        n_x : int
            Mode order in x.
        n_y : int
            Mode order in y.

        Returns
        -------
        float
            beta(n_x, n_y) = beta_x(n_x) * beta_y(n_y).
        """
        beta_x = self._mode_x.beta(n_x)
        beta_y = self._mode_y.beta(n_y)

        res = beta_x * beta_y

        return res

    def phi(self, n_x, n_y, x, y):
        """
        (n_x, n_y) coherent-mode eigenfunction evaluated at scalar coordinates.

        Parameters
        ----------
        n_x : int
            Mode order in x.
        n_y : int
            Mode order in y.
        x : float or array_like
            x coordinate [m].
        y : float or array_like
            y coordinate [m].

        Returns
        -------
        numpy.ndarray
            phi(n_x, n_y, x, y) = phi_x(n_x, x) * phi_y(n_y, y).
        """
        phi_x = self._mode_x.phi(n_x, x)
        phi_y = self._mode_y.phi(n_y, y)

        res = phi_x * phi_y

        return res

    def phi_nm(self,n_x, n_y, x_coords, y_coords):
        """
        (n_x, n_y) coherent-mode eigenfunction on a 2-D grid (outer product).

        Parameters
        ----------
        n_x : int
            Mode order in x.
        n_y : int
            Mode order in y.
        x_coords : array_like, shape (Nx,)
            x coordinates [m].
        y_coords : array_like, shape (Ny,)
            y coordinates [m].

        Returns
        -------
        numpy.ndarray, shape (Nx, Ny)
            phi_nm = outer(phi_x(n_x, x_coords), phi_y(n_y, y_coords)).
        """
        phi_x = self._mode_x.phi(n_x, x_coords)
        phi_y = self._mode_y.phi(n_y, y_coords)

        res = np.outer(phi_x,  phi_y)
        return res

    def sortedModeIndices(self, index_energy, n_points=50):
        """
        Return mode indices (n, m) sorted by descending eigenvalue.

        The sorted list is computed once and cached.

        Parameters
        ----------
        index_energy : int
            Rank of the desired mode (0 = highest eigenvalue).
        n_points : int, optional
            Number of mode orders to consider in each direction. Default 50.

        Returns
        -------
        n : int
            x-mode order of the requested ranked mode.
        m : int
            y-mode order of the requested ranked mode.
        """
        if self._sorted_mode_indices is None:
            eigenvalues_x = np.array([self._mode_x.beta(i) for i in range(n_points)])
            eigenvalues_y = np.array([self._mode_y.beta(i) for i in range(n_points)])

            f = np.outer(eigenvalues_x, eigenvalues_y)
            self._sorted_mode_indices = np.array(np.unravel_index(f.flatten().argsort()[::-1], (n_points, n_points)))

        n, m = self._sorted_mode_indices[:, index_energy]
        return n, m

if __name__ == "__main__":

    from srxraylib.plot.gol import plot,plot_image
    # 1D
    sigmaX = 100e-6
    modeX = 100
    a1D = GaussianSchellModel1D(1.0,sigmaX,100.0*sigmaX)

    x = np.linspace(-30*sigmaX,30*sigmaX,1000)
    y = a1D.phi(modeX,x)

    plot(x,y)

    print(">>",y[30])

    # # 2D
    # sigmaX = 100e-6
    # modeX = 0
    # nX = 100
    # sigmaY = 50e-6
    # modeY = 3
    # nY = 100
    # a2D = GaussianSchellModel2D(1.0,sigmaX,100.0*sigmaX,sigmaY,100.0*sigmaY)
    #
    # x = np.linspace(-5*sigmaX,5*sigmaX,nX)
    # y = np.linspace(-5*sigmaY,5*sigmaY,nY)
    #
    # Phi = a2D.phi_nm(modeX,modeY,x,y)
    #
    # print(">>",Phi[30,40])
    #
    # plot_image(Phi)