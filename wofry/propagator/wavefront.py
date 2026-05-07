"""
Abstract base classes for wofry wavefronts (Wavefront, WavefrontDimension).
"""

import pickle

class Wavefront(object):
    """
    Abstract base class for all wofry wavefronts.

    Concrete subclasses must implement :meth:`get_dimension` and
    :meth:`duplicate`. Serialisation to/from a hex string is provided
    via pickle.
    """

    def __init__(self):
        """Initialise the wavefront."""
        super().__init__()

    def get_dimension(self):
        """Return the wavefront dimension ('1' or '2').

        Returns
        -------
        str
            ``WavefrontDimension.ONE`` or ``WavefrontDimension.TWO``.
        """
        raise NotImplementedError("method is abstract")

    def duplicate(self):
        """Return a deep copy of this wavefront."""
        raise NotImplementedError("method is abstract")

    def to_hex_tring(self):
        """Serialise the wavefront to a hex string via pickle.

        Returns
        -------
        str
            Hex-encoded pickle bytes.
        """
        return pickle.dumps(self).hex()

    @classmethod
    def from_hex_tring(cls, hex_string):
        """Deserialise a wavefront from a hex string produced by :meth:`to_hex_tring`.

        Parameters
        ----------
        hex_string : str
            Hex-encoded pickle bytes.

        Returns
        -------
        Wavefront
            Reconstructed wavefront instance.
        """
        return pickle.loads(bytes.fromhex(hex_string))


class WavefrontDimension:
    """Constants identifying the spatial dimension of a wavefront."""
    ONE = "1"
    TWO = "2"
