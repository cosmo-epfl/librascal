"""
Utilities to manipulate CG coefficients, rotations, and equivariant construction in general.
Requires sympy to compute relevant coefficients.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from ..neighbourlist.structure_manager import is_ase_Atoms


# Just a few wrappers for sympy/scipy utility functions
# Only loads sympy on request given it adds a significant 
# overhead to librascal loading
def _wigner_d(l, alpha, beta, gamma):
    """ Computes a Wigner D matrix 
    D^l_{mm'}(alpha, beta, gamma)
   from sympy and converts it to numerical values.  
   (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
   """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    return np.complex128(wigner_d(l, alpha, beta, gamma))


def _rotation(alpha, beta, gamma):
    """ A Cartesian rotation matrix in the appropriate convention 
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()


def _cg(l1, l2, L):
    """ Computes CG coefficients from sympy 
    <l1 m1; l2 m2| L M>
    and converts them to numerical values.  
    Returns a full (2 * l1 + 1, 2 * l2 + 1, 2 * L + 1) array, which
    is mostly zeros.
    """
    try:
        from sympy.physics.quantum.cg import CG
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Clebsch-Gordan coefficients requires a sympy installation"
        )

    rcg = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return rcg
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            if np.abs(m1 + m2) > L:
                continue
            rcg[l1 + m1, l2 + m2, L + (m1 + m2)] += np.double(
                CG(l1, m1, l2, m2, L, m1 + m2).doit()
            )
    return rcg


# Convert from real to complex spherical harmonics (or corresponding
# coefficients) and back. Uses the convention from Wikipedia
isqrt2 = 1.0 / np.sqrt(2)
sqrt2 = np.sqrt(2)
def _r2c(sp):
    """ Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l """
    
    l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = np.zeros(len(sp), dtype=np.complex128)
    rc[l] = sp[l]
    for m in range(1, l + 1):
        rc[l + m] = (sp[l + m] + 1j * sp[l - m]) * isqrt2 * (-1) ** m
        rc[l - m] = (sp[l + m] - 1j * sp[l - m]) * isqrt2
    return rc

def _c2r(cp):
    """ Complex to real SPH. Assumes a block with 2l+1 complex 
    corresponding to Y^m_l with m indices from -l to +l """
    l = (len(cp) - 1) // 2  # infers l from the vector size
    rs = np.zeros(len(cp), dtype=np.float64)
    rs[l] = np.real(cp[l])
    for m in range(1, l + 1):
        rs[l - m] = (-1) ** m * sqrt2 * np.imag(cp[l + m])
        rs[l + m] = (-1) ** m * sqrt2 * np.real(cp[l + m])
    return rs


def spherical_expansion_reshape(spx, max_radial, max_angular, **kwargs):
    """
    Folds a list of spherical expansion coefficients in a
    n_env, n_el, nmax, (lmax+1)^2 form that is more convenient to evaluate.
    Can be called with **hypers in the same format as for a SphericalExpansion
    object.
    """
    
    lmshape = (max_angular + 1) ** 2
    nid = len(spx)
    nel = spx.shape[1] // (max_radial * lmshape)
    return spx.reshape(
        (nid, nel, max_radial, lmshape)
    )  # (lm) terms are stored in a compact form

def sph_conjugate_real(sp):
    """ Computes the "complex conjugate" of real spherical harmonics or 
    associated coefficients, basically <lm|rhat>^* = (-1)^m <l (-m)|rhat>.
    In this form, the transformation applies also to real coefficients. 
    Assumes that the argument is a m=(-l..l) block. 
    """
    
    lm = sp.shape[-1]    
    l = (lm - 1)//2
    ones = np.ones(lm)
    if l%2==0:
        ones[1::2] = -1
    else:
        ones[0::2] = -1
    return sp

def lm_slice(l):
    """
    Simple helper function to get the slice corresponding to m=-l..l in a dense
    storage block for Y^l_m-like coefficients
    """
    return slice(l * l, (l + 1) * (l + 1), 1)


class WignerDReal:
    """
    A helper class to compute Wigner D matrices given the Euler angles of a rotation,
    and apply them to spherical harmonics (or coefficients). Built to function with
    real-valued coefficients.
    """

    def __init__(self, lmax, alpha, beta, gamma):
        """
        Initialize the WignerDReal class.

        lmax: int
            maximum angular momentum channel for which the Wigner D matrices are
            computed

        alpha, beta, gamma: float
            Euler angles, in radians
        """
        self._lmax = lmax

        self._rotation = _rotation(alpha, beta, gamma)

        r2c_mats = {}
        c2r_mats = {}
        for L in range(0, self._lmax + 1):
            r2c_mats[L] = np.hstack(
                [_r2c(np.eye(2 * L + 1)[i])[:, np.newaxis] for i in range(2 * L + 1)]
            )
            c2r_mats[L] = np.conjugate(r2c_mats[L]).T
        self._wddict = {}
        for L in range(0, self._lmax + 1):
            wig = _wigner_d(L, alpha, beta, gamma)
            self._wddict[L] = np.real(c2r_mats[L] @ np.conjugate(wig) @ r2c_mats[L])

    def rotate(self, rho):
        """
        Rotates a vector of 2l+1 spherical harmonics (coefficients) according to the
        rotation defined in the initialization.

        rho: array
            List of 2l+1 coefficients

        Returns:
        --------
        (2l+1) array containing the coefficients for the rotated structure
        """

        L = (rho.shape[-1] - 1) // 2
        return rho @ self._wddict[L].T

    def rotate_frame(self, frame):
        """
        Utility function to also rotate a structure, given as an Atoms frame.
        NB: it will rotate positions and cell, and no other array.

        frame: ase.Atoms
            An atomic structure in ASE format, that will be modified in place

        Returns:
        -------
        The rotated frame
        """
        
        if is_ase_Atoms(frame):
            frame.positions = frame.positions @ self._rotation.T
            frame.cell = frame.cell @ self._rotation.T
        else:
            print(frame["positions"])
            frame["positions"] = self._rotation@frame["positions"]
            frame["cell"] = self._rotation@frame["cell"] 
        return frame


class ClebschGordanReal:
    """
    Helper class to manipulate Clebsch-Gordan coefficients with real-storage
    spherical harmonics and coefficients.
    """

    def __init__(self, lmax):
        """
        Initialize the ClebschGordanReal class, precomputing transformation coefficients
        up to lmax

        lmax: int
            Maximum angular momentum channel that is computed
        """

        self._lmax = lmax
        self._cgdict = {}

        # real-to-complex and complex-to-real transformations as matrices
        r2c_mats = {}
        c2r_mats = {}
        for L in range(0, self._lmax + 1):
            r2c_mats[L] = np.hstack(
                [_r2c(np.eye(2 * L + 1)[i])[:, np.newaxis] for i in range(2 * L + 1)]
            )
            c2r_mats[L] = np.conjugate(r2c_mats[L]).T

        for l1 in range(self._lmax + 1):
            for l2 in range(self._lmax + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._lmax, (l1 + l2)) + 1
                ):
                    # computes CG coefficients
                    ccg = _cg(l1, l2, L)

                    # applies transformations that make them act and generate real
                    # valued coefficients
                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(
                            np.einsum(
                                "abc, ax, by, zc -> xyz",
                                ccg,
                                r2c_mats[l1],
                                r2c_mats[l2],
                                c2r_mats[L],
                            )
                        )
                    else:
                        rcg = np.imag(
                            np.einsum(
                                "abc, ax, by, zc -> xyz",
                                ccg,
                                r2c_mats[l1],
                                r2c_mats[l2],
                                c2r_mats[L],
                            )
                        )

                    # tricky: real-valued transformations have a funny layout, and
                    # so it is best to store them in a sparse format: for each l
                    # channel, it stores a list of lists of tuples one list per m
                    # channel, and for each tuple (m1, m2, coefficient).
                    newcg = []
                    for M in range(2 * L + 1):
                        cgm = []
                        for m1 in range(2 * l1 + 1):
                            for m2 in range(2 * l2 + 1):
                                if np.abs(rcg[m1, m2, M]) > 1e-15:
                                    cgm.append((m1, m2, rcg[m1, m2, M]))
                        newcg.append(cgm)
                    self._cgdict[(l1, l2, L)] = newcg

    def combine(self, rho1, rho2, L):
        """
        Combines Ylm-like coefficients to generate an equivariant
        term of order L using spherical expansion coefficients.


        """

        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[-1] - 1) // 2
        l2 = (rho2.shape[-1] - 1) // 2
        if L > self._lmax or l1 > self._lmax or l2 > self._lmax:
            raise ValueError("Requested CG entry has not been precomputed")

        rho_shape = rho1.shape[:-1]
        if rho1.shape[:-1] != rho2.shape[:-1]:
            raise IndexError("Cannot combine differently-shaped feature blocks")

        rho1_reshaped = rho1.reshape((-1, 2 * l1 + 1))
        rho2_reshaped = rho2.reshape((-1, 2 * l2 + 1))
        rho = np.zeros((rho1_reshaped.shape[0], 2 * L + 1))
        if (l1, l2, L) in self._cgdict:
            # Failsafe implementation: if user requests an "impossible" coupling,
            # return an empty vector
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cgdict[(l1, l2, L)][M]:
                    rho[:, M] += rho1_reshaped[:, m1] * rho2_reshaped[:, m2] * cg

        return rho.reshape(rho_shape + (2 * L + 1,))

    def combine_einsum(self, rho1, rho2, L, combination_string):
        """
        Combines Ylm-like coefficients to generate an equivariant
        term of order L using spherical expansion coefficients.
        Uses einsum for maximum flexibility in how the two sets of
        features are combined.
        """

        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[-1] - 1) // 2
        l2 = (rho2.shape[-1] - 1) // 2
        if L > self._lmax or l1 > self._lmax or l2 > self._lmax:
            raise ValueError("Requested CG entry has not been precomputed")

        # infers the shape of the output using the einsum internals
        rho_shape = np.einsum(combination_string, rho1[..., 0], rho2[..., 0]).shape
        rho = np.zeros(rho_shape + (2 * L + 1,))

        if (l1, l2, L) in self._cgdict:
            # Failsafe implementation: if user requests an "impossible" coupling,
            # return an empty vector
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cgdict[(l1, l2, L)][M]:
                    rho[..., M] += np.einsum(
                        combination_string, rho1[..., m1], rho2[..., m2] * cg
                    )

        return rho

    def couple(self, rho2d):
        """
        Goes from an uncoupled product basis to a coupled basis, taking a
        (2l1+1)x(2l2+1) matrix and transforming it into a list of coupled
        vectors, each transforming like a Y^L irrep.

        rho2d: array
            (2l1+1)x(2l2+1) array containing coefficients that transforms like
            products of Y^l1 and Y^l2 harmonics

        Returns:
        --------
        A tuple tracking the nature of the coupled object, that contains in turns
        (l1, l2) the coefficients of the parent Ylm
        a dictionary of coupled terms, in the form L:(2L+1) array
        """
        l1 = (rho2d.shape[0] - 1) // 2
        l2 = (rho2d.shape[1] - 1) // 2

        coupled = {}
        for L in range(max(l1, l2) - min(l1, l2), min(self._lmax, (l1 + l2)) + 1):
            Lterm = np.zeros(2 * L + 1)
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cgdict[(l1, l2, L)][M]:
                    Lterm[M] += rho2d[m1, m2] * cg
            coupled[L] = Lterm
        return ((l1, l2), coupled)

    def decouple(self, rho_coupled):
        """
        Undoes the transformation enacted by couple.
        """

        l1, l2 = rho_coupled[0]

        decoupled = np.zeros((2 * l1 + 1, 2 * l2 + 1))
        for L in range(max(l1, l2) - min(l1, l2), min(self._lmax, (l1 + l2)) + 1):
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cgdict[(l1, l2, L)][M]:
                    decoupled[m1, m2] += cg * rho_coupled[1][L][M]
        return decoupled
