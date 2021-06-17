"""Kernel ridge regression: Model class and utilities

Public classes:
    Kernel  Kernel Ridge Regression model (sparse GPR only).

Public functions:
    compute_KNM             Compute GAP kernel of a set of structures
    train_gap_model         Train a GAP model given a kernel matrix and sparse points
"""
from ..utils import BaseIO
from ..lib import compute_sparse_kernel_gradients, compute_sparse_kernel_neg_stress

import scipy
import numpy as np
import ase


class SparseGPRSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the sparse GPR problem
    b = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM


    """

    def __init__(
        self, KMM, regularizer=1, jitter=0, solver="RKHS", relative_jitter=True
    ):

        self._solver = solver

        self._nM = len(KMM)
        if self._solver == "RKHS" or self._solver == "RKHS-QR":
            self._vk, self._Uk = scipy.linalg.eigh(KMM)
            self._vk = self._vk[::-1]
            self._Uk = self._Uk[:, ::-1]
        elif self._solver == "QR" or self._solver == "Normal":
            self._KMM = KMM
            # gets maximum eigenvalue of KMM to scale the numerical jitter
            self._KMM_maxeva = scipy.sparse.linalg.eigsh(
                KMM, k=1, return_eigenvectors=False
            )[0]
        else:
            raise ValueError(
                "Solver ",
                solver,
                " not supported. Possible values are [RKHS, RKHS-QR, QR, Normal].",
            )
        if relative_jitter:
            if self._solver == "RKHS" or self._solver == "RKHS-QR":
                self._jitter_scale = self._vk[0]
            elif self._solver == "QR" or self._solver == "Normal":
                self._jitter_scale = self._KMM_maxeva
        else:
            self._jitter_scale = 1.0
        self.set_regularizers(regularizer, jitter)

    def set_regularizers(self, regularizer=1.0, jitter=0.0):
        self._regularizer = regularizer
        self._jitter = jitter
        if self._solver == "RKHS" or self._solver == "RKHS-QR":
            self._nM = len(np.where(self._vk > self._jitter * self._jitter_scale)[0])
            self._PKPhi = self._Uk[:, : self._nM] * 1 / np.sqrt(self._vk[: self._nM])
        elif self._solver == "QR":
            self._VMM = scipy.linalg.cholesky(
                self._regularizer * self._KMM
                + np.eye(self._nM) * self._jitter_scale * self._jitter
            )
        self._Cov = np.zeros((self._nM, self._nM))
        self._KY = None

    def partial_fit(self, KNM, Y, accumulate_only=False):
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if self._solver == "RKHS":
            Phi = KNM @ self._PKPhi
        elif self._solver == "Normal":
            Phi = KNM
        else:
            raise ValueError(
                "Partial fit can only be realized with solver = [RKHS, Normal]"
            )
        if self._KY is None:
            self._KY = np.zeros((self._nM, Y.shape[1]))

        self._Cov += Phi.T @ Phi
        self._KY += Phi.T @ Y

        if not accumulate_only:
            if self._solver == "RKHS":
                self._weights = self._PKPhi @ scipy.linalg.solve(
                    self._Cov + np.eye(self._nM) * self._regularizer,
                    self._KY,
                    assume_a="pos",
                )
            elif self._solver == "Normal":
                self._weights = scipy.linalg.solve(
                    self._Cov
                    + self._regularizer * self._KMM
                    + np.eye(self._KMM.shape[0]) * self._jitter * self._jitter_scale,
                    self._KY,
                    assume_a="pos",
                )

    def fit(self, KNM, Y):

        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if self._solver == "RKHS":
            Phi = KNM @ self._PKPhi
            self._weights = self._PKPhi @ scipy.linalg.solve(
                Phi.T @ Phi + np.eye(self._nM) * self._regularizer,
                Phi.T @ Y,
                assume_a="pos",
            )
        elif self._solver == "RKHS-QR":
            A = np.vstack(
                [KNM @ self._PKPhi, np.sqrt(self._regularizer) * np.eye(self._nM)]
            )
            Q, R = np.linalg.qr(A)
            self._weights = self._PKPhi @ scipy.linalg.solve_triangular(
                R, Q.T @ np.hstack([Y, np.zeros((self._nM, Y.shape[1]))])
            )
        elif self._solver == "QR":
            A = np.vstack([KNM, self._VMM])
            Q, R = np.linalg.qr(A)
            self._weights = scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((KNM.shape[1], Y.shape[1]))])
            )
        elif self._solver == "Normal":
            self._weights = scipy.linalg.solve(
                KNM.T @ KNM
                + self._regularizer * self._KMM
                + np.eye(self._nM) * self._jitter * self._jitter_scale,
                KNM.T @ Y,
                assume_a="pos",
            )

    def transform(self, KTM):
        return KTM @ self._weights


class KRR(BaseIO):
    """Kernel Ridge Regression model. Only supports sparse GPR
    training for the moment.

    Parameters
    ----------
    weights : np.array
        weights of the model

    kernel : Kernel
        kernel class used to train the model

    X_train : SparsePoints
        reference samples used for the training

    self_contributions : dictionary
        map atomic number to the property baseline, e.g. isolated atoms
        energies when the model has been trained on total energies.

    description : string
        User-defined string used to describe the model for future reference

    units : dict
        Energy and length units used by the model (default: eV and Å (aka AA),
        same as used in ASE)
    """

    def __init__(
        self,
        weights,
        kernel,
        X_train,
        self_contributions,
        description="KRR potential model",
        units=None,
    ):
        # Weights of the krr model
        self.weights = weights
        self.kernel = kernel
        self.X_train = X_train
        self.self_contributions = self_contributions
        self.target_type = kernel.target_type
        self.description = description
        if units is None:
            units = dict()
        if "energy" not in units:
            units["energy"] = "eV"
        if "length" not in units:
            units["length"] = "AA"
        self.units = units

    def _get_property_baseline(self, managers):
        """build total baseline contribution for each prediction"""
        if self.target_type == "Structure":
            Y0 = np.zeros(len(managers))
            for i_manager, manager in enumerate(managers):
                if isinstance(manager, ase.Atoms):
                    numbers = manager.get_atomic_numbers()
                    for sp in numbers:
                        Y0[i_manager] += self.self_contributions[sp]
                else:
                    for at in manager:
                        Y0[i_manager] += self.self_contributions[at.atom_type]
        elif self.target_type == "Atom":
            n_centers = 0
            for manager in managers:
                n_centers += len(manager)
            Y0 = np.zeros(n_centers)
            i_center = 0
            for manager in managers:
                for center in manager:
                    Y0[i_center] = self.self_contributions[center.atom_type]
                    i_center += 1
        return Y0

    def predict(self, managers, KNM=None):
        """Predict properties associated with the atomic structures in managers.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if KNM is None:
            kernel = self.kernel(managers, self.X_train, (False, False))
        else:
            if len(managers) != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(len(managers), KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            kernel = KNM
        Y0 = self._get_property_baseline(managers)
        return Y0 + np.dot(kernel, self.weights).reshape((-1))

    def predict_forces(self, managers, KNM=None):
        """Predict negative gradients w.r.t atomic positions, e.g. forces, associated with the atomic structures in managers.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if self.kernel.kernel_type != "Sparse":
            raise NotImplementedError(
                "force prediction only implemented for kernels with kernel_type=='Sparse'"
            )
        if KNM is None:
            rep = self.kernel._representation
            gradients = compute_sparse_kernel_gradients(
                rep,
                self.kernel._kernel,
                managers.managers,
                self.X_train._sparse_points,
                self.weights.reshape((1, -1)),
            )
        else:
            n_atoms = 0
            for manager in managers:
                n_atoms += len(manager)
            if 3 * n_atoms != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(3 * n_atoms, KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            gradients = np.dot(KNM, self.weights).reshape((-1, 3))

        return -gradients

    def predict_stress(self, managers, KNM=None):
        """Predict gradients w.r.t cell parameters, e.g. stress, associated with the atomic structures in managers.
        The stress is returned using the Voigt order: xx, yy, zz, yz, xz, xy.

        Parameters
        ----------
        managers : AtomsList
            list of atomic structures with already computed features compatible
            with representation in kernel
        KNM : np.array, optional
            precomputed sparse kernel matrix

        Returns
        -------
        np.array
            predictions
        """
        if self.kernel.kernel_type != "Sparse":
            raise NotImplementedError(
                "stress prediction only implemented for kernels with kernel_type=='Sparse'"
            )

        if KNM is None:
            rep = self.kernel._representation
            neg_stress = compute_sparse_kernel_neg_stress(
                rep,
                self.kernel._kernel,
                managers.managers,
                self.X_train._sparse_points,
                self.weights.reshape((1, -1)),
            )
        else:
            if 6 * len(managers) != KNM.shape[0]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(6 * len(managers), KNM.shape[0])
                )
            elif self.X_train.size() != KNM.shape[1]:
                raise ValueError(
                    "KNM size mismatch {}!={}".format(self.X_train.size(), KNM.shape[1])
                )
            neg_stress = np.dot(KNM, self.weights).reshape((len(managers), 6))

        return -neg_stress

    def get_weights(self):
        return self.weights

    def _get_init_params(self):
        init_params = dict(
            weights=self.weights,
            kernel=self.kernel,
            X_train=self.X_train,
            self_contributions=self.self_contributions,
            description=self.description,
            units=self.units.copy(),
        )
        return init_params

    def _set_data(self, data):
        super()._set_data(data)

    def _get_data(self):
        return super()._get_data()

    def get_representation_calculator(self):
        return self.kernel._rep


def _get_kernel_strides(frames):
    """Get strides for total-energy/gradient kernels of the given structures

    Parameters
    ----------
    frames
        List of structures each indicating the number of atoms

    This assumes the final kernel will be stored in GAP energy-force
    format, i.e.  Nstructures rows for total-energy (summed) kernels and
    3*Natoms_total rows for gradients w.r.t. atomic positions.

    Returns
    -------
    int
        the number of structures
    int
        the number of gradient entries (== 3 * the total number of atoms)
    np.array(int)
        strides for assigning the gradient entries for each structure
    """
    Nstructures = len(frames)
    Ngrad_stride = [0]
    Ngrads = 0
    for frame in frames:
        n_at = len(frame)
        Ngrad_stride.append(n_at * 3)
        Ngrads += n_at * 3
    Ngrad_stride = np.cumsum(Ngrad_stride) + Nstructures
    return Nstructures, Ngrads, Ngrad_stride


def _compute_kernel_single(i_frame, frame, representation, X_sparse, kernel):
    """Compute GAP kernel of the (new) structure against the sparse points

    Parameters
    ----------
    i_frame
        frame index (ignored???)
    frame
        New structure to compute kernel for
    representation
        RepresentationCalculator to use for the structures
    X_sparse
        Sparse points to compute kernels against
    kernel
        Kernel object to use

    Returns
    -------
    en_row
        Energy kernel row
    grad_rows
        Gradient of the kernel
    """
    feat = representation.transform([frame])
    en_row = kernel(feat, X_sparse)
    grad_rows = kernel(feat, X_sparse, grad=(True, False))
    return en_row, grad_rows


def compute_KNM(frames, X_sparse, kernel, soap):
    """Compute GAP kernel of the (new) structures against the sparse points

    Parameters
    ----------
    frames
        New structures to compute kernel for
    representation
        RepresentationCalculator to use for the structures
    X_sparse
        Sparse points to compute kernels against
    kernel
        Kernel object to use

    Returns
    -------
    K_NM: np.array
        Summed total-energy kernel stacked with the atom-position gradient of the kernel

    Notes
    -----
    This function can take quite a long time to run.  To get a progress bar,
    you can wrap the `frames` parameter in a [tqdm]_ object like this:

    .. code-block:: python

        from tqdm.notebook import tqdm # for Jupyter
        #from tqdm import tqdm # on the command line
        K_NM = compute_KNM(
            tqdm(frames, desc="compute KNM", leave=False),
            X_sparse,
            kernel,
            soap
        )

    .. [tqdm] https://github.com/tqdm/tqdm
    """
    # If frames has been wrapped in a tqdm, use the underlying iterable
    # so as not to "use up" the progress bar prematurely
    if hasattr(frames, "iterable"):
        Nstructures, Ngrads, Ngrad_stride = _get_kernel_strides(frames.iterable)
    else:
        Nstructures, Ngrads, Ngrad_stride = _get_kernel_strides(frames)
    KNM = np.zeros((Nstructures + Ngrads, X_sparse.size()))
    for i_frame, frame in enumerate(frames):
        en_row, grad_rows = _compute_kernel_single(
            i_frame, frame, soap, X_sparse, kernel
        )
        KNM[Ngrad_stride[i_frame] : Ngrad_stride[i_frame + 1]] = grad_rows
        KNM[i_frame] = en_row
    return KNM


def train_gap_model(
    kernel,
    frames,
    KNM_,
    X_sparse,
    y_train,
    self_contributions,
    solver="Normal",
    grad_train=None,
    lambdas=None,
    jitter=1e-8,
):
    """
    Defines the procedure to train a SOAP-GAP model [1]:
    .. math::
        Y(A) = \sum_{i \in A} y_{a_i}(X_i),
    where :math:`Y(A)` is the predicted property function associated with the
    atomic structure :math:`A$, :math:`i` and :math:`a_i` are the index and
    species of the atoms in structure :math:`X` and :math:`y_a(A_i)` is the
    atom centered model that depends on the central atomic species.
    The individual predictions are given by:
    .. math::
        y_{a_i(A_i) = \sum_m^{M} \alpha_m \delta_{b_m a_i} k(A_i,T_m),
    where :math:`k(\cdot,\cdot)` is a kernel function, :math:`\alpha_m` are the
    weights of the model and :math:`b_m is the atom type associated with the
    sparse point :math:`T_m`.
    Hence a kernel element for the target property :math:`Y(A)` is given by:
    .. math::
        KNM_{Am} = \sum_{i \in A} \delta_{b_m a_i} k(A_i,T_m)
    and for :math:`\vec{\nabla}_iY(A)`:
    .. math::
       KNM_{A_{i}m} = \delta_{b_m a_i} \sum_{j \in A_i} \vec{\nabla}_i k(A_j,T_m)
    The training is given by:
    .. math::
        \bm{\alpha} =  K^{-1} \bm{Y},
    where :math:`K` is given by:
    .. math::
        K = K_{MM} + K_{MN} \Lambda^{-2} K_{NM},
    :math:`\bm{Y}=K_{MN} \Lambda^{-2} \bm{y}$, :math:`\bm{y}` the training
    targets and :math:`\Lambda` the regularization matrix.
    The regularization matrix is chosen to be diagonal:
    .. math::
        \Lambda^{-1}_{nn} = \delta_{nn} * lambdas[0] / \sigma_{\bm{y}} * np.sqrt(Natoms)
    for the targets and
    .. math::
        \Lambda^{-1}_{nn} = \delta_{nn} * lambdas[1] / \sigma_{\bm{y}},
    for the derivatives of the targets w.r.t. the atomic positions and
    :math:`\sigma_{\bm{y}}` is the standard deviation of the target property
    (not derivatives).

    Parameters
    ----------
    kernel : Kernel
        SparseKernel to compute KMM and initialize the model. It was used to
        build KNM_.
    frames : list(ase.Atoms)
        Training structures
    KNM_ : np.array
        kernel matrix to use in the training, typically computed with:
        KNM = kernel(managers, X_sparse)
        KNM_down = kernel(managers, X_sparse, grad=(True, False))
        KNM = np.vstack([KNM, KNM_down])
        when training with derivatives.
    X_sparse : SparsePoints
        basis samples to use in the model's interpolation
    y_train : np.array
        reference property
    self_contributions : dictionary
        map atomic number to the property baseline, e.g. training on
        total energies is not very recommended so one would provide
        the corresponding isolated atom energies so that the model
        can be trained on the corresponding formation energies and
        still predict total energies.
    grad_train : np.array, optional
        derivatives of y_train w.r.t. to the atomic motion, e.g.
        minus interatomic forces, by default None
    lambdas : list/tuple, optional
        regularisation parameter for the training, i.e. lambdas[0] -> property
        and lambdas[1] -> gradients of the property, by default None
    jitter : double, optional
        small jitter for the numerical stability of solving the linear system,
        by default 1e-8
    solver: string, optional
        method used to solve the sparse KRR equations.
        "Normal" uses a least-squares solver for the normal equations:
           (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y
        "RKHS" computes first the reproducing kernel features by diagonalizing K_MM
        and computing P_NM = K_NM @ U_MM @ Lam_MM^(-1.2) and then solves the linear
        problem for those (which is usually better conditioned)
           (P_NM.T@P_NM + 1)^(-1) P_NM.T@Y
        by default, "Normal"

    Returns
    -------
    KRR
        a trained model that can predict the property and its gradients

    .. [1] Ceriotti, M., Willatt, M. J., & Csányi, G. (2018).
        Machine Learning of Atomic-Scale Properties Based on Physical Principles.
        In Handbook of Materials Modeling (pp. 1–27). Springer, Cham.
        https://doi.org/10.1007/978-3-319-42913-7_68-1
    """
    KMM = kernel(X_sparse)
    Y = y_train.reshape((-1, 1)).copy()
    KNM = KNM_.copy()
    n_centers = Y.shape[0]
    Natoms = np.zeros(n_centers)
    Y0 = np.zeros((n_centers, 1))
    for iframe, frame in enumerate(frames):
        Natoms[iframe] = len(frame)
        for sp in frame.get_atomic_numbers():
            Y0[iframe] += self_contributions[sp]
    Y = Y - Y0
    delta = np.std(Y)
    # lambdas[0] is provided per atom hence the '* np.sqrt(Natoms)'
    # the first n_centers rows of KNM are expected to refer to the
    #  property
    KNM[:n_centers] /= lambdas[0] / delta * np.sqrt(Natoms)[:, None]
    Y /= lambdas[0] / delta * np.sqrt(Natoms)[:, None]

    if grad_train is not None:
        KNM[n_centers:] /= lambdas[1] / delta
        F = grad_train.reshape((-1, 1)).copy()
        F /= lambdas[1] / delta
        Y = np.vstack([Y, F])

    ssolver = SparseGPRSolver(
        KMM, regularizer=1, jitter=jitter, solver=solver, relative_jitter=False
    )  # in current implementation KMM incorporates regularization so it's better to use an absolute jitter value
    ssolver.fit(KNM, Y)
    model = KRR(ssolver._weights, kernel, X_sparse, self_contributions)

    del KNM, KMM, solver

    return model
