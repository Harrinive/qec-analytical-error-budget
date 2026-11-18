import qutip as qt
import numpy as np

from chencrafts.cqed.qt_helper import dnorm
from chencrafts.cqed.proc_repr import (
    pauli_basis, to_orth_chi, orth_chi_to_choi
)

from typing import List, Tuple, Dict, Literal
from . import settings
# conversions between the representations of a map, applied to every entry
# of an array of maps
to_choi_vec = np.vectorize(qt.to_choi)
to_chi_vec = np.vectorize(qt.to_chi)
to_orth_chi_vec = np.vectorize(to_orth_chi)
to_super_vec = np.vectorize(qt.to_super)
orth_chi_to_choi_vec = np.vectorize(orth_chi_to_choi)

# label of a subspace of the physical space: the integer index of an error
# space, "L" (joint error space), or "p" (leakage subspace)
_subspace_types = Literal["L", "p"] | int

def _abstract_error_op(
    logical_dim: int, num_errors: int, which_error: int
) -> qt.Qobj:
    """
    Inclusion map of the logical space into copy ``which_error`` of the
    abstract error space, which is ``num_errors`` copies of the logical
    space.
    """
    logical_basis = [
        qt.basis(logical_dim, idx) for idx in range(logical_dim)
    ]
    error_basis = [
        qt.basis(num_errors * logical_dim, which_error * logical_dim + idx)
        for idx in range(logical_dim)
    ]
    
    logi_error_map = [
        error_state * logical_state.dag()
        for error_state, logical_state in zip(error_basis, logical_basis)
    ]
    
    return sum(logi_error_map)

def abstract_encoders(num_subspaces: int, logical_dim: int = 2) -> np.ndarray:
    r"""
    Inclusion maps of the logical space into each copy of the abstract
    error space.

    The abstract error space is ``num_subspaces`` copies of the logical
    space, of dimension :math:`kd`. The map :math:`\Pi_a` sends logical
    basis state :math:`i` to basis state :math:`a d + i`. They take the
    place of the encoders :math:`E_a` for a map that already contains the
    stacked encoder :math:`S=\sum_a E_a\Pi_a^\dagger`
    (:func:`stacked_encoder`), because :math:`S\Pi_a = E_a`.

    Parameters
    ----------
    num_subspaces : int
        Number :math:`k` of logical subspaces (error spaces).
    logical_dim : int
        Logical dimension :math:`d`.

    Returns
    -------
    np.ndarray
        Object array of ``num_subspaces`` Qobj, each of dims
        ``[[k d], [d]]``.
    """
    encs = np.ndarray(num_subspaces, dtype=qt.Qobj)
    for idx in range(num_subspaces):
        encs[idx] = _abstract_error_op(logical_dim, num_subspaces, idx)
    return encs

def stacked_encoder(encoders: np.ndarray | List[qt.Qobj]) -> qt.Qobj:
    r"""
    Encoder of the abstract error space into the physical space,
    :math:`S=\sum_a E_a\Pi_a^\dagger`, where :math:`E_a` is the encoder of
    error space :math:`a` and :math:`\Pi_a` the inclusion map of
    :func:`abstract_encoders`.

    Column block :math:`a` of :math:`S` is :math:`E_a`, so
    :math:`S\Pi_a=E_a`. A dynamical map composed with
    :math:`\mathcal S(\cdot)=S\cdot S^\dagger` determines its action on
    every operator on the joint error space.

    Parameters
    ----------
    encoders : np.ndarray | List[qt.Qobj]
        Physical encoders, each of dims ``[physical, [d]]``.

    Returns
    -------
    qt.Qobj
        Operator of dims ``[physical, [k d]]``.
    """
    len_errors = len(encoders)
    dim_logical = encoders[0].dims[1][0]
    return sum(
        enc * _abstract_error_op(dim_logical, len_errors, idx).dag()
        for idx, enc in enumerate(encoders)
    )

def _constr_enc_by_subspace(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    subspace_type: _subspace_types,
    return_proj: bool = True,
    enc_from_error_space: bool = False,
) -> qt.Qobj:
    """
    Projector onto a subspace of the physical space (``return_proj`` True)
    or encoder into it (False): an error space (integer index), the joint
    error space (``"L"``), or the leakage subspace (``"p"``, projector
    only).

    The encoder of error space ``a`` acts on the logical space, or on the
    abstract error space if ``enc_from_error_space`` is True; the encoder of
    ``"L"`` acts on the abstract error space and requires
    ``enc_from_error_space`` to be True.
    """
    if isinstance(subspace_type, int):
        enc = encoders[subspace_type]
        
        if enc_from_error_space:
            len_encs = len(encoders)
            dim_logical = encoders[0].dims[1][0]
            error_to_logi_op = _abstract_error_op(
                dim_logical, len_encs, subspace_type
            ).dag()
            enc = enc * error_to_logi_op
            
        if return_proj:
            return enc * enc.dag()
        else:
            return enc
        
    elif subspace_type == "L":
        if not enc_from_error_space:
            raise ValueError("enc_from_error_space should be True for 'L' subspace.")
        
        if return_proj:
            return sum([enc * enc.dag() for enc in encoders])
        else:
            # # method 1: stack the encoders, i.e., the encoded logical states
            # # are different in different subspaces
            # enc_dim = encoders[0].dims[0]       # a list of integers
            # logi_dim = encoders[0].dims[1][0]   # an integer
            # len_enc = len(encoders)

            # stacked_enc = np.concatenate([
            #     enc.full() for enc in encoders
            # ], axis=1)
            # stacked_enc = qt.Qobj(stacked_enc, dims=[enc_dim, [logi_dim * len_enc]])
            # return stacked_enc
            
            # method 2: map the logical states back to the corresponding error space
            len_errors = len(encoders)
            dim_logical = encoders[0].dims[1][0]
            new_encoders = [
                enc * _abstract_error_op(
                    dim_logical, len_errors, idx
                ).dag()
                for idx, enc in enumerate(encoders)
            ]
            return sum(new_encoders)
        
    elif subspace_type == "p":
        # the leakage subspace has no encoder, so enc_from_error_space is
        # not used
        if return_proj == False:
            raise ValueError("The leakage subspace does not support return_proj = False.")
        
        proj_L = sum([enc * enc.dag() for enc in encoders])
        return qt.qeye_like(proj_L) - proj_L
    
    else:
        raise ValueError("The subspace_type should be either 'L', 'p', or an int.")

def _constr_enc_superop_by_subspace(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    subspace_type: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"], 
    return_type: Literal["enc", "proj", "dec"] = "enc",
    enc_from_error_space_if_possible: bool = False,
) -> qt.Qobj:
    r"""
    Projector map onto one component of an operator on the physical space
    (``return_type`` ``"proj"``), in the notation of
    :func:`component_projector_map`, or the corresponding encoding map
    (``"enc"``) or decoding map (``"dec"``).

    The encoding map takes operators on the logical space, or on the
    abstract error space if ``enc_from_error_space_if_possible`` is True,
    to operators on the physical space; the decoding map is its adjoint.
    A component that involves the leakage subspace has a projector map
    only.
    """            
    if isinstance(subspace_type, tuple):
        if subspace_type[0] != subspace_type[1]:
            # the two encoders must then act on the same space, or
            # qt.sprepost(enc_0, enc_1.dag()) and qt.sprepost(enc_1,
            # enc_0.dag()) differ in dimension
            
            if (
                isinstance(subspace_type[0], int) 
                or subspace_type[0] == "L"
            ) and (
                isinstance(subspace_type[1], int)
                or subspace_type[1] == "L"
            ):
                # both lie in the joint error space, so both encoders act
                # on the abstract error space
                if return_type != "proj" and not enc_from_error_space_if_possible:
                    raise ValueError("enc_from_error_space_if_possible must be True "
                                     "for different logical subspaces.")

            if subspace_type[0] == "p" or subspace_type[1] == "p":
                # the leakage subspace has a projector only
                if return_type != "proj":
                    raise ValueError("return_type must be 'proj' for leakage subspaces.")
        
        enc_0 = _constr_enc_by_subspace(
            encoders, subspace_type[0], 
            return_proj = (return_type == "proj"),
            enc_from_error_space = (
                enc_from_error_space_if_possible 
                or not isinstance(subspace_type[0], int)
            )
        )
            
        enc_1 = _constr_enc_by_subspace(
            encoders, subspace_type[1], 
            return_proj = (return_type == "proj"),
            enc_from_error_space = (
                enc_from_error_space_if_possible 
                or not isinstance(subspace_type[1], int)
            )
        )
        
        # the two terms below are equal if the two subspaces are the same
        if subspace_type[0] == subspace_type[1]:
            prefactor = 1/2     
        else:
            prefactor = 1   
        
        if return_type == "enc" or return_type == "proj":
            return prefactor * (
                qt.sprepost(enc_0, enc_1.dag())
                + qt.sprepost(enc_1, enc_0.dag())
            )
        else:
            return prefactor * (
                qt.sprepost(enc_0.dag(), enc_1)
                + qt.sprepost(enc_1.dag(), enc_0)
            )
            
    # "LD" or "LC": a sum over pairs of error spaces
    if return_type != "proj" and not enc_from_error_space_if_possible:
        raise ValueError("enc_from_error_space_if_possible must be True "
                         "for 'LD' or 'LC' subspace.")
    
    len_encs = len(encoders)
    superops = []
    for idx_0, idx_1 in np.ndindex(len_encs, len_encs):
        if idx_0 <= idx_1 and subspace_type == "LC":
            continue    
        if idx_0 != idx_1 and subspace_type == "LD":
            continue
        
        op = _constr_enc_superop_by_subspace(
            encoders, (idx_0, idx_1), 
            return_type,
            enc_from_error_space_if_possible
        )
        
        superops.append(op)
        
    return sum(superops)

def joint_error_space_projector(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
) -> qt.Qobj:
    r"""
    Projector onto the joint error space, the direct sum of the error
    spaces,

    .. math::
        \hat P_{\mathrm{err}} = \sum_a \hat E_a \hat E_a^\dagger ,

    with :math:`\hat E_a` the encoder of error space :math:`a`, an isometry
    from the logical space into the physical space. The error spaces are
    assumed to be orthogonal; if they are not, the sum is returned although
    it is not a projector.

    Parameters
    ----------
    encoders : np.ndarray[qt.Qobj] | List[qt.Qobj]
        Encoders of the error spaces, each of dims ``[physical, [d]]``
        with :math:`d` the dimension of the logical space, for example
        ``StateNode.ideal_encoders()``.

    Returns
    -------
    qt.Qobj
        Operator on the physical space, of dims ``[physical, physical]``.
    """
    # "L" requires enc_from_error_space=True, which does not change a
    # projector
    return _constr_enc_by_subspace(
        encoders, "L", return_proj=True, enc_from_error_space=True,
    )

def component_projector_map(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    component: Tuple[_subspace_types, _subspace_types] | Literal["LD", "LC"] = ("L", "L"),
) -> qt.Qobj:
    r"""
    Projector map onto one component of an operator on the physical space.

    Three kinds of subspaces of the physical space define the components:
    each error space :math:`a`, with projector
    :math:`\hat P_a=\hat E_a\hat E_a^\dagger`; the joint error space, with
    projector :math:`\hat P_{\mathrm{err}}=\sum_a\hat P_a`
    (:func:`joint_error_space_projector`); and the leakage subspace, with
    projector :math:`\hat P_{\mathrm{leak}}=\hat 1-\hat P_{\mathrm{err}}`.
    For two of these subspaces :math:`A` and :math:`B`, the projector map
    is

    .. math::
        \mathcal P_{AA}(\hat O) = \hat P_A \hat O \hat P_A , \qquad
        \mathcal P_{AB}(\hat O) = \hat P_A \hat O \hat P_B
            + \hat P_B \hat O \hat P_A \quad (A\neq B) .

    Three projector maps add up to the identity map:

    - ``("L", "L")``: the component within the joint error space,
    - ``("p", "p")``: the component within the leakage subspace,
    - ``("L", "p")``: the error-leakage coherence, the component between
      the joint error space and the leakage subspace.

    The component within the joint error space divides further:

    - ``(a, a)`` with an integer ``a``: the component within error space
      ``a``,
    - ``(a, b)`` with integers ``a != b``: the coherence between error
      spaces ``a`` and ``b``,
    - ``"LD"``: :math:`\sum_a \mathcal P_{aa}`, the components within the
      individual error spaces,
    - ``"LC"``: :math:`\sum_{a<b}\mathcal P_{ab}`, the coherence between
      different error spaces.

    The projector map of ``("L", "L")`` is the sum of those of ``"LD"``
    and ``"LC"``. The pair ``(a, "p")`` with an integer ``a`` gives the
    coherence between error space ``a`` and the leakage subspace. The pair
    ``(a, "L")`` is accepted, but the result is not a projector map,
    because error space ``a`` is part of the joint error space.

    In the column-stacking convention of qutip, the map
    :math:`\hat O\mapsto\hat P_A\hat O\hat P_B` has the matrix
    ``np.kron(P_B.T, P_A)``.

    Parameters
    ----------
    encoders : np.ndarray[qt.Qobj] | List[qt.Qobj]
        Encoders of the error spaces, each of dims ``[physical, [d]]``
        with :math:`d` the dimension of the logical space, for example
        ``StateNode.ideal_encoders()``.
    component : tuple | Literal["LD", "LC"]
        The component, in the notation listed above. An entry of a tuple
        is ``"L"`` (joint error space), ``"p"`` (leakage subspace), or
        the integer index of an error space. The default ``("L", "L")``
        is the component within the joint error space.

    Returns
    -------
    qt.Qobj
        Superoperator on the physical space. For ``"LC"`` with a single
        error space, the sum is empty and the integer 0 is returned.
    """
    return _constr_enc_superop_by_subspace(
        encoders, component, return_type="proj",
    )

def process_block(
    process: qt.Qobj,
    init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LD", "LC"],
    final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LD", "LC"],
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    truncate_leakage_space: bool = True,
    truncate_irrelevant_logical_space_if_possible: bool = True,
) -> qt.Qobj:
    r"""
    Projected dynamical map
    :math:`\mathcal P_{\rm final}\,\mathcal E\,\mathcal P_{\rm init}`,
    with :math:`\mathcal E` the dynamical map ``process``.

    The projector map :math:`\mathcal P_{\rm init}` keeps one component of
    an operator before :math:`\mathcal E` acts, and
    :math:`\mathcal P_{\rm final}` keeps one component after it.
    
    Parameters
    ----------
    process : qt.Qobj
        The dynamical map :math:`\mathcal E`, a superoperator on the
        physical space (qutip ``super`` representation).
    init_subspace, final_subspace : tuple | Literal["LD", "LC"]
        The component kept before the map acts and the component kept
        after it, in the notation of :func:`component_projector_map`.
    init_encoders, final_encoders : np.ndarray[qt.Qobj] | List[qt.Qobj]
        Encoders of the error spaces before the map acts and of the
        error spaces after it.
    truncate_leakage_space : bool
        If True (default) and neither component involves the leakage
        subspace, the result acts on operators on the logical space or on
        the abstract error space instead of the physical space: the
        encoding map replaces :math:`\mathcal P_{\rm init}` and the
        decoding map replaces :math:`\mathcal P_{\rm final}`.
    truncate_irrelevant_logical_space_if_possible : bool
        If True (default), a component within a single error space
        ``(a, a)`` is written on the logical space, without the other
        error spaces. Otherwise, and for every other component, it is
        written on the abstract error space.

    Returns
    -------
    qt.Qobj
        The projected dynamical map as a superoperator.
    """
    if truncate_leakage_space and (
        init_subspace[0] != "p"
        and init_subspace[1] != "p"
        and final_subspace[0] != "p"
        and final_subspace[1] != "p"
    ):
        init_return_type = "enc"
        final_return_type = "dec"
    else:
        init_return_type = "proj"
        final_return_type = "proj"
    
    init_superop = _constr_enc_superop_by_subspace(
        init_encoders, init_subspace, return_type=init_return_type,
        enc_from_error_space_if_possible=(
            not truncate_irrelevant_logical_space_if_possible
            or init_subspace[0] != init_subspace[1]
            or init_subspace[0] == "L"
            or init_subspace[1] == "L"
        )
    )
    final_superop = _constr_enc_superop_by_subspace(
        final_encoders, final_subspace, return_type=final_return_type,
        enc_from_error_space_if_possible=(
            not truncate_irrelevant_logical_space_if_possible
            or final_subspace[0] != final_subspace[1]
            or final_subspace[0] == "L"
            or final_subspace[1] == "L"
        )
    )
    
    return final_superop * process * init_superop

def effective_logical_process(
    process: qt.Qobj,
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
) -> np.ndarray[qt.Qobj]:
    r"""
    Effective logical maps of a dynamical map,
    :math:`\tilde{\mathcal E}_{ba}(\hat\rho)=\hat E_b^\dagger\,\mathcal E(\hat E_a\hat\rho\hat E_a^\dagger)\,\hat E_b`:
    the map encodes the logical qubit into error space :math:`a`, applies
    the dynamical map :math:`\mathcal E`, and decodes from error space
    :math:`b`.

    Parameters
    ----------
    process : qt.Qobj
        The dynamical map :math:`\mathcal E`, a superoperator on the
        physical space (qutip ``super`` representation).
    init_encoders, final_encoders : np.ndarray[qt.Qobj] | List[qt.Qobj]
        Encoders :math:`\hat E_a` of the error spaces before the map acts
        and :math:`\hat E_b` of the error spaces after it.
    repr : {"super", "choi", "chi", "orth_chi"}
        Representation of each map: the qutip ``super``, ``choi``, or
        ``chi`` representation, or the process matrix of
        ``chencrafts.cqed.proc_repr.to_orth_chi``.
    
    Returns
    -------
    np.ndarray[qt.Qobj]
        Object array of shape (number of error spaces after the map,
        number of error spaces before it); entry ``[b, a]`` is
        :math:`\tilde{\mathcal E}_{ba}`.
    """
    len_init_subspace = len(init_encoders)
    len_final_subspace = len(final_encoders)
    
    ELP = np.ndarray(
        (len_final_subspace, len_init_subspace), 
        dtype=qt.Qobj
    )
    
    for idx_final, idx_init in np.ndindex(*ELP.shape):
        proc = process_block(
            process,
            (idx_init, idx_init),
            (idx_final, idx_final),
            init_encoders,
            final_encoders,
            truncate_leakage_space = True,
            truncate_irrelevant_logical_space_if_possible = True,
        )
        
        ELP[idx_final, idx_init] = proc
        
    if repr == "super":
        return ELP
    elif repr == "choi":
        return to_choi_vec(ELP)
    elif repr == "chi":
        return to_chi_vec(ELP)
    elif repr == "orth_chi":
        return to_orth_chi_vec(ELP)
    else:
        raise ValueError(
            "The type of effective logical process should be "
            "either 'super', 'choi', 'chi', or 'orth_chi'."
        )

def target_process_for_dnorm(
    process: qt.Qobj,
    I_prob: float | None = None,
) -> qt.Qobj:
    r"""
    Scaled identity map :math:`p\,\mathcal I` on a qubit, the map with
    which a map on the logical qubit is compared in a diamond norm.
    
    Parameters
    ----------
    process : qt.Qobj
        Map on a qubit in the ``super``, ``choi``, ``chi``, or
        ``orth_chi`` representation.
    I_prob : float | None
        The factor :math:`p`. If None, :math:`p=\mathrm{Tr}(J)/2`, with
        :math:`J` the Choi matrix of ``process``, which is 1 for a
        trace-preserving map.

    Returns
    -------
    qt.Qobj
        :math:`p\,\mathcal I` in the representation of ``process``.
    """
    assert process.shape == (4, 4), "Only support single qubit process for now."
    
    if I_prob is None:
        # the Choi matrix and the orth_chi matrix are related by a unitary
        # transformation; the Choi trace of a trace-preserving map is 2
        if process.superrep == "orth_chi":
            choi = orth_chi_to_choi(process)
        else:
            choi = qt.to_choi(process)
        I_prob = choi.tr()
    else:
        # I_op below is the identity divided by sqrt(2)
        I_prob = I_prob * 2 
        
    I_op = pauli_basis[0]
        
    I_super = qt.sprepost(I_op, I_op) * I_prob
    
    if process.superrep == "orth_chi":
        I_super = to_orth_chi(I_super)
    elif process.superrep == "choi":
        I_super = qt.to_choi(I_super)
    elif process.superrep == "chi":
        I_super = qt.to_chi(I_super)
    elif process.superrep == "super":
        pass
    else:
        raise ValueError("The superoperator representation of the process should be "
                         "either 'super', 'choi', 'chi', or 'orth_chi'.")

    return I_super

def process_block_dnorm(
    process: qt.Qobj,
    init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LD", "LC"],
    final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LD", "LC"],
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    solver: str | None = None,
    **dnorm_kwargs,
):
    r"""
    Diamond norm of a projected dynamical map
    :math:`\mathcal P_{\rm final}\,\mathcal E\,\mathcal P_{\rm init}`,
    with :math:`\mathcal E` the dynamical map ``process``.

    The projector map :math:`\mathcal P_{\rm init}` keeps one component of
    an operator before :math:`\mathcal E` acts, and
    :math:`\mathcal P_{\rm final}` keeps one component after it. If
    neither component involves the leakage subspace, the map is written on
    the logical space or on the abstract error space
    (:func:`process_block`).

    Parameters
    ----------
    process : qt.Qobj
        The dynamical map :math:`\mathcal E`, a superoperator on the
        physical space (qutip ``super`` representation).
    init_subspace, final_subspace : tuple | Literal["LD", "LC"]
        The component kept before the map acts and the component kept
        after it, in the notation of :func:`component_projector_map`.
    init_encoders, final_encoders : np.ndarray[qt.Qobj] | List[qt.Qobj]
        Encoders of the error spaces before the map acts and of the
        error spaces after it.
    solver : str | None
        Solver of the semidefinite program of the diamond norm
        (``"SCS"``, ``"MOSEK"``, ``"CVXOPT"``), passed to
        :func:`chencrafts.cqed.qt_helper.dnorm`. If None (default) and no
        further argument is given, the norm is ``Qobj.dnorm()`` of the
        projected dynamical map, with the default solver of qutip.
    **dnorm_kwargs
        Further arguments of :func:`chencrafts.cqed.qt_helper.dnorm`
        (``cp_atol``, ``cp_only``, ``sparse``, ...). If any is given,
        that function evaluates the norm, with its default solver if
        ``solver`` is None.

    Returns
    -------
    float
        :math:`\|\mathcal P_{\rm final}\,\mathcal E\,\mathcal P_{\rm init}\|_\diamond`,
        or one of the other values that
        :func:`chencrafts.cqed.qt_helper.dnorm` returns for the options
        given (for example ``nan`` with ``cp_only``).
    """
    block = process_block(
        process, init_subspace, final_subspace, init_encoders, final_encoders,
        truncate_leakage_space = True,
        truncate_irrelevant_logical_space_if_possible = True,
    )

    # Qobj.dnorm takes the map as it is, whereas qt_helper.dnorm first
    # modifies it
    if solver is None and not dnorm_kwargs:
        return block.dnorm()

    # qt_helper.dnorm decides between the closed formulas and the
    # semidefinite program
    if solver is not None:
        dnorm_kwargs["solver"] = solver
    return dnorm(block, **dnorm_kwargs)
    
