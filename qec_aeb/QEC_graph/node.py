__all__ = [
    'StateNode',
    'StateEnsemble',
]

import qutip as qt
import numpy as np
from copy import deepcopy
from warnings import warn

from chencrafts.cqed.qt_helper import (
    projector_w_basis,
    normalization_factor,
    trans_by_kets,
    orthonormalize,
)
from .utils import (
    abstract_encoders,
    stacked_encoder,
    joint_error_space_projector,
    effective_logical_process,
    target_process_for_dnorm,
    leakage_process,
    truncate_leakage_process,
    bound_fidelity,
    to_choi_vec,
    to_chi_vec,
    to_orth_chi_vec,
    to_super_vec,
)
from . import settings

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Literal
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .edge import Edge

MeasurementRecord = List[Tuple[int, ...]]

class TerminationError(Exception):
    """
    Error raised when a quantity of the effective logical map (the map
    itself, its process fidelity, its diamond norm) is requested from a
    terminated node, which has no effective logical map.
    """
    def __init__(self, message: str = "Operations not allowed on terminated node"):
        self.message = message
        super().__init__(self.message)


class NodeBase(ABC):
    """
    Node of an evolution graph; it represents the state after a number of
    steps.
    """
    # the state, a density matrix whose trace is the probability of the
    # measurement record
    state: qt.Qobj
    
    # dynamical map from the initial node to this node, applied to encoded
    # initial states only: E S(.)S^dag, with E the composed dynamical maps
    # of the steps and S the stacked encoder of the initial node. Its
    # operand lives on the abstract error space of the initial node, so its
    # matrix is (d_phys^2) x (2k)^2 instead of d_phys^4.
    encoded_process: qt.Qobj
    
    # encoders of the error spaces of the initial node, from the logical
    # qubit into the physical space
    init_encoders: np.ndarray[qt.Qobj]
    
    index: int
    
    # the edge that ends at this node; a tree node has exactly one, and the
    # initial node has None
    in_edge: "Edge | None" = None
    
    def __init__(
        self,
    ):
        self.out_edges: List["Edge"] = []
    
    @property
    @abstractmethod
    def fidelity(self) -> float:
        """
        Overlap of the state with the ideal states of the node.
        """
        pass
    
    def assign_index(self, index: int) -> None:
        """Set the index of the node in its graph."""
        self.index = index
    
    def to_nx(self) -> Tuple[int, Dict[str, Any]]:
        """
        The node as a ``networkx`` node: its index and a dictionary of
        attributes.
        """
        return (
            self.index,
            {
                "state": self,
            }
        )

    @abstractmethod
    def deepcopy(self):
        """
        Copy of the node with copies of its state and ideal logical states
        and without its edges, which would make the copy reference the
        whole graph.
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    def add_out_edges(self, edge) -> None:
        """Record an edge that starts from this node."""
        self.out_edges.append(edge)
    
    @abstractmethod
    def clear_evolution_data(self):
        """Delete the state and the other results of an evolution."""
        pass
    
    def expect(self, op: qt.Qobj) -> float:
        """
        Expectation value of an operator in the state of the node, which is
        not normalized by the probability of the node.
        """
        return qt.expect(op, self.state)
    
    @abstractmethod
    def accept(self, **kwargs):
        """
        Store the result of the step that ends at this node, replacing the
        state that the node holds.
        """
        pass
    
    @abstractmethod
    def join(self, **kwargs):
        """
        Add the result of a step to the state that the node holds.
        """
        pass
    
    def effective_logical_process(self):
        """
        Effective logical maps from the initial node to this node.
        """
        pass
    
    def fidelity_by_process(self):
        """
        Process fidelity to the identity map of the effective logical maps
        from the initial node to this node.
        """
        pass
    
    def error_rate_by_process(self):
        """
        Error rate of the effective logical maps from the initial node to
        this node, compared with the identity map.
        """
        pass

    
class StateNode(NodeBase):
    r"""
    Node that represents the state at :math:`(n, \mathbf r)`, after
    :math:`n` steps with measurement record :math:`\mathbf r`.

    Besides the state, the node holds the measurement record and the ideal
    logical states: one pair for each error space at :math:`(n, \mathbf r)`.
    """
    # whether each pair of ideal logical states is orthonormalized, and the
    # method: "GS" (Gram-Schmidt) or "symm" (symmetric)
    ORTHOGONALIZE_LOGICAL_STATES = True
    ORTHOGONALIZE_METHOD: Literal["GS", "symm"] = "symm"
    
    meas_record: MeasurementRecord
    
    # amplitudes of the encoded state on logical 0 and logical 1
    _prob_amp_01: Tuple[float, float]
    
    # ideal logical states before orthonormalization, an array of shape
    # (k, 2): entry [a, i] is logical state i of error space a
    _raw_ideal_logical_states: np.ndarray[qt.Qobj]
    # effective logical maps as Choi matrices, kept after the first
    # evaluation
    _effective_logical_process: np.ndarray[qt.Qobj] | None
    
    # a terminated node is not evolved further. It represents the state for
    # an unexpected measurement record, one that no correctable error can
    # produce; it stays in every later ensemble.
    terminated: bool = False
    
    # set once the warning about a terminated node with a nonzero fidelity
    # has been issued
    term_fid_warning_issued = False
    
    # probability of each error space of the node, from the product of the
    # Choi traces of the step-wise effective logical maps that lead to it
    # (EvolutionTree.traverse_and_approx). The sum agrees to first order
    # with the probability of the node.
    traj_prob: np.ndarray[float] | None = None
    
    # product of the step-wise effective logical maps that lead to each
    # error space of the node (EvolutionTree.traverse_and_approx)
    _accum_logical_process: np.ndarray[qt.Qobj] | None = None

    def accept(
        self, 
        meas_record: MeasurementRecord,
        state: qt.Qobj,
        prob_amp_01: Tuple[float, float],
        raw_ideal_logical_states: np.ndarray[qt.Qobj],
        encoded_process: qt.Qobj,
        init_encoders: np.ndarray[qt.Qobj],
        **kwargs,
    ):
        r"""
        Store the result of the step that ends at this node, replacing the
        state that the node holds.

        Parameters
        ----------
        meas_record : MeasurementRecord
            The measurement record.
        state : qt.Qobj
            The state after the step, a density matrix whose trace is the
            probability of the measurement record.
        prob_amp_01 : Tuple[float, float]
            Amplitudes of the encoded state on logical 0 and logical 1.
        raw_ideal_logical_states : np.ndarray[qt.Qobj]
            Array of shape ``(k, 2)`` of kets: the two ideal logical
            states of each of the :math:`k` error spaces, before
            orthonormalization.
        encoded_process : qt.Qobj
            :math:`\mathcal E\,\mathcal S`, with :math:`\mathcal E` the
            composed dynamical maps of the steps from the initial node to
            this node, :math:`\mathcal S(\cdot)=\hat S\cdot\hat S^\dagger`,
            and :math:`\hat S` the stacked encoder of the initial node
            (``utils.stacked_encoder``). It is a superoperator (qutip
            ``super`` representation) from operators on the abstract error
            space of the initial node to operators on the physical space.
        init_encoders : np.ndarray[qt.Qobj]
            Encoders of the error spaces of the initial node.
        """
        for ideal_state in raw_ideal_logical_states.ravel():
            assert ideal_state.type == "ket"
        assert np.allclose(np.sum(np.abs(prob_amp_01)**2), 1)
        assert encoded_process.type == "super" and encoded_process.superrep == "super"

        # the encoded process maps into the physical space; its operand
        # lives on the abstract error space of the initial node
        subsys_dims = state.dims[0]
        assert encoded_process.dims[0] == [subsys_dims] * 2
        for ideal_state in raw_ideal_logical_states.ravel():
            assert ideal_state.dims[0] == subsys_dims

        self.meas_record = meas_record
        self.state = state
        self._prob_amp_01 = prob_amp_01
        self._raw_ideal_logical_states = raw_ideal_logical_states
        self.encoded_process = encoded_process
        self.init_encoders = init_encoders
        
        # the effective logical maps of the previous state no longer apply
        self._effective_logical_process = None      
        
    @property
    def prob_amp_01(self) -> Tuple[float, float]:
        """
        Amplitudes of the encoded state on logical 0 and logical 1.

        Setting them normalizes the pair. If the node is not terminated
        and has a single error space, the state of the node is replaced by
        the pure encoded state with these amplitudes; otherwise the state
        is kept.
        """
        return self._prob_amp_01
        
    @prob_amp_01.setter
    def prob_amp_01(self, prob_amp_01: Tuple[float, float]):
        self._prob_amp_01 = prob_amp_01 / np.sqrt(np.sum(np.abs(prob_amp_01)**2))

        if self.terminated:
            if settings.ISSUE_WARNING:
                warn("The probability amplitude of |0> and |1> is reset manually. "
                    "Usually it's not allowed for a terminated node. \n")
            return
        elif self._raw_ideal_logical_states.shape[0] > 1:
            if settings.ISSUE_WARNING:
                warn("The probability amplitude of |0> and |1> is reset manually. "
                    "While the state is not reset as the ideal logical states are "
                    "not unique. \n")
            return
        elif self._raw_ideal_logical_states.shape[0] == 1:
            if settings.ISSUE_WARNING:
                warn("The probability amplitude of |0> and |1> and the state are "
                    "reset manually. \n")
            if self.ORTHOGONALIZE_LOGICAL_STATES:
                logical_states = self._orthonormalize(self._raw_ideal_logical_states)
            else:
                logical_states = self._raw_ideal_logical_states

            self.state = qt.ket2dm(
                self._prob_amp_01[0] * logical_states[0, 0] 
                + self._prob_amp_01[1] * logical_states[0, 1]
            ).unit()
        else:
            if settings.ISSUE_WARNING:
                warn("The probability amplitude of |0> and |1> is reset manually, "
                    "but the situation is not expected. \n")
            return 

    def join(self, **kwargs):
        """Not supported: raises ``ValueError``."""
        raise ValueError("StateNode does not support join method.")
    
    def add_out_edges(self, edge) -> None:
        """
        Record an edge that starts from this node; a terminated node
        raises ``ValueError``.
        """
        if self.terminated:
            raise ValueError("The node is terminated and cannot have out edges.")
        
        super().add_out_edges(edge)

    @staticmethod
    def _orthonormalize(
        state_arr: np.ndarray[qt.Qobj],
    ) -> np.ndarray[qt.Qobj]:
        r"""
        The two logical states of every row of an array of shape (N, 2),
        orthonormalized with the method ``ORTHOGONALIZE_METHOD``.
        """
        method_names = {"GS": "gram_schmidt", "symm": "symmetric"}
        if StateNode.ORTHOGONALIZE_METHOD not in method_names:
            raise ValueError(
                "The orthonormalization method should be either "
                "'GS' or 'symm'."
            )
        method = method_names[StateNode.ORTHOGONALIZE_METHOD]

        new_state_arr = np.empty_like(state_arr)
        for i in range(len(state_arr)):
            (
                new_state_arr[i, 0], new_state_arr[i, 1]
            ) = orthonormalize(list(state_arr[i]), method=method)
        
        return new_state_arr
    
    def fidelity_drop_by_orth(self) -> float:
        r"""
        Decrease of the fidelity caused by the orthonormalization of the
        ideal logical states,
        :math:`p\,(1-|\langle\psi_{\rm orth}|\psi_{\rm raw}\rangle|^2)`, with
        :math:`p` the probability of the node and
        :math:`|\psi_{\rm orth}\rangle`, :math:`|\psi_{\rm raw}\rangle` the
        ideal states with and without the orthonormalization.

        The node must have a single error space; the value is 0 for a
        terminated node.
        """
        if self.terminated:
            return 0

        raw_ideal_state = self._ideal_states(orthonormalize=False)
        orthonormal_ideal_state = self._ideal_states(orthonormalize=True)
        
        if not len(raw_ideal_state) == 1:
            raise ValueError("This method only works if there is only one pair "
                            "of ideal logical states.")
        
        fid = 1 - np.abs(orthonormal_ideal_state[0].overlap(raw_ideal_state[0]))**2
        fid *= self.probability
        
        return fid
    
    @staticmethod
    def _qobj_unit(qobj: qt.Qobj) -> qt.Qobj:
        """
        The normalized Qobj, as a function that ``np.vectorize`` accepts.
        """
        return qobj.unit()
    
    def _ideal_states(
        self,
        orthonormalize: bool,
    ) -> np.ndarray[qt.Qobj]:
        """
        Ideal state of each error space: the superposition of its two
        ideal logical states with the amplitudes ``prob_amp_01``, with or
        without orthonormalization of the pair.
        """
        if len(self._raw_ideal_logical_states) == 0:
            # no ideal logical state was kept: a zero ket
            dim = self.state.dims[0]
            return np.array([qt.Qobj(
                np.zeros(self.state.shape), 
                dims=[dim, np.ones_like(dim).astype(int).tolist()]
            )], dtype=qt.Qobj)
        
        if orthonormalize:
            orthonormal_states = self._orthonormalize(self._raw_ideal_logical_states)
            return (
                self._prob_amp_01[0] * orthonormal_states[:, 0]
                + self._prob_amp_01[1] * orthonormal_states[:, 1]
            )
        else:
            qobj_array_unit = np.vectorize(
                self._qobj_unit, otypes = [qt.Qobj]
            )
            raw_unit = qobj_array_unit(self._raw_ideal_logical_states)
            return qobj_array_unit(
                self._prob_amp_01[0] * raw_unit[:, 0] 
                + self._prob_amp_01[1] * raw_unit[:, 1]
            )

    @property
    def ideal_states(
        self,
    ) -> np.ndarray[qt.Qobj]:
        """
        Ideal state of each error space of the node: the superposition of
        its two ideal logical states with the amplitudes ``prob_amp_01``.

        Returns
        -------
        np.ndarray[qt.Qobj]
            One ket for each error space.
        """
        return self._ideal_states(self.ORTHOGONALIZE_LOGICAL_STATES)
    
    @property
    def ideal_logical_states(self) -> np.ndarray[qt.Qobj]:
        """
        Ideal logical states of the node, orthonormalized within each pair
        if ``ORTHOGONALIZE_LOGICAL_STATES`` is True.

        Returns
        -------
        np.ndarray[qt.Qobj]
            Array of shape ``(k, 2)``: entry ``[a, i]`` is logical state
            :math:`i` of error space :math:`a`.
        """
        if self.ORTHOGONALIZE_LOGICAL_STATES:
            return self._orthonormalize(self._raw_ideal_logical_states)
        else:
            return self._raw_ideal_logical_states
    
    @property
    def ideal_projector(self) -> qt.Qobj:
        """
        Projector onto the space that the ideal states of the node span
        (``projector_w_basis`` of ``ideal_states``).
        """
        return projector_w_basis(self.ideal_states)
    
    @property
    def fidelity(self) -> float:
        r"""
        Overlap of the state with the ideal states,
        :math:`\mathrm{tr}(\hat\rho\hat P)`, with :math:`\hat\rho` the state
        of the node, which is not normalized by its probability, and
        :math:`\hat P` the projector ``ideal_projector``.
        """
        fid = ((self.state * self.ideal_projector).tr()).real
    
        if not self.term_fid_warning_issued and settings.ISSUE_WARNING:
            # the flag prevents an infinite recursion: the warning prints
            # the node, and printing the node evaluates the fidelity
            if self.terminated and fid > 1e-10:
                self.term_fid_warning_issued = True
                warn(f"Terminated branch [{self}] has a total fidelity larger than 1e-10.\n")

        return fid
    
    @property
    def probability(self) -> float:
        """
        Probability of the measurement record of the node, the trace of
        its state.
        """
        return (self.state.tr()).real

    def deepcopy(self) -> "StateNode":
        """
        Copy of the node with copies of its measurement record, state,
        ideal logical states, and amplitudes ``prob_amp_01``, and without
        its edges, which would make the copy reference the whole graph.
        The copy is terminated if the node is. It refers to the same encoded
        process as the node: that superoperator is the largest object on a
        node and is replaced, never modified, when a node accepts a step.
        """

        copied_node = StateNode()
        copied_node.meas_record = deepcopy(self.meas_record)
        copied_node.state = deepcopy(self.state)
        copied_node._raw_ideal_logical_states = deepcopy(self._raw_ideal_logical_states)
        # the ideal states and the fidelity of the copy need the amplitudes
        copied_node._prob_amp_01 = deepcopy(self._prob_amp_01)
        copied_node.terminated = self.terminated
        if "encoded_process" in vars(self):
            # the encoders of the initial node are read-only as well
            copied_node.encoded_process = self.encoded_process
            copied_node.init_encoders = self.init_encoders
            copied_node._effective_logical_process = None

        return copied_node
    
    @classmethod
    def initial_note(
        cls, 
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
    ) -> "StateNode":
        r"""
        Initial node of a tree: the encoded logical qubit before any step.

        Parameters
        ----------
        init_prob_amp_01 : Tuple[float, float]
            Amplitudes :math:`(c_0, c_1)` of the encoded state
            :math:`c_0|0_L\rangle+c_1|1_L\rangle`, which is normalized.
        logical_0, logical_1 : qt.Qobj
            Kets of the two logical states on the physical space. They are
            orthonormalized if ``ORTHOGONALIZE_LOGICAL_STATES`` is True.

        Returns
        -------
        StateNode
            Node with an empty measurement record and a single error
            space, the code space.
        """
        logical_state_arr = np.empty((1, 2), dtype=object)
        logical_state_arr[:] = [[logical_0, logical_1]]

        if cls.ORTHOGONALIZE_LOGICAL_STATES:
            logical_state_arr = cls._orthonormalize(logical_state_arr)
        state = (
            init_prob_amp_01[0] * logical_state_arr[0, 0] 
            + init_prob_amp_01[1] * logical_state_arr[0, 1]
        ).unit()
        state_dm = qt.ket2dm(state)

        init_node = cls()
        # the encoders are built from the node's own ideal logical states, so
        # those are set before accept() needs the encoders
        init_node._raw_ideal_logical_states = logical_state_arr
        init_encoders = init_node.ideal_encoders()

        # before any step the dynamical map is the identity, so the encoded
        # process is the encoding map S(.)S^dag itself
        enc = stacked_encoder(init_encoders)
        init_node.accept(
            meas_record = [],
            state = state_dm,
            prob_amp_01 = init_prob_amp_01,
            raw_ideal_logical_states = logical_state_arr,
            encoded_process = qt.sprepost(enc, enc.dag()),
            init_encoders = init_encoders,
        )
        # full_process stops its walk back through the edges here
        init_node.is_initial = True

        return init_node

    def to_nx(self) -> Tuple[int, Dict[str, Any]]:
        """
        The node as a ``networkx`` node: its index and a dictionary with
        the node, its fidelity, and its probability.
        """
        try:
            fidelity = self.fidelity
            probability = self.probability
        except AttributeError:
            fidelity = np.nan
            probability = np.nan

        return (
            self.index,
            {
                "state": self,
                "fidelity": fidelity,
                "probability": probability,
            }
        )

    def clear_evolution_data(self):
        """
        Delete the state, the ideal logical states, the measurement record
        and the encoded process of the node; an attribute that the node does
        not hold is skipped.
        """
        # each attribute on its own, so that a missing one does not keep
        # the others
        for name in (
            "state", "_raw_ideal_logical_states", "meas_record",
            "encoded_process",
        ):
            try:
                delattr(self, name)
            except AttributeError:
                pass

    def __str__(self) -> str:
        try:
            idx = self.index
        except AttributeError:
            idx = "No Index"
            
        try:
            fail = ", Terminated" if self.terminated else ""
            return (
                f"StateNode ({idx}){fail}, record {self.meas_record}, "
                + f"prob {self.probability:.3f}, fid {self.fidelity:.3f}"
            )
        except AttributeError:
            return f"StateNode ({idx})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def bloch_vector(self) -> np.ndarray:
        r"""
        Expectation values of the logical Pauli operators in the state of
        the node.

        Returns
        -------
        np.ndarray
            :math:`(\langle X\rangle, \langle Y\rangle, \langle Z\rangle,
            \langle I\rangle)` in the basis of the ideal logical states,
            not normalized by the probability of the node. Zeros for a
            terminated node; ``nan`` if the node has more than one error
            space.
        """
        if self.terminated:
            return np.zeros(4)

        if self._raw_ideal_logical_states.shape[0] > 1:
            if settings.ISSUE_WARNING:
                warn("The ideal logical states are not unique. Returned nan.\n")
            return np.nan * np.ones(4)
        
        trans = trans_by_kets(self.ideal_logical_states[0])

        X = trans * qt.sigmax() * trans.dag()
        Y = trans * qt.sigmay() * trans.dag()
        Z = trans * qt.sigmaz() * trans.dag()
        I = trans * qt.qeye(2) * trans.dag()
        op_list: List[qt.Qobj] = [X, Y, Z, I]

        dims = self.state.dims[0]
        for op in op_list:
            op.dims = [dims, dims]

        return np.array([
            self.expect(op) for op in op_list
        ])
        
    def ideal_encoders(self) -> np.ndarray[qt.Qobj]:
        r"""
        Encoders of the error spaces of the node,
        :math:`\hat E_a=|0_a\rangle\langle 0|+|1_a\rangle\langle 1|`, with
        :math:`|0_a\rangle, |1_a\rangle` the ideal logical states of error
        space :math:`a` and :math:`|0\rangle, |1\rangle` the basis of the
        logical qubit. The adjoint :math:`\hat E_a^\dagger` decodes.
        
        Returns
        -------
        np.ndarray[qt.Qobj]
            One operator of dims ``[physical, [2]]`` for each error space.
        """
        ideal_logical_states = self.ideal_logical_states
        len_subspace = ideal_logical_states.shape[0]
        encoders = np.ndarray(len_subspace, dtype=qt.Qobj)
        
        # [1, 1, ..., 1], the dims of a ket on the physical space
        subsys_ones = ideal_logical_states[0, 0].dims[1]
        logical_0 = qt.basis(2, 0)
        logical_0.dims = [[2], subsys_ones]
        logical_1 = qt.basis(2, 1)
        logical_1.dims = [[2], subsys_ones]
        
        for idx in range(len_subspace):
            encoders[idx] = (
                ideal_logical_states[idx, 0] * logical_0.dag()
                + ideal_logical_states[idx, 1] * logical_1.dag()
            )

        return encoders

    def joint_error_space_projector(self) -> qt.Qobj:
        r"""
        Projector onto the joint error space at the step and measurement
        record of this node,

        .. math::
            \hat P_{\mathrm{err}} = \sum_a \hat E_a \hat E_a^\dagger ,

        with :math:`\hat E_a` the encoder of error space :math:`a`
        (:meth:`ideal_encoders`). The complement
        :math:`\hat 1-\hat P_{\mathrm{err}}` projects onto the leakage
        subspace.

        Returns
        -------
        qt.Qobj
            Operator on the physical space.

        Raises
        ------
        AttributeError
            If the node has not been evolved, so that it has no ideal
            logical states.
        """
        # the bare name is the function imported from utils, which has
        # the same name as this method
        return joint_error_space_projector(self.ideal_encoders())

    def full_process(self) -> qt.Qobj:
        r"""
        Dynamical map from the initial node to this node,
        :math:`\mathcal E_n\cdots\mathcal E_1`, the composition of the
        dynamical maps of the steps that lead to this node. It acts on
        every operator on the physical space, whereas ``encoded_process``
        acts on encoded initial states only.

        Returns
        -------
        qt.Qobj
            Superoperator on the physical space (qutip ``super``
            representation); the identity for the initial node.

        Raises
        ------
        ValueError
            If a node on the way back to the initial node has no record of
            its incoming edge.
        """
        edges = []
        node = self
        while getattr(node, "in_edge", None) is not None:
            edges.append(node.in_edge)
            node = node.in_edge.init_state
        if not getattr(node, "is_initial", False):
            raise ValueError(
                "The path to the initial node is unknown (the tree was "
                "probably pickled before edges were recorded on nodes). "
                "Rebuild the tree to use full_process."
            )

        # compose from the initial node forward, starting from the identity
        eye_op = qt.qeye_like(self.state)
        process = qt.sprepost(eye_op, eye_op)
        for edge in reversed(edges):
            process = edge.physical_map() * process
        return process

    def __setstate__(self, state: Dict[str, Any]):
        r"""
        Restore a pickled node. A node of an actual evolution tree that
        stores the dynamical map on the whole physical space as ``process``
        gets ``encoded_process``, that map composed with the encoding map
        of the initial node, in its place. A node of a logical evolution
        tree, which has no ``init_encoders``, keeps ``process``.
        """
        if (
            "process" in state
            and "encoded_process" not in state
            and len(state.get("init_encoders", [])) > 0
        ):
            enc = stacked_encoder(state["init_encoders"])
            state["encoded_process"] = state.pop("process") * qt.sprepost(enc, enc.dag())
        self.__dict__.update(state)
    
    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ) -> np.ndarray[qt.Qobj]:
        r"""
        Effective logical maps from the initial node to this node.

        For error space :math:`a` of this node, the map encodes the
        logical qubit into the code space of the initial node, applies the
        dynamical maps of the steps that lead to this node, and decodes
        from error space :math:`a`.
        
        Parameters
        ----------
        repr : {"super", "choi", "chi", "orth_chi"}
            Representation of each map: the qutip ``super``, ``choi``, or
            ``chi`` representation, or the process matrix of
            ``chencrafts.cqed.proc_repr.to_orth_chi``.
        
        Returns
        -------
        np.ndarray[qt.Qobj]
            Object array of shape ``(k, 1)`` for a node with :math:`k`
            error spaces.

        Raises
        ------
        TerminationError
            If the node is terminated.
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        # evaluated once for each state that the node accepts
        if self._effective_logical_process is None:
            # encoded_process already contains the encoders of the initial
            # node, so the inclusion maps of the abstract error space take
            # their place here
            self._effective_logical_process = effective_logical_process(
                process = self.encoded_process,
                init_encoders = abstract_encoders(len(self.init_encoders)),
                final_encoders = self.ideal_encoders(),
                repr = "choi",
            )
            
        if repr == "super":
            return to_super_vec(self._effective_logical_process)
        elif repr == "choi":
            return self._effective_logical_process
        elif repr == "chi":
            return to_chi_vec(self._effective_logical_process)
        elif repr == "orth_chi":
            return to_orth_chi_vec(self._effective_logical_process)
        else:
            raise ValueError(f"The representation {repr} is not supported.")

    def fidelity_by_process(
        self, 
        type: Literal["avg", "etg"] = "etg",
        approx_proc: bool = False,
    ) -> np.ndarray[float]:
        r"""
        Process fidelity to the identity map of each effective logical map
        from the initial node to this node.

        The process fidelity of an effective logical map
        :math:`\tilde{\mathcal E}` to the identity map is

        .. math::
            F = \mathrm{tr}(S) / d^2 ,

        where :math:`S` is the superoperator matrix of
        :math:`\tilde{\mathcal E}` and :math:`d=2` is the dimension of the
        logical qubit. :math:`F` is linear in the map and is not divided
        by the trace that the map retains, so the values of the error
        spaces and of the measurement records add up to the process
        fidelity of the summed map. It is evaluated as
        ``qutip.process_fidelity`` of the map without a target.

        Parameters
        ----------
        type : {"avg", "etg"}
            ``"etg"`` returns :math:`F`. ``"avg"`` (average fidelity)
            raises ``NotImplementedError``, because the conversion from
            :math:`F` is not defined for a map that is not trace
            preserving.
        approx_proc : bool
            If False (default), :math:`\tilde{\mathcal E}` is the
            effective logical map of the composed dynamical maps of the
            steps (:meth:`effective_logical_process`). If True, it is the
            product of the step-wise effective logical maps
            (:meth:`accum_logical_process`).

        Returns
        -------
        np.ndarray
            Array of floats with the shape of the array of effective
            logical maps: entry ``[a, 0]`` is :math:`F` for error space
            :math:`a` of this node.

        Raises
        ------
        TerminationError
            If the node is terminated; it has no effective logical map.
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        if not approx_proc:
            realized_process = self.effective_logical_process(repr="super")
        else:
            realized_process = self.accum_logical_process(repr="super")
        
        fidelity = np.zeros(realized_process.shape)
        for idx, proc in np.ndenumerate(realized_process):
            # without a target, qutip returns tr(S)/d^2, which is linear in
            # the map; with the identity map as a target it takes matrix
            # square roots of the Choi matrices, which are accurate to
            # about 1e-8 only
            process_fidelity = qt.process_fidelity(proc)
            
            if type == "avg":
                raise NotImplementedError(
                    "Average fidelity is not implemented. As the conversion "
                    "from process fidelity isn't clear when the process isn't "
                    "CPTP."
                )
                # wrong when the process is not TP
                # fidelity[idx] = proc_fid_2_ave_fid(process_fidelity, 2)
            elif type == "etg":
                fidelity[idx] = process_fidelity
            else:
                raise ValueError("The type of fidelity should be either 'avg' or 'etg'.")

        return fidelity
        
    def process_dnorm(
        self,
        approx_proc: bool = False,
    ) -> np.ndarray[float]:
        r"""
        Diamond norm of the difference between each effective logical map
        from the initial node to this node and a scaled identity map,
        :math:`\|\tilde{\mathcal E} - p\,\mathcal I\|_\diamond`, with
        :math:`p=\mathrm{Tr}(J)/2` and :math:`J` the Choi matrix of the
        effective logical map :math:`\tilde{\mathcal E}`.

        Parameters
        ----------
        approx_proc : bool
            If True, the product of the step-wise effective logical maps
            (:meth:`accum_logical_process`) replaces the effective logical
            map.
        
        Returns
        -------
        np.ndarray
            Entry ``[a, 0]`` belongs to error space :math:`a` of this node.
        """        
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        if not approx_proc:
            processes = self.effective_logical_process(repr = "choi")
        else:
            processes = self.accum_logical_process(repr = "choi")
            
        dnorms = np.zeros(processes.shape)
        for idx, process in np.ndenumerate(processes):
            compare_process = target_process_for_dnorm(process)
            dnorms[idx] = process.dnorm(compare_process)
            
        return dnorms
    
    def process_choi_trace(
        self,
        approx_proc: bool = False,
    ) -> np.ndarray:
        """
        Trace of the Choi matrix of each effective logical map from the
        initial node to this node, which is 2 for a trace-preserving map
        on the logical qubit.

        Parameters
        ----------
        approx_proc : bool
            If True, the product of the step-wise effective logical maps
            (:meth:`accum_logical_process`) replaces the effective logical
            map.

        Returns
        -------
        np.ndarray
            Entry ``[a, 0]`` belongs to error space :math:`a` of this node.
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")

        if not approx_proc:
            processes = self.effective_logical_process(repr = "choi")
        else:
            processes = self.accum_logical_process(repr = "choi")
            
        traces = np.zeros(processes.shape)
        for idx, process in np.ndenumerate(processes):
            traces[idx] = process.tr()
            
        return traces
    
    def outgoing_dnorm(self) -> np.ndarray[float]:
        """
        Sum of ``Edge.process_dnorm`` over the outgoing edges and over the
        error spaces after the step.

        Returns
        -------
        np.ndarray
            One entry for each error space of this node.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        dnorms = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            edge_dnorm = edge.process_dnorm()
            dnorms += np.sum(edge_dnorm, axis=0)
            
        return dnorms   
    
    def outgoing_infid(
        self,
        type: Literal["avg", "etg"] = "etg",
    ) -> np.ndarray[float]:
        """
        Failure probability of the next step: one minus the sum of the
        step-wise success probabilities (``Edge.fidelity_by_process``)
        over the outgoing edges and over the error spaces after the step.

        Parameters
        ----------
        type : {"avg", "etg"}
            Passed to ``Edge.fidelity_by_process``; only ``"etg"`` is
            implemented.

        Returns
        -------
        np.ndarray
            One entry for each error space of this node.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        fid = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            edge_infid = edge.fidelity_by_process(type)
            fid += np.sum(edge_infid, axis=0)
        
        return 1 - fid
    
    def outgoing_leakage_prob(self) -> np.ndarray[float]:
        """
        Probability that the next step takes the maximally mixed logical
        state of an error space of this node out of the error spaces after
        the step: one minus half the sum of the Choi traces
        (``Edge.process_choi_trace``) over the outgoing edges and over the
        error spaces after the step.

        Returns
        -------
        np.ndarray
            One entry for each error space of this node.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        total_trace = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            total_trace += np.sum(edge.process_choi_trace(), axis=0)
        
        # trace-preserving maps on the logical qubit give a total of 2
        return 1 - total_trace / 2
    
    def accum_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ) -> np.ndarray[qt.Qobj]:
        """
        Product of the step-wise effective logical maps of the steps that
        lead to this node, which ``EvolutionTree.traverse_and_approx``
        stores.

        Parameters
        ----------
        repr : {"super", "choi", "chi", "orth_chi"}
            Representation of each map: the qutip ``super``, ``choi``, or
            ``chi`` representation, or the process matrix of
            ``chencrafts.cqed.proc_repr.to_orth_chi``.

        Returns
        -------
        np.ndarray[qt.Qobj]
            Object array of shape ``(k, 1)`` for a node with :math:`k`
            error spaces.
        """
        if repr == "super":
            return self._accum_logical_process
        elif repr == "choi":
            return to_choi_vec(self._accum_logical_process)
        elif repr == "chi":
            return to_chi_vec(self._accum_logical_process)
        elif repr == "orth_chi":
            return to_orth_chi_vec(self._accum_logical_process)
        else:
            raise ValueError(f"The representation {repr} is not supported.")
        
    def leakage_process_deprecated(
        self,
        truncate_init_space: bool = True,
        reorder_final_space: bool = True,
    ) -> np.ndarray[qt.Qobj]:
        r"""
        Maps from the error spaces of the initial node to the coherence
        between an error space of this node and the leakage subspace,
        ``utils.leakage_process`` of the dynamical map from the initial
        node to this node. The diamond norm of a projected dynamical map
        (``Edge.process_block_dnorm``) supersedes this method.
        
        Parameters
        ----------
        truncate_init_space : bool
            If True, the maps act on operators on the logical qubit; if
            False, on operators on the physical space.
        reorder_final_space : bool
            If True, the result is written in a basis whose first states
            are the ideal logical states of the error space, followed by
            orthonormal states of the rest of the physical space.
        
        Returns
        -------
        np.ndarray[qt.Qobj]
            Object array of shape (number of error spaces of this node,
            number of error spaces of the initial node).
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No leakage process.")
        
        if truncate_init_space:
            # encoded_process already contains the encoders of the initial
            # node, so the inclusion maps of the abstract error space take
            # their place
            return leakage_process(
                process = self.encoded_process,
                init_encoders = abstract_encoders(len(self.init_encoders)),
                final_encoders = self.ideal_encoders(),
                truncate_init_space = True,
                reorder_final_space = reorder_final_space,
            )
        # operators outside the code space of the initial node need the
        # dynamical map on the whole physical space
        return leakage_process(
            process = self.full_process(),
            init_encoders = self.init_encoders,
            final_encoders = self.ideal_encoders(),
            truncate_init_space = False,
            reorder_final_space = reorder_final_space,
        )
            
    def leakage_fidelity_bound_deprecated(
        self,
        return_full_result: bool = False,
        final_space_trunc_threshold: float | None = None,
        constr_proc2_fid: float | None = None,
        constr_proc2_seepage: float | None = None,
    ) -> float | Dict[str, np.ndarray]:
        """
        Upper bound of the process fidelity that a later trace-preserving
        map can recover from the coherence between the error spaces and
        the leakage subspace (``utils.bound_fidelity`` of each map of
        ``leakage_process_deprecated``). The diamond norm of a projected
        dynamical map (``Edge.process_block_dnorm``) supersedes this method.
        
        Parameters
        ----------
        return_full_result : bool
            If True, return the bound, the later map that attains it, and
            the composition of the two maps, for each error space.
        final_space_trunc_threshold : float | None
            If given, a level of the leakage subspace is dropped when every
            matrix element of the map that involves it is below this
            threshold.
        constr_proc2_fid : float | None
            Lower bound of the process fidelity of the later map on the
            logical qubit.
        constr_proc2_seepage : float | None
            Upper bound of the (unnormalized) seepage of the later map.

        Returns
        -------
        float | Dict[str, np.ndarray]
            The sum of the bounds over the error spaces, or, if
            ``return_full_result``, the dictionary with the keys
            ``"fidelity"``, ``"process_2"``, and ``"process_total"``.
        """
        # the truncation and the constraints assume that the ideal logical
        # states come first in the basis after the map
        processes = self.leakage_process_deprecated(
            truncate_init_space = True,
            reorder_final_space = True, 
        )
        
        logical_dim = 2
        full_dim = int(np.sqrt(processes.shape[0]))
        comp_dim = self.ideal_logical_states.size
        leakage_dim = full_dim - comp_dim
        
        bound = np.zeros_like(processes)
        if return_full_result:
            process_2 = np.zeros_like(processes, dtype=qt.Qobj)
            process_total = np.zeros_like(processes, dtype=qt.Qobj)
        
        for idx, proc in np.ndenumerate(processes):
            if final_space_trunc_threshold is not None:
                kept_levels, proc = truncate_leakage_process(
                    proc,
                    logical_dim = logical_dim,
                    threshold = final_space_trunc_threshold,
                )
                
                # the truncation removes the other error spaces as well,
                # because the map has no matrix element to them, so only
                # one logical dimension is subtracted
                leakage_dim = len(kept_levels) - logical_dim
            
            result = bound_fidelity(
                process_1 = proc,
                logical_dim = logical_dim,
                leakage_space_dim = leakage_dim,
                maximize = True,
                constr_proc2_fid = constr_proc2_fid,
                constr_proc2_seepage = constr_proc2_seepage,
            )
            bound[idx] = result["fidelity"]
            if return_full_result:
                process_2[idx] = result["process_2"]
                process_total[idx] = result["process_total"]
            
        if return_full_result:
            return {
                "fidelity": bound,
                "process_2": process_2,
                "process_total": process_total,
            }
        else:
            return np.sum(bound)
    

Node = StateNode    # the only node type


class StateEnsemble:
    """
    Set of nodes after the same number of steps, one for each measurement
    record. A quantity of the ensemble is the sum of that quantity over
    its nodes.
    """

    def __init__(
        self, 
        nodes: List[StateNode] | None = None,
    ):
        if nodes is None:
            nodes = []
        self.nodes: List[StateNode] = nodes

    @property
    def no_further_evolution(self) -> bool:
        """
        Whether no node of the ensemble that is not terminated has an
        outgoing edge.
        """
        no_further_evolution = True
        for node in self.active_nodes():
            if node.out_edges != []:
                no_further_evolution = False
                break

        return no_further_evolution

    def append(self, node: StateNode) -> None:
        """Add a node; a node already in the ensemble raises ``ValueError``."""
        if node in self:
            raise ValueError("The node is already in the ensemble.")
        self.nodes.append(node)

    def is_trace_1(self) -> bool:
        """
        Whether the total probability of the nodes is 1 within 1e-6.
        """
        return np.abs(self.probability - 1) < 1e-6
    
    def fidelity_drop_by_orth(self) -> float:
        """
        Sum of ``StateNode.fidelity_drop_by_orth`` over the nodes.
        """
        return sum([node.fidelity_drop_by_orth() for node in self.nodes])
    
    @property
    def probability(self) -> float:
        """
        Sum of the probabilities of the nodes.
        """
        for node in self.nodes:
            try: 
                node.state
            except AttributeError:
                raise RuntimeError("A node has not been evolved.")
            
        return sum([node.probability for node in self.nodes])
    
    @property
    def state(self) -> qt.Qobj:
        """
        Sum of the states of the nodes, the state averaged over the
        measurement records.
        """
        for node in self.nodes:
            try:
                node.state
            except AttributeError:
                raise AttributeError(f"A node {node} has not been evolved.")

        if not self.is_trace_1():
            if settings.ISSUE_WARNING:  
                warn("The total trace is not 1. The averaged state is not "
                    "physical. \n")
        return sum([node.state for node in self.nodes])

    @property
    def fidelity(self) -> float:
        """ 
        Sum of the fidelities of the nodes (``StateNode.fidelity``).
        """ 
        return sum([node.fidelity for node in self.nodes])

    def deepcopy(self) -> List[StateNode]:
        """ 
        Copies of the nodes without their edges (``StateNode.deepcopy``).
        """ 
        return [
            node.deepcopy() for node in self.nodes
        ]
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, index) -> StateNode:
        return self.nodes[index]
    
    def __len__(self):
        return len(self.nodes)
    
    def order_by_fidelity(self) -> List[StateNode]:
        """
        The nodes in the order of decreasing fidelity.
        """
        return sorted(self.nodes, key=lambda node: node.fidelity, reverse=True)
    
    def order_by_probability(self) -> List[StateNode]:
        """
        The nodes in the order of decreasing probability.
        """
        return sorted(self.nodes, key=lambda node: node.probability, reverse=True)
    
    def expect(self, op: qt.Qobj) -> float:
        """
        Expectation value of an operator in the sum of the states of the
        nodes.
        """
        return sum([node.expect(op) for node in self.nodes])
    
    def next_step_name(self) -> str:
        """
        Name of the first outgoing edge of the first node, which names the
        next step when one step is applied to every node of the ensemble.
        """
        if self.nodes == []:
            return "[NO NEXT STEP DUE TO EMPTY ENSEMBLE]"
        
        if self.nodes[0].out_edges == []:
            return "[NO NEXT STEP DUE TO NO OUT EDGES]"

        return self.nodes[0].out_edges[0].name
    
    def __str__(self) -> str:
        try:
            return (
                f"StateEnsemble before {self.next_step_name()}, "
                + f"prob {self.state.tr().real:.3f}, fid {self.fidelity:.3f}"
            )
        except AttributeError:
            return f"StateEnsemble before {self.next_step_name()}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def active_nodes(self) -> "StateEnsemble":
        """
        The nodes that are not terminated.
        """
        return StateEnsemble([
            node for node in self.nodes if not node.terminated
        ])
    
    def terminated_nodes(self) -> "StateEnsemble":
        """
        The nodes that are terminated.
        """
        return StateEnsemble([
            node for node in self.nodes if node.terminated
        ])
    
    def bloch_vectors(self) -> np.ndarray:
        """
        Sum of ``StateNode.bloch_vector`` over the nodes.
        """
        return np.sum([
            node.bloch_vector() for node in self.nodes
        ], axis=0)
        
    def process(self,) -> qt.Qobj:
        """
        Dynamical map from the initial node to the ensemble, summed over
        the measurement records (the sum of ``StateNode.full_process``), a
        superoperator on the physical space.
        """
        return sum([node.full_process() for node in self.nodes])
        
    def encoded_process(self,) -> qt.Qobj:
        """
        Dynamical map from the initial node to the ensemble applied to
        encoded initial states only, summed over the measurement records
        (the sum of ``StateNode.encoded_process``).
        """
        return sum([node.encoded_process for node in self.nodes])

    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
        force_sum: bool = False,
    ) -> qt.Qobj:
        """
        Effective logical map from the initial node to the ensemble: the
        sum of the effective logical maps of the nodes that are not
        terminated.

        Parameters
        ----------
        repr : {"super", "choi", "chi", "orth_chi"}
            Representation of the map (``StateNode.effective_logical_process``).
        force_sum : bool
            If True, the maps of the error spaces of a node are added, as
            if each error space were decoded right after the last step. If
            False (default), a node with more than one error space raises
            ``ValueError``.
        
        Returns
        -------
        qt.Qobj
            Map on the logical qubit.
        """
        processes = []
        for node in self.active_nodes():
            proc = node.effective_logical_process(repr)

            if proc.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the effective logical process for an ensemble, "
                    "the effective logical process for each node should be "
                    "unique / well defined, or in other words, the ideal "
                    "final decoders should be unique."
                )
            else:
                processes.append(np.sum(proc))

        return sum(processes)
    
    def accum_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
        force_sum: bool = False,
    ) -> qt.Qobj:
        """
        Sum, over the nodes that are not terminated, of the product of the
        step-wise effective logical maps that lead to each node
        (``StateNode.accum_logical_process``).

        Parameters
        ----------
        repr : {"super", "choi", "chi", "orth_chi"}
            Representation of the map.
        force_sum : bool
            If True, the maps of the error spaces of a node are added. If
            False (default), a node with more than one error space raises
            ``ValueError``.
        
        Returns
        -------
        qt.Qobj
            Map on the logical qubit.
        """
        processes = []
        for node in self.active_nodes():
            proc = node.accum_logical_process(repr)

            if proc.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the accumulated logical process for an ensemble, "
                    "the accumulated logical process for each node should be "
                    "unique / well defined, or in other words, the ideal "
                    "final decoders should be unique."
                )
            else:
                processes.append(np.sum(proc))

        return sum(processes)

    def fidelity_by_process(
        self,
        type: Literal["avg", "etg"] = "etg",
        approx_proc: bool = False,
        force_sum: bool = False,
    ) -> float:
        r"""
        Process fidelity to the identity map of the effective logical map
        from the initial node to this ensemble.

        The effective logical map of the ensemble is the sum of the
        effective logical maps of its nodes that are not terminated, one
        node for each measurement record. The process fidelity to the
        identity map, :math:`F=\mathrm{tr}(S)/d^2` with :math:`S` the
        superoperator matrix of a map and :math:`d` the dimension of the
        logical space, is linear in the map, so the value of the ensemble
        is the sum of the values of the nodes
        (:meth:`StateNode.fidelity_by_process`).

        Parameters
        ----------
        type : {"avg", "etg"}
            ``"etg"`` returns :math:`F`; ``"avg"`` is not implemented
            (see :meth:`StateNode.fidelity_by_process`).
        approx_proc : bool
            If False (default), each node contributes its effective
            logical map of the dynamical map on the full space; if True,
            the product of its step-wise effective logical maps.
        force_sum : bool
            If True, the values of the error spaces of a node are added.
            If False (default), a node with more than one error space
            raises ``ValueError``.

        Returns
        -------
        float
            The sum of :math:`F` over the nodes that are not terminated.
        """
        fids = []
        for node in self.active_nodes():
            fid = node.fidelity_by_process(
                type = type,
                approx_proc = approx_proc,
            )

            if fid.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the fidelity by process for an ensemble, the fidelity "
                    "by process for each node should be unique / well defined."
                )
            else:
                fids.append(np.sum(fid))
            
        ensemble_fid = sum(fids)
        
        # F = tr(S)/d^2 is linear in the map, so the sum over the nodes
        # must equal F of the summed effective logical map
        if not approx_proc:
            eff_proc = self.effective_logical_process("super", force_sum)
            fid_compare = qt.process_fidelity(eff_proc)
            assert np.abs(fid_compare - ensemble_fid) < 1e-6, "The fidelity by process should be equal to the process fidelity."
        
        return ensemble_fid
    
    def process_dnorm(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        r"""
        Diamond norm of the difference between the effective logical map
        from the initial node to the ensemble and the identity map,
        :math:`\|\sum_i\tilde{\mathcal E}_i-\mathcal I\|_\diamond`, with
        :math:`\tilde{\mathcal E}_i` the effective logical map of node
        :math:`i`.

        Parameters
        ----------
        force_sum : bool
            If True, the maps of the error spaces of a node are added. If
            False (default), a node with more than one error space raises
            ``ValueError``.
        approx_proc : bool
            If True, the product of the step-wise effective logical maps
            replaces the effective logical map of each node.
        
        Returns
        -------
        float
            The diamond norm.
        """
        if not approx_proc:
            proc = self.effective_logical_process(
                repr="super", 
                force_sum=force_sum,
            )
        else:
            proc = self.accum_logical_process(
                repr="super", 
                force_sum=force_sum,
            )
        return (proc - qt.qeye_like(proc)).dnorm()
    
    def process_choi_trace(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        """
        Trace of the Choi matrix of the effective logical map from the
        initial node to the ensemble: the sum of
        ``StateNode.process_choi_trace`` over the nodes that are not
        terminated.

        Parameters
        ----------
        force_sum : bool
            If True, the traces of the error spaces of a node are added.
            If False (default), a node with more than one error space
            raises ``ValueError``.
        approx_proc : bool
            If True, the product of the step-wise effective logical maps
            replaces the effective logical map of each node.

        Returns
        -------
        float
            The trace; 2 for a trace-preserving map on the logical qubit.
        """
        traces = []
        for node in self.active_nodes():
            tr = node.process_choi_trace(
                approx_proc=approx_proc,
            )

            if tr.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the process choi trace for an ensemble, the process "
                    "choi trace for each node should be unique / well defined."
                )
            else:
                traces.append(np.sum(tr))
                
        return sum(traces)
    
    def process_dnorm_by_sum(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        r"""
        Sum over the nodes of the diamond norms of
        ``StateNode.process_dnorm``, plus the leakage probability.
        
        With :math:`\tilde{\mathcal E}_i` the effective logical map of node
        :math:`i`, the returned value is the last line of
        
        .. math::
            \|\textstyle\sum_i \tilde{\mathcal E}_i - \mathcal I\|_\diamond
            = \|\textstyle\sum_i (\tilde{\mathcal E}_i - p_i\mathcal I) - p_{\rm leak}\mathcal I\|_\diamond
            \leq \textstyle\sum_i \|\tilde{\mathcal E}_i - p_i\mathcal I\|_\diamond + p_{\rm leak}\|\mathcal I\|_\diamond
            \quad \text{(triangle inequality)} ,

        an upper bound of the diamond norm of ``process_dnorm``. Here
        :math:`p_i` is half the trace of the Choi matrix of
        :math:`\tilde{\mathcal E}_i`, :math:`p_{\rm leak}=1-\sum_i p_i` is
        the probability of the leakage, and
        :math:`\|\mathcal I\|_\diamond=1`.

        Parameters
        ----------
        force_sum : bool
            If True, the values of the error spaces of a node are added.
            If False (default), a node with more than one error space
            raises ``ValueError``.
        approx_proc : bool
            If True, the product of the step-wise effective logical maps
            replaces the effective logical map of each node.

        Returns
        -------
        float
            The sum.
        """
        dnorms = []
        for node in self.active_nodes():
            dn = node.process_dnorm(
                approx_proc=approx_proc,
            )

            if dn.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the process dnorm for an ensemble, the process "
                    "dnorm for each node should be unique / well defined."
                )
            else:
                dnorms.append(np.sum(dn))
        
        # the traces of the same maps whose diamond norms are summed
        p_leak = 1 - self.process_choi_trace(force_sum, approx_proc) / 2
        
        return sum(dnorms) + p_leak

    def outgoing_dnorm(self) -> np.ndarray[float]:
        """
        Sum, over the nodes that are not terminated and over their error
        spaces, of ``StateNode.outgoing_dnorm`` weighted by the probability
        of the error space (``traj_prob``) plus
        ``StateNode.outgoing_leakage_prob``, which the diamond norms do not
        contain.
        """
        return np.sum([
            node.outgoing_dnorm() * node.traj_prob + node.outgoing_leakage_prob()
            for node in self.active_nodes()
        ])

    def outgoing_infid(self) -> np.ndarray[float]:
        """
        Sum, over the nodes that are not terminated and over their error
        spaces, of ``StateNode.outgoing_infid`` weighted by the probability
        of the error space (``traj_prob``).
        """
        return np.sum([
            node.outgoing_infid() * node.traj_prob
            for node in self.active_nodes()
        ])