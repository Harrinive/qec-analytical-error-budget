__all__ = [
    'EvolutionGraph',
    'EvolutionTree',
    'LogicalEvolutionTree',
    'step_wise_success_probability',
    'process_matrix_diagonal',
]

import numpy as np
import qutip as qt

from chencrafts.cqed.qt_helper import Pauli_twirl
from .node import StateEnsemble

from typing import Iterable, Iterator, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node
    from .edge import Edge
    import networkx as nx

class EvolutionGraph:
    """
    Directed graph of an evolution: a node represents a state, and an edge
    represents the dynamical map of one step applied to the state of its
    initial node.
    """
    def __init__(self):
        self.nodes: List["Node"] = []
        self.edges: List["Edge"] = []

    def add_node(self, node: "Node") -> None:
        """
        Add a node to the graph and give it the next free index.
        """
        node.assign_index(self.node_num)
        self.nodes.append(node)

    def add_edge_connect(
        self, edge: "Edge", init_node: "Node", final_node: "Node",
    ) -> None:
        """
        Add an edge between two nodes that the graph already contains.

        Parameters
        ----------
        edge : Edge
            The edge; it receives the next free index.
        init_node, final_node : Node
            The nodes that represent the states before and after the step.
        """
        assert init_node in self.nodes
        assert final_node in self.nodes

        edge.assign_index(self.edge_num)

        init_node.add_out_edges(edge)
        edge.connect(
            init_node, final_node,
        )

        self.edges.append(edge)

    @property
    def node_num(self) -> int:
        """Number of nodes of the graph."""
        return len(self.nodes)
    
    @property
    def edge_num(self) -> int:
        """Number of edges of the graph."""
        return len(self.edges)
    
    def to_nx(self) -> "nx.DiGraph":
        """
        The graph as a ``networkx.DiGraph``, an optional dependency.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is a optional dependency for QEC_graph module."
                "Please install it via 'pip install networkx' or 'conda install networkx'."
            )

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from([node.to_nx() for node in self.nodes])
        nx_graph.add_edges_from([edge.to_nx() for edge in self.edges])

        return nx_graph
    
    def clear_evolution_data(
        self, 
        exclude: List["Node"] = [],
    ) -> None:
        """
        Delete the states that the nodes hold, except for the nodes in
        ``exclude``.
        """
        for node in self.nodes:
            if node not in exclude:
                node.clear_evolution_data()
    

class EvolutionTree(EvolutionGraph):
    r"""
    Tree of an evolution: every node except the initial node has one
    incoming edge.

    A node represents the state at :math:`(n, \mathbf r)`, after :math:`n`
    steps with measurement record :math:`\mathbf r`. A measurement has one
    outgoing edge for each outcome.
    """

    def _traverse_single_step(
        self, 
        initial_ensemble: StateEnsemble,
        evolve: bool = True,
    ) -> StateEnsemble:
        """
        Nodes one step after the nodes of ``initial_ensemble``; the edges
        of the step are evolved if ``evolve`` is True.
        """
        final_ensemble = StateEnsemble()

        # check if the evolution reaches the final state, namely no node
        # has out edges
        if initial_ensemble.no_further_evolution:
            return final_ensemble

        # evolve
        for node in initial_ensemble:
            if node.terminated:
                # a manually terminated node will not be evolved and it
                # will always be in the final ensemble, unchanged
                final_ensemble.append(node)
                continue

            if node.out_edges == []:
                # this node has no out edges (and not terminated),
                # usually because the occupation 
                # of the state (normalization factor) is very small
                continue

            for edge in node.out_edges:
                if evolve:
                    edge.evolve()

                if edge.final_state not in final_ensemble:
                    # the final state has not been traversed
                    final_ensemble.append(edge.final_state)

        return final_ensemble
    
    def traverse(
        self,
        steps: int,
        initial_ensemble: StateEnsemble | None = None,
        evolve: bool = True,
    ) -> StateEnsemble:
        """
        Evolve the states through a number of steps.

        Parameters
        ----------
        steps : int
            Number of steps. The evolution stops earlier, with a printed
            message, at nodes without an outgoing edge.
        initial_ensemble : StateEnsemble | None
            Nodes from which the evolution starts. If None, the initial
            node of the tree.
        evolve : bool
            If False, the edges are followed without evolving the states.

        Returns
        -------
        StateEnsemble
            The nodes after the last step.
        """
        if initial_ensemble is None:
            current_ensemble = StateEnsemble([self.nodes[0]])
        else:
            current_ensemble = initial_ensemble

        for stp in range(steps):
            current_ensemble = self._traverse_single_step(
                current_ensemble, evolve=evolve
            )
            if current_ensemble.no_further_evolution:
                if stp+1 < steps:
                    print(f"The evolution stops earlier at step {stp+1}")
                break

        return current_ensemble
    
    def attr_by_step(
        self, 
        attr: str, 
        handle_error: bool = True,
        *args, 
        **kwargs,
    ) -> List:
        """
        A quantity of the ensemble of nodes after each step.

        Parameters
        ----------
        attr : str
            Name of an attribute or a method of ``StateEnsemble``. A method
            is called with ``*args`` and ``**kwargs``.
        handle_error : bool
            If True (default), an entry whose evaluation raises an
            exception is None. If False, the exception is raised.

        Returns
        -------
        List
            Entry :math:`n` belongs to the nodes after :math:`n` steps;
            entry 0 belongs to the initial node.
        """
        def _get_attr(ensemble: "StateEnsemble"):
            try:
                current_attr = getattr(ensemble, attr)
                if callable(current_attr):
                    return current_attr(*args, **kwargs)
                else:
                    return current_attr
                
            except Exception as e:
                if handle_error:
                    return None
                else:
                    raise e

        current_ensemble = StateEnsemble([self.nodes[0]])
        attr_by_stp = [_get_attr(current_ensemble)]

        while True:
            current_ensemble = self._traverse_single_step(
                current_ensemble, evolve=False
            )
            attr_by_stp.append(_get_attr(current_ensemble))
            if current_ensemble.no_further_evolution:
                break
            
        return attr_by_stp

    def process_fidelity_by_step(self) -> np.ndarray:
        r"""
        Process fidelity to the identity map of the effective logical map
        after each step.

        After :math:`n` steps, each measurement record :math:`\mathbf r`
        that a correctable error can produce and each error space
        :math:`a` at :math:`(n, \mathbf r)` contribute one map on the
        logical qubit: it encodes the logical qubit, applies the dynamical
        maps of the :math:`n` steps with record :math:`\mathbf r`, and
        decodes from error space :math:`a`. The effective logical map
        after :math:`n` steps is the sum of these maps, and its process
        fidelity to the identity map is

        .. math::
            F_n = \sum_{\mathbf r, a} \mathrm{tr}(S_{n,\mathbf r,a}) / d^2 ,

        where :math:`S_{n,\mathbf r,a}` is the superoperator matrix (qutip
        ``super`` representation) of the map of record :math:`\mathbf r`
        and error space :math:`a`, and :math:`d=2` is the dimension of the
        logical qubit. :math:`F_n` is linear in the map and is not divided
        by the trace that the map retains.

        Returns
        -------
        np.ndarray
            Shape ``(N+1,)`` for a tree of :math:`N` steps. Entry
            :math:`n` is the process fidelity after :math:`n` steps;
            entry 0 belongs to the initial state, before any step.
        """
        return np.array(
            self.attr_by_step(
                "fidelity_by_process", handle_error=False,
                type="etg", force_sum=True,
            ),
            dtype=float,
        )

    def iter_steps(
        self,
        skip_terminated: bool = True,
    ) -> Iterator[Tuple[int, "Node", List["Edge"]]]:
        r"""
        Iterate over the nodes of the tree in the order of the steps,
        yielding each node together with the edges of the step that
        follows it.

        A yielded node represents the state at :math:`(n-1, \mathbf r)`,
        after :math:`n-1` steps with measurement record :math:`\mathbf r`.
        Each of its outgoing edges represents the dynamical map of step
        :math:`n` applied to that state (in a logical evolution tree, a
        step-wise effective logical map). Every node after :math:`n-1`
        steps is yielded before any node after :math:`n` steps; nodes
        after the same number of steps keep the order in which their
        incoming edges were added to the tree.

        Parameters
        ----------
        skip_terminated : bool
            If True (default), an edge whose final node is terminated is
            left out, and the iteration does not continue from that node.
            A terminated node represents the state for an unexpected
            measurement record, one that no correctable error can
            produce. If False, every outgoing edge is kept.

        Yields
        ------
        step : int
            Index :math:`n \geq 1` of the step whose dynamical maps the
            edges represent.
        node : Node
            The node that represents the state before step :math:`n`.
        edges : List[Edge]
            The outgoing edges of ``node`` that are kept. A node with no
            kept edge is not yielded.
        """
        # breadth-first, so that the steps come out in order: a layer
        # holds the nodes after the same number of steps, each paired with
        # that number
        layer = [(self.nodes[0], 0)]
        while layer:
            next_layer = []
            for node, step in layer:
                # keep the edges of the next step; an edge into a
                # terminated node is dropped when skip_terminated is set,
                # which also keeps that node out of the next layer
                edges = [
                    edge for edge in node.out_edges
                    if not (skip_terminated and edge.final_state.terminated)
                ]
                if edges:
                    yield step + 1, node, edges
                next_layer.extend(
                    (edge.final_state, step + 1) for edge in edges
                )
            layer = next_layer

    def final_nodes(
        self,
        skip_terminated: bool = True,
    ) -> List["Node"]:
        r"""
        Nodes that represent the states after the last step of the tree,
        in the order of ``nodes``.

        A node is final when no edge of a later step starts from it.

        Parameters
        ----------
        skip_terminated : bool
            A terminated node represents the state for an unexpected
            measurement record, one that no correctable error can
            produce. If True (default), a terminated node is not returned,
            and an edge into a terminated node does not count as an
            outgoing edge. If False, the result is every node without an
            outgoing edge, terminated or not.

        Returns
        -------
        List[Node]
            The final nodes. They depend on the edges that the tree
            contains, not on how many steps have been evolved.
        """
        # with terminated nodes included, a final node is a node without
        # any outgoing edge
        if not skip_terminated:
            return [node for node in self.nodes if not node.out_edges]

        # otherwise terminated nodes are ignored at both ends of an edge:
        # a final node is not terminated, and none of its edges leads to a
        # node that is not terminated
        return [
            node for node in self.nodes
            if not node.terminated and not [
                edge for edge in node.out_edges
                if not edge.final_state.terminated
            ]
        ]

    def traverse_and_approx(self) -> None:
        """
        Store on each node the product of the step-wise effective logical
        maps of the steps that lead to it, and the probability of that
        sequence of steps.
        
        The probability of a node (``traj_prob``, one entry for each error
        space) is the product, over the edges that lead to the node, of
        half the trace of the Choi matrix of the step-wise effective
        logical map. It agrees to first order in the error probabilities
        with the probability of the state of the node. The product of the
        maps is returned by ``StateNode.accum_logical_process``.
        Terminated nodes are skipped.
        """
        init_node = self.nodes[0]
        current_layer = [init_node]
        next_layer = []
        init_node.traj_prob = np.array([1.0])
        init_node._accum_logical_process = np.array([[qt.to_super(qt.qeye(2))]])
        
        while current_layer:
            for initial_state in current_layer:
                for edge in initial_state.out_edges:
                    final_state = edge.final_state
                    if final_state.terminated:
                        continue
                    
                    # probability of the transitions between error spaces
                    # (half the Choi trace, for a logical qubit) times the
                    # probability of the error spaces before the step
                    final_state.traj_prob = (
                        edge.process_choi_trace() / 2    # dim=2
                        @ initial_state.traj_prob.reshape(-1, 1)
                    ).reshape(-1)
                    
                    # matrix product over the error spaces, in which each
                    # entry is a map and the product of two entries is
                    # their composition
                    edge_process = edge.effective_logical_process(repr = "super")
                    accum_proc = np.ndarray(
                        (edge_process.shape[0], 1), 
                        dtype=object
                    )
                    for idx, proc in np.ndenumerate(edge_process):
                        result_process = (
                            proc * initial_state._accum_logical_process[idx[1], 0]
                        )
                        if idx[1] == 0:
                            accum_proc[idx[0], 0] = result_process
                        else:
                            accum_proc[idx[0], 0] += result_process
                    final_state._accum_logical_process = accum_proc
                    
                    next_layer.append(final_state)
                    
            current_layer = next_layer
            next_layer = []

    def logical_evolution_tree(self) -> "LogicalEvolutionTree":
        r"""
        Build the logical evolution tree from this actual evolution tree.

        Each node of the returned tree is one error space after one step.
        Each edge is one step-wise effective logical map of the ideal
        evolution tree: its superoperator is that map, and its ideal operator
        is the identity on the logical qubit. An omitted error-space
        transition is not given an edge. An error-accumulation step becomes
        one edge for each error record the ideal evolution tree appends,
        including the no-loss error record and the one-photon-loss error
        record.

        The codewords must be orthogonalized with the method that was used
        when this actual evolution tree was built. Each returned node stores
        the composed step-wise effective logical maps in ``process`` and the
        step index in ``step_index``.

        Returns
        -------
        LogicalEvolutionTree
            Logical evolution tree. Its edges are not built by ``evolve``.

        Raises
        ------
        TypeError
            If this object is already a logical evolution tree.
        """
        if isinstance(self, LogicalEvolutionTree):
            raise TypeError(
                "logical_evolution_tree builds a logical evolution tree from "
                "an actual evolution tree. This object is already a logical "
                "evolution tree."
            )
        from .edge import PropagatorEdge
        from .node import StateNode

        logical = LogicalEvolutionTree()
        eye = qt.to_super(qt.qeye(2))
        root = StateNode()
        # The root is the encoded logical qubit, before any step.
        root.process = eye
        root.step_index = 0
        root.meas_record = []
        logical.add_node(root)

        # One success state for each (node of the actual evolution tree, error space).
        node_for = {(id(self.nodes[0]), 0): root}
        pending = [self.nodes[0]]
        visited = set()
        while pending:
            actual = pending.pop(0)
            if id(actual) in visited:
                continue
            visited.add(id(actual))
            for edge in actual.out_edges:
                final = edge.final_state
                if getattr(final, "terminated", False):
                    continue
                # The record is written by evolve(), as a node's state is
                # written when the node is evolved. An edge saved before the
                # record existed is evolved here before it is read.
                if not hasattr(edge, "init_error_space_by_final"):
                    edge.evolve()
                init_error_space_by_final = edge.init_error_space_by_final
                # Rows are final error spaces, columns are initial error spaces.
                error_space_maps = edge.effective_logical_process(repr="super")
                for final_idx, init_idx in enumerate(init_error_space_by_final):
                    if (
                        init_idx >= error_space_maps.shape[1]
                        or final_idx >= error_space_maps.shape[0]
                    ):
                        continue
                    parent = node_for.get((id(actual), init_idx))
                    if parent is None:
                        # This initial error space is not on the logical evolution tree.
                        continue
                    
                    # step-wise effective logical map
                    selm = error_space_maps[final_idx, init_idx]
                    child = StateNode()
                    child.step_index = parent.step_index + 1
                    child.process = selm * parent.process
                    child.meas_record = list(getattr(final, "meas_record", []))
                    logical.add_node(child)
                    node_for[(id(final), final_idx)] = child
                    link = PropagatorEdge(
                        f"{edge.name} | error space {init_idx} to {final_idx}",
                        selm,
                        [qt.qeye(2)],
                        error_accumulation=bool(
                            getattr(edge, "error_accumulation", False)
                        ),
                    )
                    # One final error space, produced from initial error space 0.
                    link.init_error_space_by_final = [0]
                    logical.add_edge_connect(link, parent, child)
                if id(final) not in visited:
                    pending.append(final)
        return logical


class LogicalEvolutionTree(EvolutionTree):
    r"""
    Logical evolution tree: each node is one error space after one step,
    and each edge holds one step-wise effective logical map, a
    superoperator on the logical qubit.
    """

    def logical_evolution_tree(self) -> "LogicalEvolutionTree":
        r"""
        Refuse to build a logical evolution tree from a logical evolution tree.

        Raises
        ------
        TypeError
            This object is already a logical evolution tree.
        """
        return super().logical_evolution_tree()

    def failure_markov_chain(self) -> np.ndarray:
        r"""
        Transition matrix of the failure Markov chain of this logical evolution tree.

        Row ``i`` is success state ``i``, in the same order as ``nodes``.
        The last row is the failure state. The entry from a success state to
        the success state at the end of one of its outgoing edges is the
        step-wise success probability of that edge. The entry from a success
        state that is not at the last step to the failure state is the
        failure probability, one minus the sum of those step-wise success
        probabilities. A success state at the last step and the failure
        state stay where they are. Each row sums to one.

        Returns
        -------
        np.ndarray
            Shape ``(n_success + 1, n_success + 1)``.
        """
        n_success = self.node_num
        transition = np.zeros((n_success + 1, n_success + 1), dtype=float)
        failure = n_success
        last_step = max(getattr(node, "step_index", 0) for node in self.nodes)
        for edge in self.edges:
            origin = edge.init_state.index
            destination = edge.final_state.index
            transition[origin, destination] = step_wise_success_probability(
                edge.real_map
            )
        for node in self.nodes:
            origin = node.index
            outgoing = float(transition[origin, :n_success].sum())
            at_last_step = getattr(node, "step_index", 0) == last_step
            no_descendant = len(node.out_edges) == 0
            if at_last_step and no_descendant:
                # A completed trajectory does not fail again.
                transition[origin, :] = 0.0
                transition[origin, origin] = 1.0
            else:
                transition[origin, failure] = max(0.0, 1.0 - outgoing)
        transition[failure, failure] = 1.0
        return transition

    def process_fidelity_by_step(self) -> np.ndarray:
        r"""
        Process fidelity to the identity map of the partially ideal logical
        map after each step.

        Entry :math:`n` is the sum, over the success states after :math:`n`
        steps, of :math:`\mathrm{tr}(S)/4`, with :math:`S` the superoperator
        matrix of the composed step-wise effective logical maps that lead to
        the success state. The expression is linear in the map, so the sum
        is the process fidelity of the sum of the composed maps.

        Returns
        -------
        np.ndarray
            Shape ``(N+1,)`` for a tree of :math:`N` steps; entry 0 belongs
            to the encoded logical qubit, before any step.
        """
        return _sum_node_weight_by_step(
            self,
            lambda node: step_wise_success_probability(node.process),
        )

    def process_fidelity_twirled_by_step(
        self,
        untwirled_steps: Iterable[int] = (),
        through_node: "Node | None" = None,
    ) -> np.ndarray:
        r"""
        Process fidelity after each step of the logical evolution tree in
        which every step-wise effective logical map is replaced by its Pauli
        twirl, except at chosen steps.

        The Pauli twirl (:func:`~chencrafts.cqed.qt_helper.Pauli_twirl`)
        keeps the diagonal of the process matrix of a map in the Pauli basis
        :math:`(I, X, Y, Z)` and removes its off-diagonal entries: it
        replaces the step-wise effective logical map by the stochastic Pauli
        map with the same probabilities of no logical error and of the
        logical :math:`X`, :math:`Y`, and :math:`Z` errors. Entry
        :math:`n` is the sum, over the success states after :math:`n`
        steps, of :math:`\mathrm{tr}(S)/4`, with :math:`S` the superoperator
        matrix of the composed maps that lead to the success state, as in
        :meth:`process_fidelity_by_step`.

        With every step untwirled the result is that of
        :meth:`process_fidelity_by_step`. The twirl does not change the
        process fidelity of a single map, and an off-diagonal part composed
        with stochastic Pauli maps keeps a zero diagonal. The fidelity with
        only the steps :math:`i` and :math:`j` untwirled, minus the fidelity
        with every step twirled, is therefore the contribution of the
        off-diagonal entries of these two steps together.

        Parameters
        ----------
        untwirled_steps : Iterable[int]
            Steps whose maps keep their off-diagonal entries; step :math:`n`
            is the one that leads to the success states after :math:`n`
            steps (1 for the first step).
        through_node : Node | None
            If given, only this success state and the success states that
            the tree reaches from it are summed; every other entry is zero.
            None (default) sums every success state.

        Returns
        -------
        np.ndarray
            Shape ``(N+1,)`` for a tree of :math:`N` steps; entry 0 belongs
            to the encoded logical qubit, before any step.
        """
        untwirled = set(untwirled_steps)
        root = next(node for node in self.nodes if node.step_index == 0)
        composed = {id(root): qt.to_super(qt.qeye(2))}
        # the edges in the order of their steps, so that the composition up
        # to the initial state of an edge exists before the edge is applied
        for edge in sorted(self.edges, key=lambda edge: edge.final_state.step_index):
            if edge.final_state.step_index in untwirled:
                selm = edge.real_map
            else:
                selm = Pauli_twirl(edge.real_map)
            composed[id(edge.final_state)] = selm * composed[id(edge.init_state)]

        # the success states reached from through_node, which itself included
        counted = None
        if through_node is not None:
            counted = set()
            pending = [through_node]
            while pending:
                node = pending.pop()
                counted.add(id(node))
                pending.extend(edge.final_state for edge in node.out_edges)
        return _sum_node_weight_by_step(
            self,
            lambda node: (
                step_wise_success_probability(composed[id(node)])
                if counted is None or id(node) in counted else 0.0
            ),
        )

    def intermediate_success_prob_by_step(self) -> np.ndarray:
        r"""
        Total probability of the success states of the failure Markov chain
        after each step.

        The chain starts in the success state of the encoded logical qubit
        and evolves with the transition matrix of ``failure_markov_chain``.
        Entry :math:`n` is the sum of the probabilities of the success
        states after :math:`n` steps.

        Returns
        -------
        np.ndarray
            Shape ``(N+1,)`` for a tree of :math:`N` steps; entry 0 belongs
            to the encoded logical qubit, before any step.
        """
        transition = self.failure_markov_chain()
        last_step = max(node.step_index for node in self.nodes)
        totals = np.zeros(last_step + 1, dtype=float)
        # Probability of each success state, plus the failure state.
        probability = np.zeros(transition.shape[0], dtype=float)
        root = next(node for node in self.nodes if node.step_index == 0)
        probability[root.index] = 1.0
        for step in range(last_step + 1):
            for node in self.nodes:
                if node.step_index == step:
                    totals[step] += probability[node.index]
            probability = probability @ transition
        return totals

    def failure_prob_by_step_and_manifestation(self) -> np.ndarray:
        r"""
        Probability that the failure Markov chain fails during each step,
        resolved by logical manifestation: a logical :math:`X`, :math:`Y`,
        or :math:`Z` error, or leakage.

        A success state :math:`\mu` before step :math:`n` has the
        intermediate success probability :math:`p_\mu`, and its outgoing
        step-wise effective logical maps
        :math:`\tilde{\mathcal E}_\nu` have the process matrices
        :math:`\mathcal X(\tilde{\mathcal E}_\nu)`
        (:func:`process_matrix_diagonal`). The entries of step :math:`n` are

        .. math::
            \Delta p^{i}_n = \sum_\mu p_\mu \sum_\nu
            \mathcal X(\tilde{\mathcal E}_\nu)_{ii},
            \qquad
            \Delta p^{\mathrm{leak}}_n = \sum_\mu p_\mu
            \Bigl[1 - \sum_\nu \mathrm{tr}\,
            \mathcal X(\tilde{\mathcal E}_\nu)\Bigr]

        with :math:`i` one of :math:`X, Y, Z`. The leakage entry is the
        probability that no outgoing map keeps, evaluated on the maximally
        mixed logical state. The four entries of a step add up to the
        decrease of ``intermediate_success_prob_by_step`` at that step.

        Returns
        -------
        np.ndarray
            Shape ``(N+1, 4)`` for a tree of :math:`N` steps, with the
            columns :math:`X`, :math:`Y`, :math:`Z`, leakage. Row
            :math:`n` belongs to step :math:`n`; row 0 is zero.
        """
        transition = self.failure_markov_chain()
        last_step = max(node.step_index for node in self.nodes)
        failure_prob = np.zeros((last_step + 1, 4), dtype=float)
        # Probability of each success state, plus the failure state.
        probability = np.zeros(transition.shape[0], dtype=float)
        root = next(node for node in self.nodes if node.step_index == 0)
        probability[root.index] = 1.0
        for step in range(last_step):
            for node in self.nodes:
                if node.step_index != step or not node.out_edges:
                    continue
                # The outgoing maps of one success state are added before
                # the split: what none of them keeps is leakage.
                diagonal = sum(
                    process_matrix_diagonal(edge.real_map)
                    for edge in node.out_edges
                )
                p_mu = probability[node.index]
                failure_prob[step + 1, :3] += p_mu * diagonal[1:]
                failure_prob[step + 1, 3] += p_mu * (1.0 - diagonal.sum())
            probability = probability @ transition
        return failure_prob


def process_matrix_diagonal(superop: qt.Qobj) -> np.ndarray:
    r"""
    Diagonal of the process matrix of a map on the logical qubit in the
    Pauli basis :math:`(I, X, Y, Z)`.

    The process matrix :math:`\mathcal X` is defined by
    :math:`\mathcal E(\rho)=\sum_{ij}\mathcal X_{ij}\,
    \sigma_i\rho\,\sigma_j^\dagger`, so the identity map has
    :math:`\mathcal X_{II}=1` (``qutip.to_chi`` divided by 4). The entry
    :math:`\mathcal X_{II}` is the process fidelity to the identity map,
    the other three entries are the probabilities of the logical
    :math:`X`, :math:`Y`, and :math:`Z` errors when the process matrix is
    diagonal, and the sum of the four is the probability that the map
    retains for the maximally mixed state.

    Parameters
    ----------
    superop : qutip.Qobj
        Map on the logical qubit, in the qutip ``super`` representation.

    Returns
    -------
    np.ndarray
        Shape ``(4,)``, the real entries
        :math:`\mathcal X_{II}, \mathcal X_{XX}, \mathcal X_{YY},
        \mathcal X_{ZZ}`.
    """
    return np.real(np.diag(qt.to_chi(superop).full())) / 4.0


def step_wise_success_probability(selm: qt.Qobj) -> float:
    r"""
    Step-wise success probability of one step-wise effective logical map,
    its process fidelity to the identity map.

    The step-wise success probability is :math:`g = \chi_{II}/4`, where
    :math:`\chi` is the process matrix of the map in the Pauli basis
    :math:`(I, X, Y, Z)` as ``qutip.to_chi`` normalizes it, with
    :math:`\chi_{II}=4` for the identity map. Equivalently,
    :math:`g=\mathrm{Tr}(S)/4`, with :math:`S` the superoperator matrix
    of the map. It is not divided by the probability that the map
    retains: the map :math:`c\,\mathcal I`, with :math:`\mathcal I` the
    identity map, has :math:`g=c`.

    Parameters
    ----------
    selm : qutip.Qobj
        The step-wise effective logical map, as a superoperator on the
        logical qubit.

    Returns
    -------
    float
        The step-wise success probability :math:`g`.
    """
    return float(process_matrix_diagonal(selm)[0])


def _sum_node_weight_by_step(tree: "EvolutionTree", weight) -> np.ndarray:
    r"""
    Sum of ``weight(node)`` over the success states after each step of a
    logical evolution tree; entry 0 belongs to the state before any step.
    """
    last_step = max(getattr(node, "step_index", 0) for node in tree.nodes)
    totals = np.zeros(last_step + 1, dtype=float)
    for node in tree.nodes:
        totals[node.step_index] += weight(node)
    return totals