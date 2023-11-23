from behaviors import TrajectoryNode

# Test of trajectory
children = [
    TrajectoryNode(np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3), 0, 1, "ee")
]
sequence_node = SequenceNode(children)

state: Dict = {"ee": {"p": np.zeros(3)}, "time": 0}
ref: Dict = {"ee": {"p": np.zeros(3)}}

sequence_node.execute(state, ref)
state["time"] = 0.5
sequence_node.execute(state, ref)
state["time"] = 1.0
sequence_node.execute(state, ref)
