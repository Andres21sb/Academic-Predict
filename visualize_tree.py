import graphviz

with open('Results/decision_tree.dot') as f:
    dot_graph = f.read()


# Visualize the decision tree
graphviz.Source(dot_graph)


