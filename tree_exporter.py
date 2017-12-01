from graphviz import Digraph


def export_node(node, show_details=True):
    g = Digraph(node.get_name(), format="png")
    _create_node(g, node, node.get_msg())
    _expand_node(g, node, show_details)
    return g


def _expand_node(g: Digraph, parent, show_details=True):
    if parent.is_leaf():
        return

    for sub_node in sorted(parent.children.values(), key=lambda act_node: act_node.index):
        if show_details:
            label = sub_node.get_msg()
        else:
            label = sub_node.get_name()
        _create_node(g, sub_node, label)
        g.edge(parent.get_name(), sub_node.get_name(), label=str(sub_node.P))
        #
        _expand_node(g, sub_node, show_details)


def _create_node(g: Digraph, node, label):
    if node.N > 0:
        g.attr('node', style="filled", color="green")
    else:
        g.attr('node', style="filled", color="lightgrey")

    g.node(name=node.get_name(), label=label)

