from graphviz import Digraph


def export_node(node, expand=True):
    g = Digraph(node.get_name(), format="png")
    parent = node
    _create_node(g, parent, parent.get_msg())

    while not parent.is_leaf():
        selected_node = parent.select()
        for sub_node in sorted(parent.children.values(), key=lambda act_node: act_node.index):
            selected = selected_node.index == sub_node.index
            if selected or not expand:
                label = sub_node.get_msg()
            else:
                label = sub_node.get_name()
            _create_node(g, sub_node, label, selected)
            g.edge(parent.get_name(), sub_node.get_name(), label=str(sub_node.P) if selected or not expand else '')
        parent = selected_node
        if not expand:
            break

    return g


def _create_node(g: Digraph, node, label, selected=False):
    if selected:
        g.attr('node', style="filled", color="red")
    elif node.N > 0:
        g.attr('node', style="filled", color="green")
    else:
        g.attr('node', style="filled", color="lightgrey")

    g.node(name=node.get_name(), label=label)

