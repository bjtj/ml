from graphviz import Digraph
import re
import torch
import torch.nn.functional
from torch.autograd import Variable
import torchvision.models as models


def make_dot(var, params):

    param_map = {id(v): k for k, v, in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled', shape='box', align='left',
                     fontsize='12', ranksep='0.1', height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['{:d}'.format(v) for v in size]) + ')'


    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '{:s}\n {:s}'.format(param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def test():
    inputs = torch.randn(1, 3, 224, 224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))
    g = make_dot(y, resnet18.state_dict())
    g.render('x.gv', view=True)


def main():
    test()

if __name__ == '__main__':
    main()

