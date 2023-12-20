def get_sym_node_fn(name):
    def fn(self):
        print(f"call self._sym_{name}", self.a)
        return 0.0

    fn.__name__ = fn.__qualname__ = f"SymNode.sym_{name}"
    return fn

class SymNode:
    def __init__(self):
        self.a = 123

math_op_names = ("sqrt", "cos", "sin")
for name in math_op_names:
    sym_name = f"sym_{name}"
    setattr(SymNode, sym_name, get_sym_node_fn(name))

print(SymNode.__dict__)
n = SymNode()
n.sym_sqrt()
