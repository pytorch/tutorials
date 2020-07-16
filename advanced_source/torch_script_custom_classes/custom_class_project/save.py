import torch

torch.classes.load_library('build/libcustom_class.so')


class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s: str) -> str:
        stack = torch.classes.my_classes.MyStackClass(["hi", "mom"])
        return stack.pop() + s


scripted_foo = torch.jit.script(Foo())
print(scripted_foo.graph)

scripted_foo.save('foo.pt')
