# export_attr.py
import torch

torch.classes.load_library('build/libcustom_class.so')


class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.classes.my_classes.MyStackClass(["just", "testing"])

    def forward(self, s: str) -> str:
        return self.stack.pop() + s


scripted_foo = torch.jit.script(Foo())

scripted_foo.save('foo.pt')
loaded = torch.jit.load('foo.pt')

print(loaded.stack.pop())
