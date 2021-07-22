import torch

# `torch.classes.load_library()` allows you to pass the path to your .so file
# to load it in and make the custom C++ classes available to both Python and
# TorchScript
torch.classes.load_library("build/libcustom_class.so")
# You can query the loaded libraries like this:
print(torch.classes.loaded_libraries)
# prints {'/custom_class_project/build/libcustom_class.so'}

# We can find and instantiate our custom C++ class in python by using the
# `torch.classes` namespace:
#
# This instantiation will invoke the MyStackClass(std::vector<T> init)
# constructor we registered earlier
s = torch.classes.my_classes.MyStackClass(["foo", "bar"])

# We can call methods in Python
s.push("pushed")
assert s.pop() == "pushed"

# Test custom operator
s.push("pushed")
torch.ops.my_classes.manipulate_instance(s)  # acting as s.pop()
assert s.top() == "bar" 

# Returning and passing instances of custom classes works as you'd expect
s2 = s.clone()
s.merge(s2)
for expected in ["bar", "foo", "bar", "foo"]:
    assert s.pop() == expected

# We can also use the class in TorchScript
# For now, we need to assign the class's type to a local in order to
# annotate the type on the TorchScript function. This may change
# in the future.
MyStackClass = torch.classes.my_classes.MyStackClass


@torch.jit.script
def do_stacks(s: MyStackClass):  # We can pass a custom class instance
    # We can instantiate the class
    s2 = torch.classes.my_classes.MyStackClass(["hi", "mom"])
    s2.merge(s)  # We can call a method on the class
    # We can also return instances of the class
    # from TorchScript function/methods
    return s2.clone(), s2.top()


stack, top = do_stacks(torch.classes.my_classes.MyStackClass(["wow"]))
assert top == "wow"
for expected in ["wow", "mom", "hi"]:
    assert stack.pop() == expected

