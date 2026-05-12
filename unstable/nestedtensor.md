Note

Go to the end
to download the full example code.

# Getting Started with Nested Tensors

Nested tensors generalize the shape of regular dense tensors, allowing for representation
of ragged-sized data.

- for a regular tensor, each dimension is regular and has a size
- for a nested tensor, not all dimensions have regular sizes; some of them are ragged

Nested tensors are a natural solution for representing sequential data within various domains:

- in NLP, sentences can have variable lengths, so a batch of sentences forms a nested tensor
- in CV, images can have variable shapes, so a batch of images forms a nested tensor

In this tutorial, we will demonstrate basic usage of nested tensors and motivate their usefulness
for operating on sequential data of varying lengths with a real-world example. In particular,
they are invaluable for building transformers that can efficiently operate on ragged sequential
inputs. Below, we present an implementation of multi-head attention using nested tensors that,
combined usage of `torch.compile`, out-performs operating naively on tensors with padding.

Nested tensors are currently a prototype feature and are subject to change.

```
import numpy as np
import timeit
import torch
import torch.nn.functional as F

from torch import nn

torch.manual_seed(1)
np.random.seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Nested tensor initialization

From the Python frontend, a nested tensor can be created from a list of tensors.
We denote nt[i] as the ith tensor component of a nestedtensor.

```
nt = torch.nested.nested_tensor([torch.arange(12).reshape(
 2, 6), torch.arange(18).reshape(3, 6)], dtype=torch.float, device=device)
print(f"{nt=}")
```

```
/var/lib/ci-user/.local/lib/python3.10/site-packages/torch/nested/__init__.py:256: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
 return _nested.nested_tensor(
nt=nested_tensor([
 tensor([[ 0., 1., 2., 3., 4., 5.],
 [ 6., 7., 8., 9., 10., 11.]], device='cuda:0'),
 tensor([[ 0., 1., 2., 3., 4., 5.],
 [ 6., 7., 8., 9., 10., 11.],
 [12., 13., 14., 15., 16., 17.]], device='cuda:0')
], device='cuda:0')
```

By padding every underlying tensor to the same shape,
a nestedtensor can be converted to a regular tensor.

```
padded_out_tensor = torch.nested.to_padded_tensor(nt, padding=0.0)
print(f"{padded_out_tensor=}")
```

```
padded_out_tensor=tensor([[[ 0., 1., 2., 3., 4., 5.],
 [ 6., 7., 8., 9., 10., 11.],
 [ 0., 0., 0., 0., 0., 0.]],

 [[ 0., 1., 2., 3., 4., 5.],
 [ 6., 7., 8., 9., 10., 11.],
 [12., 13., 14., 15., 16., 17.]]], device='cuda:0')
```

All tensors posses an attribute for determining if they are nested;

```
print(f"nt is nested: {nt.is_nested}")
print(f"padded_out_tensor is nested: {padded_out_tensor.is_nested}")
```

```
nt is nested: True
padded_out_tensor is nested: False
```

It is common to construct nestedtensors from batches of irregularly shaped tensors.
i.e. dimension 0 is assumed to be the batch dimension.
Indexing dimension 0 gives back the first underlying tensor component.

```
print("First underlying tensor component:", nt[0], sep='\n')
print("last column of 2nd underlying tensor component:", nt[1, :, -1], sep='\n')

# When indexing a nestedtensor's 0th dimension, the result is a regular tensor.
print(f"First underlying tensor component is nested: {nt[0].is_nested}")
```

```
First underlying tensor component:
tensor([[ 0., 1., 2., 3., 4., 5.],
 [ 6., 7., 8., 9., 10., 11.]], device='cuda:0')
last column of 2nd underlying tensor component:
tensor([ 5., 11., 17.], device='cuda:0')
First underlying tensor component is nested: False
```

An important note is that slicing in dimension 0 has not been supported yet.
Which means it not currently possible to construct a view that combines the underlying
tensor components.

## Nested Tensor Operations

As each operation must be explicitly implemented for nestedtensors,
operation coverage for nestedtensors is currently narrower than that of regular tensors.
For now, only basic operations such as index, dropout, softmax, transpose, reshape, linear, bmm are covered.
However, coverage is being expanded.
If you need certain operations, please file an [issue](https://github.com/pytorch/pytorch)
to help us prioritize coverage.

**reshape**

The reshape op is for changing the shape of a tensor.
Its full semantics for regular tensors can be found
[here](https://pytorch.org/docs/stable/generated/torch.reshape.html).
For regular tensors, when specifying the new shape,
a single dimension may be -1, in which case it is inferred
from the remaining dimensions and the number of elements.

The semantics for nestedtensors are similar, except that -1 no longer infers.
Instead, it inherits the old size (here 2 for `nt[0]` and 3 for `nt[1]`).
-1 is the only legal size to specify for a jagged dimension.

```
nt_reshaped = nt.reshape(2, -1, 2, 3)
print(f"{nt_reshaped=}")
```

```
nt_reshaped=nested_tensor([
 tensor([[[ 0., 1., 2.],
 [ 3., 4., 5.]],

 [[ 6., 7., 8.],
 [ 9., 10., 11.]]], device='cuda:0'),
 tensor([[[ 0., 1., 2.],
 [ 3., 4., 5.]],

 [[ 6., 7., 8.],
 [ 9., 10., 11.]],

 [[12., 13., 14.],
 [15., 16., 17.]]], device='cuda:0')
], device='cuda:0')
```

**transpose**

The transpose op is for swapping two dimensions of a tensor.
Its full semantics can be found
[here](https://pytorch.org/docs/stable/generated/torch.transpose.html).
Note that for nestedtensors dimension 0 is special;
it is assumed to be the batch dimension,
so transposes involving nestedtensor dimension 0 are not supported.

```
nt_transposed = nt_reshaped.transpose(1, 2)
print(f"{nt_transposed=}")
```

```
nt_transposed=nested_tensor([
 tensor([[[ 0., 1., 2.],
 [ 6., 7., 8.]],

 [[ 3., 4., 5.],
 [ 9., 10., 11.]]], device='cuda:0'),
 tensor([[[ 0., 1., 2.],
 [ 6., 7., 8.],
 [12., 13., 14.]],

 [[ 3., 4., 5.],
 [ 9., 10., 11.],
 [15., 16., 17.]]], device='cuda:0')
], device='cuda:0')
```

**others**

Other operations have the same semantics as for regular tensors.
Applying the operation on a nestedtensor is equivalent to
applying the operation to the underlying tensor components,
with the result being a nestedtensor as well.

```
nt_mm = torch.nested.nested_tensor([torch.randn((2, 3, 4)), torch.randn((2, 3, 5))], device=device)
nt3 = torch.matmul(nt_transposed, nt_mm)
print(f"Result of Matmul:\n {nt3}")

nt4 = F.dropout(nt3, 0.1)
print(f"Result of Dropout:\n {nt4}")

nt5 = F.softmax(nt4, -1)
print(f"Result of Softmax:\n {nt5}")
```

```
Result of Matmul:
 nested_tensor([
 tensor([[[ 0.7781, 1.7332, 2.5551, -1.7998],
 [ -6.3416, 0.6039, 3.3571, -21.6835]],

 [[ -3.0563, 1.1609, -6.8225, 19.4126],
 [ -7.3476, -0.8315, -15.4485, 44.0489]]], device='cuda:0'),
 tensor([[[ -0.7215, 3.0998, -0.2846, 4.7335, 3.6254],
 [-17.8239, 9.9335, 14.5221, 25.6358, 15.9261],
 [-34.9263, 16.7672, 29.3289, 46.5381, 28.2268]],

 [[ 5.9445, 3.1823, 7.7202, -15.5639, 9.8096],
 [ 13.5947, 9.8521, 19.5695, -38.9003, 20.3403],
 [ 21.2450, 16.5219, 31.4188, -62.2367, 30.8710]]], device='cuda:0')
], device='cuda:0')
Result of Dropout:
 nested_tensor([
 tensor([[[ 0.8646, 0.0000, 2.8390, -1.9998],
 [ -0.0000, 0.6710, 3.7301, -24.0928]],

 [[ -3.3959, 1.2899, -7.5805, 0.0000],
 [ -8.1640, -0.9239, -17.1650, 48.9432]]], device='cuda:0'),
 tensor([[[ -0.8017, 3.4442, -0.3162, 5.2595, 4.0282],
 [-19.8043, 11.0372, 16.1357, 28.4842, 17.6957],
 [-38.8070, 18.6302, 32.5877, 51.7090, 0.0000]],

 [[ 6.6050, 3.5359, 0.0000, -17.2933, 10.8996],
 [ 15.1053, 10.9468, 21.7439, -43.2226, 22.6003],
 [ 23.6055, 0.0000, 34.9098, -69.1519, 34.3011]]], device='cuda:0')
], device='cuda:0')
Result of Softmax:
 nested_tensor([
 tensor([[[1.1520e-01, 4.8526e-02, 8.2971e-01, 6.5685e-03],
 [2.2401e-02, 4.3820e-02, 9.3378e-01, 7.7075e-13]],

 [[7.1808e-03, 7.7842e-01, 1.0935e-04, 2.1429e-01],
 [1.5800e-25, 2.2030e-22, 1.9480e-29, 1.0000e+00]]], device='cuda:0'),
 tensor([[[1.5961e-03, 1.1144e-01, 2.5935e-03, 6.8453e-01, 1.9984e-01],
 [1.0679e-21, 2.6476e-08, 4.3361e-06, 9.9998e-01, 2.0634e-05],
 [4.8915e-40, 4.3061e-15, 4.9628e-09, 1.0000e+00, 3.4921e-23]],

 [[1.3451e-02, 6.2494e-04, 1.8206e-05, 5.6215e-13, 9.8591e-01],
 [3.8999e-04, 6.0961e-06, 2.9797e-01, 1.8180e-29, 7.0163e-01],
 [7.9792e-06, 4.4690e-16, 6.4764e-01, 0.0000e+00, 3.5236e-01]]],
 device='cuda:0')
], device='cuda:0')
```

## Why Nested Tensor

When data is sequential, it is often the case that each sample has a different length.
For example, in a batch of sentences, each sentence has a different number of words.
A common technique for handling varying sequences is to manually pad each data tensor
to the same shape in order to form a batch.
For example, we have 2 sentences with different lengths and a vocabulary
In order to represent his as single tensor we pad with 0 to the max length in the batch.

```
sentences = [["goodbye", "padding"],
 ["embrace", "nested", "tensor"]]
vocabulary = {"goodbye": 1.0, "padding": 2.0,
 "embrace": 3.0, "nested": 4.0, "tensor": 5.0}
padded_sentences = torch.tensor([[1.0, 2.0, 0.0],
 [3.0, 4.0, 5.0]])
nested_sentences = torch.nested.nested_tensor([torch.tensor([1.0, 2.0]),
 torch.tensor([3.0, 4.0, 5.0])])
print(f"{padded_sentences=}")
print(f"{nested_sentences=}")
```

```
padded_sentences=tensor([[1., 2., 0.],
 [3., 4., 5.]])
nested_sentences=nested_tensor([
 tensor([1., 2.]),
 tensor([3., 4., 5.])
])
```

This technique of padding a batch of data to its max length is not optimal.
The padded data is not needed for computation and wastes memory by allocating
larger tensors than necessary.
Further, not all operations have the same semnatics when applied to padded data.
For matrix multiplications in order to ignore the padded entries, one needs to pad
with 0 while for softmax one has to pad with -inf to ignore specific entries.
The primary objective of nested tensor is to facilitate operations on ragged
data using the standard PyTorch tensor UX, thereby eliminating the need
for inefficient and complex padding and masking.

```
padded_sentences_for_softmax = torch.tensor([[1.0, 2.0, float("-inf")],
 [3.0, 4.0, 5.0]])
print(F.softmax(padded_sentences_for_softmax, -1))
print(F.softmax(nested_sentences, -1))
```

```
tensor([[0.2689, 0.7311, 0.0000],
 [0.0900, 0.2447, 0.6652]])
nested_tensor([
 tensor([0.2689, 0.7311]),
 tensor([0.0900, 0.2447, 0.6652])
])
```

Let us take a look at a practical example: the multi-head attention component
utilized in [Transformers](https://arxiv.org/pdf/1706.03762.pdf).
We can implement this in such a way that it can operate on either padded
or nested tensors.

```
class MultiHeadAttention(nn.Module):
 """
 Computes multi-head attention. Supports nested or padded tensors.

 Args:
 E_q (int): Size of embedding dim for query
 E_k (int): Size of embedding dim for key
 E_v (int): Size of embedding dim for value
 E_total (int): Total embedding dim of combined heads post input projection. Each head
 has dim E_total // nheads
 nheads (int): Number of heads
 dropout_p (float, optional): Dropout probability. Default: 0.0
 """
 def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int,
 nheads: int, dropout_p: float = 0.0):
 super().__init__()
 self.nheads = nheads
 self.dropout_p = dropout_p
 self.query_proj = nn.Linear(E_q, E_total)
 self.key_proj = nn.Linear(E_k, E_total)
 self.value_proj = nn.Linear(E_v, E_total)
 E_out = E_q
 self.out_proj = nn.Linear(E_total, E_out)
 assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
 self.E_head = E_total // nheads

 def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
 """
 Forward pass; runs the following process:
 1. Apply input projection
 2. Split heads and prepare for SDPA
 3. Run SDPA
 4. Apply output projection

 Args:
 query (torch.Tensor): query of shape (N, L_t, E_q)
 key (torch.Tensor): key of shape (N, L_s, E_k)
 value (torch.Tensor): value of shape (N, L_s, E_v)

 Returns:
 attn_output (torch.Tensor): output of shape (N, L_t, E_q)
 """
 # Step 1. Apply input projection
 # TODO: demonstrate packed projection
 query = self.query_proj(query)
 key = self.key_proj(key)
 value = self.value_proj(value)

 # Step 2. Split heads and prepare for SDPA
 # reshape query, key, value to separate by head
 # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
 query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
 # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
 key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
 # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
 value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

 # Step 3. Run SDPA
 # (N, nheads, L_t, E_head)
 attn_output = F.scaled_dot_product_attention(
 query, key, value, dropout_p=dropout_p, is_causal=True)
 # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
 attn_output = attn_output.transpose(1, 2).flatten(-2)

 # Step 4. Apply output projection
 # (N, L_t, E_total) -> (N, L_t, E_out)
 attn_output = self.out_proj(attn_output)

 return attn_output
```

set hyperparameters following [the Transformer paper](https://arxiv.org/pdf/1706.03762.pdf)

```
N = 512
E_q, E_k, E_v, E_total = 512, 512, 512, 512
E_out = E_q
nheads = 8
```

except for dropout probability: set to 0 for correctness check

```
dropout_p = 0.0
```

Let us generate some realistic fake data from Zipf's law.

```
def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
 # generate fake corpus by unigram Zipf distribution
 # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
 sentence_lengths = np.empty(batch_size, dtype=int)
 for ibatch in range(batch_size):
 sentence_lengths[ibatch] = 1
 word = np.random.zipf(alpha)
 while word != 3 and word != 386 and word != 858:
 sentence_lengths[ibatch] += 1
 word = np.random.zipf(alpha)
 return torch.tensor(sentence_lengths)
```

Create nested tensor batch inputs

```
def gen_batch(N, E_q, E_k, E_v, device):
 # generate semi-realistic data using Zipf distribution for sentence lengths
 sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)

 # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
 # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
 # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
 query = torch.nested.nested_tensor([
 torch.randn(l.item(), E_q, device=device)
 for l in sentence_lengths
 ], layout=torch.jagged)

 key = torch.nested.nested_tensor([
 torch.randn(s.item(), E_k, device=device)
 for s in sentence_lengths
 ], layout=torch.jagged)

 value = torch.nested.nested_tensor([
 torch.randn(s.item(), E_v, device=device)
 for s in sentence_lengths
 ], layout=torch.jagged)

 return query, key, value, sentence_lengths

query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device)
```

Generate padded forms of query, key, value for comparison

```
def jagged_to_padded(jt, padding_val):
 # TODO: do jagged -> padded directly when this is supported
 return torch.nested.to_padded_tensor(
 torch.nested.nested_tensor(list(jt.unbind())),
 padding_val)

padded_query, padded_key, padded_value = (
 jagged_to_padded(t, 0.0) for t in (query, key, value)
)
```

Construct the model

```
mha = MultiHeadAttention(E_q, E_k, E_v, E_total, nheads, dropout_p).to(device=device)
```

Check correctness and performance

```
def benchmark(func, *args, **kwargs):
 torch.cuda.synchronize()
 begin = timeit.default_timer()
 output = func(*args, **kwargs)
 torch.cuda.synchronize()
 end = timeit.default_timer()
 return output, (end - begin)

output_nested, time_nested = benchmark(mha, query, key, value)
output_padded, time_padded = benchmark(mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
 output_padded[i, entry_length:] = 0.0

print("=== without torch.compile ===")
print("nested and padded calculations differ by", (jagged_to_padded(output_nested, 0.0) - output_padded).abs().max().item())
print("nested tensor multi-head attention takes", time_nested, "seconds")
print("padded tensor multi-head attention takes", time_padded, "seconds")

# warm up compile first...
compiled_mha = torch.compile(mha)
compiled_mha(query, key, value)
# ...now benchmark
compiled_output_nested, compiled_time_nested = benchmark(
 compiled_mha, query, key, value)

# warm up compile first...
compiled_mha(padded_query, padded_key, padded_value)
# ...now benchmark
compiled_output_padded, compiled_time_padded = benchmark(
 compiled_mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
 compiled_output_padded[i, entry_length:] = 0.0

print("=== with torch.compile ===")
print("nested and padded calculations differ by", (jagged_to_padded(compiled_output_nested, 0.0) - compiled_output_padded).abs().max().item())
print("nested tensor multi-head attention takes", compiled_time_nested, "seconds")
print("padded tensor multi-head attention takes", compiled_time_padded, "seconds")
```

```
=== without torch.compile ===
nested and padded calculations differ by 0.0
nested tensor multi-head attention takes 0.01278725400015901 seconds
padded tensor multi-head attention takes 0.009616458999971655 seconds
/var/lib/ci-user/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py:542: UserWarning: NestedTensor does not implement _stable_hash_for_caching. For PT2-compatible tensor subclasses, it is recommended to implement _stable_hash_for_caching(self) -> str for stable AOT autograd caching.
 warn_once(
/var/lib/ci-user/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py:320: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
 warnings.warn(
=== with torch.compile ===
nested and padded calculations differ by 0.0
nested tensor multi-head attention takes 0.0025538539998706256 seconds
padded tensor multi-head attention takes 0.009623788999988392 seconds
```

Note that without `torch.compile`, the overhead of the python subclass nested tensor
can make it slower than the equivalent computation on padded tensors. However, once
`torch.compile` is enabled, operating on nested tensors gives a multiple x speedup.
Avoiding wasted computation on padding becomes only more valuable as the percentage
of padding in the batch increases.

```
print(f"Nested speedup: {compiled_time_padded / compiled_time_nested:.3f}")
```

```
Nested speedup: 3.768
```

## Conclusion

In this tutorial, we have learned how to perform basic operations with nested tensors and
how implement multi-head attention for transformers in a way that avoids computation on padding.
For more information, check out the docs for the
[torch.nested](https://pytorch.org/docs/stable/nested.html) namespace.

## See Also

- [Accelerating PyTorch Transformers by replacing nn.Transformer with Nested Tensors and torch.compile](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)

**Total running time of the script:** (0 minutes 6.261 seconds)

[`Download Jupyter notebook: nestedtensor.ipynb`](../_downloads/0e22044ad9c3abd953c575aedd5e4595/nestedtensor.ipynb)

[`Download Python source code: nestedtensor.py`](../_downloads/f7cbd2a4028223851c72d0d81ce67897/nestedtensor.py)

[`Download zipped: nestedtensor.zip`](../_downloads/c7ac4597a28e534417e00d48ed79c97f/nestedtensor.zip)