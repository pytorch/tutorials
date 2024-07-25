import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Timer, Compare

def profile(fn, inputs):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(activities=activities, with_stack=True) as prof:
        fn(*inputs)

    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

def compute_speedup(fn, inputs, device, times=100):
    lst = []

    fn = fn._torchdynamo_orig_callable
    fn_opt = torch.compile(fullgraph=True)(fn)
    fx_g = make_fx(fn)

    for nt in (1, 2, 4, 8, 16):
        opt = Timer(
            setup='fn_opt(*inputs)',
            stmt='fn_opt(*inputs)',
            globals={'fn_opt': fn_opt, 'inputs': inputs},
            label=fn.__name__,
            sub_label='@torch.compile',
            description=device,
            num_threads=nt,
        ).timeit(times)

        fx = Timer(
            setup='fx_g(*inputs)',
            stmt='fx_g(*inputs)',
            globals={'fx_g': fx_g, 'inputs': inputs},
            label=fn.__name__,
            sub_label='make_fx',
            description=device,
            num_threads=nt,
        ).timeit(times)

        eager = Timer(
            setup='fn(*inputs)',
            stmt='fn(*inputs)',
            globals={'fn': fn, 'inputs': inputs},
            label=fn.__name__,
            sub_label='eager',
            description=device,
            num_threads=nt,
        ).timeit(times)
        lst.extend([opt, fx, eager])

    Compare(lst).print()