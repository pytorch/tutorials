(prototype) Quantization in PyTorch 2.0 Export Tutorial (Work in Progress)
==============================================================

**Author**: `Leslie Fang <https://github.com/leslie-fang-intel>`_, `Weiwen Xia <https://github.com/Xia-Weiwen>`__, `Jiong Gong <https://github.com/jgong5>`__

Today we have `FX Graph Mode
Quantization <https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization>`__
which uses ``symbolic_trace`` to capture the model into a graph, and then
perform quantization transformations on top of the captured model. In a
similar way, for Quantization 2.0 flow, we will now use the PT2 Export
workflow to capture the model into a graph, and perform quantization
transformations on top of the ATen dialect graph. This approach is expected to
have significantly higher model coverage, better programmability, and
a simplified UX.

Prerequisites:
-----------------------

-  `Understanding of torchdynamo concepts in PyTorch <https://pytorch.org/docs/stable/dynamo/index.html>`__
-  `Understanding of the quantization concepts in PyTorch <https://pytorch.org/docs/master/quantization.html#quantization-api-summary>`__
-  `Understanding of FX Graph Mode post training static quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`__
-  `Understanding of BackendConfig in PyTorch Quantization FX Graph Mode <https://pytorch.org/tutorials/prototype/backend_config_tutorial.html?highlight=backend>`__
-  `Understanding of QConfigMapping in PyTorch Quantization FX Graph Mode <https://pytorch.org/tutorials/prototype/backend_config_tutorial.html#set-up-qconfigmapping-that-satisfies-the-backend-constraints>`__

Previously in ``FX Graph Mode Quantization`` we were using ``QConfigMapping`` for users to specify how the model to be quantized
and ``BackendConfig`` to specify the supported ways of quantization in their backend.
This API covers most use cases relatively well, but the main problem is that this API is not fully extensible
with two main limitations:

-  Limitation around expressing quantization intentions for complicated operator patterns such as in the discussion of
   `issue-96288 <https://github.com/pytorch/pytorch/issues/96288>`__ to support ``conv add`` fusion with oneDNN library.
   It also requires some changes to current already complicated pattern matching code such as in the
   `PR-97122 <https://github.com/pytorch/pytorch/pull/97122>`__ to support ``conv add`` fusion.
-  Limitation around supporting user's advanced quantization intention to quantize their model. For example, if backend
   developer only wants to quantize inputs and outputs when the ``linear`` has a third input, it requires co-work from quantization
   team and backend developer.

To address these scalability issues, 
`Quantizer <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L160>`__
is introduced for quantization in PyTorch 2.0 export. ``Quantizer`` is a class that users can use to
programmatically set the quantization specifications for input and output of each node in the model graph. It adds flexibility
to the quantization API and allows modeling users and backend developers to configure quantization programmatically.
This will allow users to express how they want an operator pattern to be observed in a more explicit
way by annotating the appropriate nodes. A backend specific quantizer inherited from base quantizer,
some methods that need to be implemented:

-  `annotate method <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py#L269>`__
   is used to annotate nodes in the graph with 
   `QuantizationAnnotation <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L144>`__
   objects to convey the desired way of quantization.

Imagine a backend developer who wishes to integrate a third-party backend
with PyTorch's quantization 2.0 flow. To accomplish this, they would only need
to define the backend specific quantizer. The high level architecture of
quantization 2.0 with quantizer could look like this:

::

    float_model(Python)                               Input
        \                                              /
         \                                            /
    —-------------------------------------------------------
    |                    Dynamo Export                     |
    —-------------------------------------------------------
                                |
                        FX Graph in ATen     QNNPackQuantizer,
                                |            or X86InductorQuantizer,
                                |            or <Other Backend Quantizer>
                                |                /
    —--------------------------------------------------------
    |                 prepare_pt2e_quantizer                |
    —--------------------------------------------------------
                                |
                         Calibrate/Train
                                |
    —--------------------------------------------------------
    |                      convert_pt2e                     |
    —--------------------------------------------------------
                                |
                    Reference Quantized Model
                                |
    —--------------------------------------------------------
    |                        Lowering                       |
    —--------------------------------------------------------
                                |
            Executorch, or Inductor, or <Other Backends>

Note: ``prepare_pt2e_quantizer`` will be updated to ``prepare_pt2e`` soon.

An existing quantizer object defined for QNNPack/XNNPack is in
`QNNPackQuantizer <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py>`__.
Taking QNNPackQuantizer as an example, the overall Quantization 2.0 flow could be:

::

    import torch
    import torch._dynamo as torchdynamo
    from torch.ao.quantization._quantize_pt2e import convert_pt2e, prepare_pt2e
    import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 10)

        def forward(self, x):
            return self.linear(x)

    example_inputs = (torch.randn(1, 5),)
    model = M().eval()

    # Step 1: Trace the model into an FX graph of flattened ATen operators
    exported_graph_module, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )

    # Step 2: Insert observers or fake quantize modules
    quantizer = qq.QNNPackQuantizer()
    operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)
    prepared_graph_module = prepare_pt2e_quantizer(exported_graph_module, quantizer)

    # Step 3: Quantize the model
    convered_graph_module = convert_pt2e(prepared_graph_module)

    # Step 4: Lower Reference Quantized Model into the backend

Inside the Quantizer, we will use the ``QuantizationAnnotation API``
to convey user's intent for what quantization spec to use and how to
observe certain tensor values in the prepare step. Now, we will have a step-by-step
tutorial for how to use the ``QuantizationAnnotation API`` with different types of
``QuantizationSpec``.

1. Annotate common operator patterns
--------------------------------------------------------

In order to use the quantized pattern/operators, e.g. ``quantized add``,
backend developers will have intent to quantize (as expressed by
`QuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L38>`__
) inputs, output of the pattern. Following is an example flow (take ``add`` operator as example)
of how this intent is conveyed in the quantization workflow with annotation API.

-  Step 1: Identify the original floating point pattern in the FX graph. There are
   several ways to identify this pattern: User may use a pattern matcher (e.g. SubgraphMatcher)
   to match the operator pattern; User may go through the nodes from start to the end and compare
   the node's target type to match the operator pattern. In this example, we can use the
   `get_source_partitions <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/fx/passes/utils/source_matcher_utils.py#L51>`__
   to match this pattern. The original floating point ``add`` pattern only contain a single ``add`` node.

::

    add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add])
    add_partitions = list(itertools.chain(*add_partitions.values()))
    for add_partition in add_partitions:
        add_node = add_partition.output_nodes[0]

-  Step 2: Define the ``QuantizationSpec`` for inputs and output of the pattern. ``QuantizationSpec``
   defines the ``data type``, ``qscheme``, and other quantization parameters about users' intent of
   how to observe or fake quantize a tensor.

::

    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
    )

    input_act_qspec = act_quantization_spec
    output_act_qspec = act_quantization_spec

-  Step 3: Annotate the inputs and output of the pattern with
   `QuantizationAnnotation <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L144>`__.
   ``QuantizationAnnotation`` is a ``dataclass`` with several fields as: ``input_qspec_map`` field is of class ``Dict``
   to map each input ``Node`` to a ``QuantizationSpec``; ``output_qspec`` field expresses the ``QuantizationSpec`` used for
   output node; ``_annotated`` field indicates if this node has already been annotated by quantizer.
   In this example, we will create the ``QuantizationAnnotation`` object with the ``QuantizationSpec`` objects
   created in above step 2 for two inputs and one output of ``add`` node.

::

    input_qspec_map = {}
    input_act0 = add_node.args[0]
    input_qspec_map[input_act0] = input_act_qspec

    input_act1 = add_node.args[1]
    input_qspec_map[input_act1] = input_act_qspec
         
    add_node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )

After we annotate the ``add`` node like this, in the following up quantization flow, ``HistogramObserver`` will
be inserted at its two input nodes and one output node in prepare phase. And ``HistogramObserver`` will be substituted with
``quantize`` node and ``dequantize`` node in the convert phase.

2. Annotate sharing qparams operators
--------------------------------------------------------

It is natural that users want to annotate a quantized model where quantization
parameters can be shared among some tensors explicitly. Two typical use cases are:

-  Example 1: One example is for ``add`` where having both inputs sharing quantization
   parameters makes operator implementation much easier. Without using of
   `SharedQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L90>`__,
   we must annotate ``add`` as example in above section 1, in which two inputs of ``add``
   has different quantization parameters.
-  Example 2: Another example is that of sharing quantization parameters between inputs and output.
   This typically results from operators such as ``maxpool``, ``average_pool``, ``concat`` etc.

``SharedQuantizationSpec`` is designed for this use case to annotate tensors whose quantization
parameters are shared with other tensors. Input of ``SharedQuantizationSpec`` is an ``EdgeOrNode`` object which 
can be an input edge or an output value. 

-  Input edge is the connection between input node and the node consuming the input,
   so it's a ``Tuple[Node, Node]``.
-  Output value is an FX ``Node``.

Now, if we want to rewrite ``add`` annotation example with ``SharedQuantizationSpec`` to indicate
two input tensors as sharing quantization parameters. We can define its ``QuantizationAnnotation``
as this:

-  Step 1: Identify the original floating point pattern in the FX graph. We can use the same
   methods introduced in ``QuantizationSpec`` example to identify the ``add`` pattern.
-  Step 2: Annotate input_act0 of ``add`` with ``QuantizationSpec``.
-  Step 3: Create a ``SharedQuantizationSpec`` object with input edge defined as ``(input_act0, add_node)`` which means to
   share the observer used for this edge. Then, user can annotate input_act1 with this ``SharedQuantizationSpec``
   object.

::

    input_qspec_map = {}
    share_qparams_with_input_act0_qspec = SharedQuantizationSpec((input_act0, add_node))
    input_qspec_map = {input_act0: act_quantization_spec, input_act1: share_qparams_with_input_act0_qspec}

    add_node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=act_quantization_spec,
        _annotated=True,
    )

3. Annotate fixed qparams operators
--------------------------------------------------------

Another typical use case to annotate a quantized model is for tensors whose
quantization parameters are known beforehand. For example, operator like ``sigmoid``, which has
predefined and fixed scale/zero_point at input and output tensors.
`FixedQParamsQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L90>`__
is designed for this use case. To use ``FixedQParamsQuantizationSpec``, users need to pass in parameters
of ``scale`` and ``zero_point`` explicitly.

-  Step 1: Identify the original floating point pattern in the FX graph. We can use the same
   methods introduced in ``QuantizationSpec`` example to identify the ``sigmoid`` pattern.
-  Step 2: Create ``FixedQParamsQuantizationSpec`` object with inputs of fixed ``scale``, ``zero_point`` value.
   These values will be used to create the ``quantize`` node and ``dequantize`` node in the convert phase.
-  Step 3: Annotate inputs and output to use this ``FixedQParamsQuantizationSpec`` object.

::

    act_qspec = FixedQParamsQuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
        scale=1.0 / 256.0,
        zero_point=0,
    )
    sigmoid_node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map={input_act: act_qspec},
        output_qspec=act_qspec,
        _annotated=True,
    )

4. Annotate tensor with derived quantization parameters
---------------------------------------------------------------

Another use case is to define the constraint for tensors whose quantization parameters are derived from other tensors.
For example, if we want to annotate a convolution node, and define the ``scale`` of its bias input tensor
as product of the activation tensor's ``scale`` and weight tensor's ``scale``. We can use
`DerivedQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L102>`__
to annotate this conv node.

-  Step 1: Identify the original floating point pattern in the FX graph. We can use the same
   methods introduced in ``QuantizationSpec`` example to identify the ``convolution`` pattern.
-  Step 2: Define ``derive_qparams_fn`` function, it accepts list of ``ObserverOrFakeQuantize`` (
   `ObserverBase <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/ao/quantization/observer.py#L124>`__
   or `FakeQuantizeBase <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/ao/quantization/fake_quantize.py#L60>`__)
   as input. From each ``ObserverOrFakeQuantize`` object, user can get the ``scale``, ``zero point`` value.
   User can define its heuristic about how to derive new ``scale``, ``zero point`` value based on the
   quantization parameters calculated from the observer or fake quant instances.
-  Step 3: Define ``DerivedQuantizationSpec`` obejct, it accepts inputs of: list of ``EdgeOrNode`` objects.
   The observer corresponding to each ``EdgeOrNode`` object will be passed into the ``derive_qparams_fn`` function;
   ``derive_qparams_fn`` function; several other quantization parameters such as ``dtype``, ``qscheme``.
-  Step 4: Annotate the inputs and output of this conv node with ``QuantizationAnnotation``.

::

    def derive_qparams_fn(obs_or_fqs: List[ObserverOrFakeQuantize]) -> Tuple[Tensor, Tensor]:
        assert len(obs_or_fqs) == 2, \
            "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(len(obs_or_fq))
        act_obs_or_fq = obs_or_fqs[0]
        weight_obs_or_fq = obs_or_fqs[1]
        act_scale, act_zp = act_obs_or_fq.calculate_qparams()
        weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
        return torch.tensor([act_scale * weight_scale]).to(torch.float32), torch.tensor([0]).to(torch.int32)

    bias_qspec = DerivedQuantizationSpec(
        derived_from=[(input_act, node), (weight, node)],
        derive_qparams_fn=derive_qparams_fn,
        dtype=torch.int32,
        quant_min=-2**31,
        quant_max=2**31 - 1,
        qscheme=torch.per_tensor_symmetric,
    )
    input_qspec_map = {input_act: act_quantization_spec, weight: weight_quantization_spec, bias: bias_qspec}
    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=act_quantization_spec,
        _annotated=True,
    )

5. A Toy Example with Resnet18 
--------------------------------------------------------

After above annotation methods defined with ``QuantizationAnnotation API``, we can now put them together to construct a ``BackendQuantizer``
and run a `toy example <https://gist.github.com/leslie-fang-intel/b78ed682aa9b54d2608285c5a4897cfc>`__
with ``Torchvision Resnet18``. To better understand the final example, here are the classes and utility
functions that are used in the example:

-  `QuantizationConfig <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L103-L109>`__
   consists of ``QuantizationSpec`` for activation, weight, and bias separately.
-  When annotating the model,
   `get_input_act_qspec <https://github.com/pytorch/pytorch/blob/47cfcf566ab76573452787335f10c9ca185752dc/torch/ao/quantization/_pt2e/quantizer/utils.py#L10>`__,
   `get_output_act_qspec <https://github.com/pytorch/pytorch/blob/47cfcf566ab76573452787335f10c9ca185752dc/torch/ao/quantization/_pt2e/quantizer/utils.py#L23>`__,
   `get_weight_qspec <https://github.com/pytorch/pytorch/blob/47cfcf566ab76573452787335f10c9ca185752dc/torch/ao/quantization/_pt2e/quantizer/utils.py#L36>`__, and
   `get_bias_qspec <https://github.com/pytorch/pytorch/blob/47cfcf566ab76573452787335f10c9ca185752dc/torch/ao/quantization/_pt2e/quantizer/utils.py#L53>`__
   can be used to get the ``QuantizationSpec`` from ``QuantizationConfig`` for a specific pattern.

6. Conclusion
---------------------

With this tutorial, we introduce the new quantization path in PyTorch 2.0. Users can learn about
how to define a ``BackendQuantizer`` with the ``QuantizationAnnotation API`` and integrate it into the quantization 2.0 flow.
Examples of ``QuantizationSpec``, ``SharedQuantizationSpec``, ``FixedQParamsQuantizationSpec``, and ``DerivedQuantizationSpec``
are given for specific annotation use case.
