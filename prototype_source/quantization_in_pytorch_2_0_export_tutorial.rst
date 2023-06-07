(prototype) Quantization in PyTorch 2.0 Export Tutorial
==============================================================

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

-  `Understanding of the quantization concepts in PyTorch <https://pytorch.org/docs/master/quantization.html#quantization-api-summary>`__
-  `Understanding of FX graph mode post training static quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`__
-  `Understanding of torchdynamo concepts in PyTorch <https://pytorch.org/docs/stable/dynamo/index.html>`__

Previously in ``FX Graph Mode Quantization`` we were using ``QConfigMapping`` for users to specify how the model to be quantized
and ``BackendConfig`` to specify the supported ways of quantization in their backend.
This API covers most use cases relatively well, but the main problem is that this API is not fully extensible
with two main limitations:

-  Limitation around expressing quantization intentions for complicated operator patterns such as in the
   `discussion <https://github.com/pytorch/pytorch/issues/96288>`__ to support `conv add` fusion with oneDNN library.
   It also requires some changes to current already complicated pattern matching code such as in the
   `PR <https://github.com/pytorch/pytorch/pull/97122>`__ to support `conv add` fusion.
-  Limitation around supporting user's advanced intention to quantize their model. For example, ``FX Graph Mode Quantization``
   doesn't support this quantization intention: only quantize inputs and outputs when the ``linear`` has a third input.

To address these scalability issues, 
`Quantizer <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L160>`__
is introduced for quantization in PyTorch 2.0 export. ``Quantizer`` is a class that users can use to
programmably set the observer or fake quant objects for each node in the model graph. It adds flexibility
to the quantization API and allows modeling users and backend developers to configure quantization programmatically.
This will allow users to express how they want an operator pattern to be observed in a more explicit
way by annotating the appropriate nodes. To define a backend specific quantizer, user mainly need to override
several APIs:

-  `annotate method <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py#L269>`__
   is used to annotate nodes in the graph with observer or fake quant constructors to convey the desired way of quantization.
- `validate method <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py#L721>`__
   is used to validate if the annotated graph is supported by the backend.
- `set_global method <https://github.com/pytorch/pytorch/blob/3e988316b5976df560c51c998303f56a234a6a1f/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py#LL259C9-L259C19>`__
   is used to set the global ``QuantizationConfig`` object for this quantizer to specify how the model will be quantized.

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

An existing quantizer object defined for QNNPack/XNNPack is located in
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

In order to use the quantized operators, e.g. ``quantized add``,
backend developers will have intent to quantize (as expressed by
`QuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L38>`__
) input, output of the operator. Following is an example flow (with ``add``)
of how this intent is conveyed in the quantization workflow with node annotation API.

-  Step 1: Identify the original floating point ``add`` node in the FX graph. There are
   several ways to identify this node: 1. User may use a pattern matcher (e.g. SubgraphMatcher)
   to match the operator pattern. 2. User may go through the nodes from start to the end and compare
   the node's target type.
-  Step 2: Define the ``QuantizationSpec`` for two inputs and one output of the ``add`` node to specify
   how to quantize input tensors and output tensor which includes parameters of ``observer type``,
   ``dtype``, ``quant_min``, and ``quant_max`` etc.
-  Step 3: Annotate the inputs and output of the ``add`` node. User will create the ``QuantizationAnnotation``
   object and add it into ``add`` node's ``meta`` property.

::

    def _annotate_add(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        # Step1: Identify the ``add`` node in the original floating point FX graph.
        add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add])
        add_partitions = list(itertools.chain(*add_partitions.values()))
        for add_partition in add_partitions:
            add_node = add_partition.output_nodes[0]
            if _is_annotated([add_node]):
                continue

            act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = \
                HistogramObserver
            act_quantization_spec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
                is_dynamic=False,
                observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
            )

            # Step2: The ``add`` node has two inputs and one output. We define the ``QuantizationSpec``
            # for each input and output.
            input_act_qspec = act_quantization_spec
            output_act_qspec = act_quantization_spec

            input_qspec_map = {}
            input_act0 = add_node.args[0]
            if isinstance(input_act0, Node):
                input_qspec_map[input_act0] = input_act_qspec

            input_act1 = add_node.args[1]
            if isinstance(input_act1, Node):
                input_qspec_map[input_act1] = input_act_qspec

            # Step3: Annotate the inputs and outputs of the ``add`` node.
            add_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )

2. Annotate sharing qparams operators
--------------------------------------------------------

It is natural that users want to annotate a quantized model where quantization
parameters can be shared among some tensors explicitly. Two typical use cases are:

-  Example 1: One example is for ``add`` where having both inputs sharing quantization
   parameters makes operator implementation much easier. Without using of
   `SharedQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L90>`__,
   we have to annotate ``add`` as example in above section 1, in which two inputs of ``add``
   has different quantization parameters.
-  Example 2: Another example is that of sharing quantization parameters between inputs and output.
   This typically results from operators such as ``maxpool``, ``average_pool``, ``concat`` etc.

``SharedQuantizationSpec`` is designed for this use case to annotate tensors whose quantization
parameters are shared with other tensors. Input of ``SharedQuantizationSpec`` can be an input edge
or an output value. Input edge is the connection between input node and the node consuming the input,
so it's a Tuple[Node, Node]. Output value is an fx Node.

Now, we have a example to rewrite ``add`` annotation example with ``SharedQuantizationSpec``.

::

    def _annotate_add(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add])
        add_partitions = list(itertools.chain(*add_partitions.values()))
        for add_partition in add_partitions:
            add_node = add_partition.output_nodes[0]
            if _is_annotated([add_node]):
                continue

            act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = \
                HistogramObserver
            act_quantization_spec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
                is_dynamic=False,
                observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
            )
            act_qspec = act_quantization_spec

            input_qspec_map = {}
            input_act0 = add_node.args[0]
            input_act1 = add_node.args[1]

            share_qparams_with_input_act0_qspec = SharedQuantizationSpec((input_act0, add_node))

            input_qspec_map = {input_act0: act_qspec, input_act1: share_qparams_with_input_act0_qspec}

            add_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=act_qspec,
                _annotated=True,
            )

3. Annotate fixed qparams operators
--------------------------------------------------------

Another typical use case to annotate a quantized model is for tensors whose
quantization parmaters are known beforehand. For example, operator like ``sigmoid``, which has
predefined and fixed scale/zero_point at input and output tensors.
`FixedQParamsQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L90>`__
is designed for this use case. To use ``FixedQParamsQuantizationSpec``, users need to pass in parameters
of ``scale`` and ``zero_point`` explicitly.

::

    def _annotate_sigmoid(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        sigmoid_partitions = get_source_partitions(gm.graph, [torch.nn.Sigmoid])
        sigmoid_partitions = list(itertools.chain(*sigmoid_partitions.values()))
        for sigmoid_partition in sigmoid_partitions:
            sigmoid_node = sigmoid_partition.output_nodes[0]

            input_act = sigmoid_node.args[0]
            assert isinstance(input_act, Node)
            act_qspec = FixedQParamsQuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                scale=2.0 / 256.0,
                zero_point=128,
            )
            sigmoid_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={
                    input_act: act_qspec,
                },
                output_qspec=act_qspec,
                _annotated=True,
            )

4. Annotate tensor with derived quantization parameters
---------------------------------------------------------------

We also may need to define the constraint for tensors whose quantization parameters are derived from other tensors.
For example, if we want to annotate a convolution node, and define the ``scale`` of its bias input tensor
as product of the activation tensor's ``scale`` and weight tensor's ``scale``. We can use
`DerivedQuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L102>`__
to annotate this bias tensor.

::

    def _annotate_conv2d_derived_bias(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            node = conv_partition.output_nodes[0]
            input_act = node.args[0]
            weight = node.args[1]
            bias = node.args[2]

            act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = \
                HistogramObserver
            act_quantization_spec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
                is_dynamic=False,
                observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
            )
            weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PerChannelMinMaxObserver
            extra_args: Dict[str, Any] = {"eps": 2**-12}
            weight_quantization_spec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-127,
                quant_max=127,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
                is_dynamic=False,
                observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args),
            )
            act_qspec = act_quantization_spec
            weight_qspec = weight_quantization_spec

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
            input_qspec_map = {input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=act_qspec,
                _annotated=True,
            )

5. A Toy Example with Resnet18 
--------------------------------------------------------

To better understand the final example, here are some basic concepts before we move on to this part:

- `QuantizationSpec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L28-L66>`__
  defines the ``data type``, ``qscheme``, and other quantization parameters used to quantize a tensor.
- `QuantizationConfig <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L103-L109>`__
  consists of ``QuantizationSpec`` for activation, weight, and bias separately.
- When annotating the model, methods of
  `get_act_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L9>`__,
  `get_weight_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L26>`__, and
  `get_bias_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#LL42C5-L42C19>`__
  can be used to get the ``QuantizationSpec`` from ``QuantizationConfig`` for a specific node.

After above annotation methods defined with ``QuantizationAnnotation API``, we can now put them together to construct a ``BackendQuantizer``
to run a `toy example <https://gist.github.com/leslie-fang-intel/b78ed682aa9b54d2608285c5a4897cfc>`__
with Torchvision Resnet18.

6. Conclusion
---------------------

With this tutorial, we introduce the new quantization path in PyTorch 2.0. Users can learn about
how to define a ``BackendQuantizer`` with the ``QuantizationAnnotation API`` and integrate it into the quantization 2.0 flow.
Examples of ``QuantizationSpec``, ``SharedQuantizationSpec``, ``FixedQParamsQuantizationSpec``, and ``DerivedQuantizationSpec``
are given for specific annotation use case.
