How to Write a ``Quantizer`` for PyTorch 2 Export Quantization
================================================================

**Author**: `Leslie Fang <https://github.com/leslie-fang-intel>`_, `Weiwen Xia <https://github.com/Xia-Weiwen>`__, `Jiong Gong <https://github.com/jgong5>`__, `Kimish Patel <https://github.com/kimishpatel>`__, `Jerry Zhang <https://github.com/jerryzh168>`__

Prerequisites:
^^^^^^^^^^^^^^^^

Required:

-  `Torchdynamo concepts in PyTorch <https://pytorch.org/docs/stable/dynamo/index.html>`__
   
-  `Quantization concepts in PyTorch <https://pytorch.org/docs/master/quantization.html#quantization-api-summary>`__
   
-  `(prototype) PyTorch 2 Export Post Training Quantization <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html>`__

Optional:

-  `FX Graph Mode post training static quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`__
   
-  `BackendConfig in PyTorch Quantization FX Graph Mode <https://pytorch.org/tutorials/prototype/backend_config_tutorial.html?highlight=backend>`__
   
-  `QConfig and QConfigMapping in PyTorch Quantization FX Graph Mode <https://pytorch.org/tutorials/prototype/backend_config_tutorial.html#set-up-qconfigmapping-that-satisfies-the-backend-constraints>`__   

Introduction
^^^^^^^^^^^^^

`(prototype) PyTorch 2 Export Post Training Quantization <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html>`__ introduced the overall API for pytorch 2 export quantization, main difference from fx graph mode quantization in terms of API is that we made it explicit that quantiation is targeting a specific backend. So to use the new flow, backend need to implement a ``Quantizer`` class that encodes:
(1). What is supported quantized operator or patterns in the backend
(2). How can users express the way they want their floating point model to be quantized, for example, quantized the whole model to be int8 symmetric quantization, or quantize only linear layers etc.

Please see `here <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html#motivation-of-pytorch-2-export-quantization>`__ For motivations for the new API and ``Quantizer``.

An existing quantizer object defined for ``XNNPACK`` is in
`QNNPackQuantizer <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/pt2e/quantizer/xnnpack_quantizer.py>`__

Annotation API
^^^^^^^^^^^^^^^^^^^

``Quantizer`` uses annotation API to convey quantization intent for different operators/patterns.
Annotation API mainly consists of
`QuantizationSpec <https://github.com/pytorch/pytorch/blob/1ca2e993af6fa6934fca35da6970308ce227ddc7/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L38>`__
and 
`QuantizationAnnotation <https://github.com/pytorch/pytorch/blob/07104ca99c9d297975270fb58fda786e60b49b38/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L144>`__.

``QuantizationSpec`` is used to convey intent of how a tensor will be quantized,
e.g. dtype, bitwidth, min, max values, symmetric vs. asymmetric etc.
Furthermore, ``QuantizationSpec`` also allows quantizer to specify how a
tensor value should be observed, e.g. ``MinMaxObserver``, or ``HistogramObserver``
, or some customized observer.

``QuantizationAnnotation`` composed of ``QuantizationSpec`` objects is used to annotate input tensors
and output tensor of a pattern. Annotating input tensors is equivalent of annotating input edges,
while annotating output tensor is equivalent of annotating node. ``QuantizationAnnotation`` is a ``dataclass``
with several fields:

-  ``input_qspec_map`` field is of class ``Dict`` to map each input tensor (as input edge) to a ``QuantizationSpec``.
-  ``output_qspec`` field expresses the ``QuantizationSpec`` used to annotate the output tensor;
-  ``_annotated`` field indicates if this node has already been annotated by quantizer.

To conclude, annotation API requires quantizer to annotate edges (input tensors) or
nodes (output tensor) of the graph. Now, we will have a step-by-step tutorial for
how to use the annotation API with different types of ``QuantizationSpec``.

1. Annotate Common Operator Patterns
--------------------------------------------------------

In order to use the quantized pattern/operators, e.g. ``quantized add``,
backend developers will have intent to quantize (as expressed by ``QuantizationSpec``)
inputs, output of the pattern. Following is an example flow (take ``add`` operator as example)
of how this intent is conveyed in the quantization workflow with annotation API.

-  Step 1: Identify the original floating point pattern in the FX graph. There are
   several ways to identify this pattern: Quantizer may use a pattern matcher
   to match the operator pattern; Quantizer may go through the nodes from start to the end and compare
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

-  Step 3: Annotate the inputs and output of the pattern with ``QuantizationAnnotation``.
   In this example, we will create the ``QuantizationAnnotation`` object with the ``QuantizationSpec``
   created in above step 2 for two inputs and one output of the ``add`` node.

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

2. Annotate Operators that Shares Quantization Params
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

.. note::

   * Sharing is transitive

     Some tensors might be effectively using shared quantization spec due to:
     
     * Two nodes/edges are configured to use ``SharedQuantizationSpec``.
     * There is existing sharing of some nodes.
     
     For example, let's say we have two ``conv`` nodes ``conv1`` and ``conv2``, and both of them are fed into a ``cat``
     node: ``cat([conv1_out, conv2_out], ...)``. Let's say the output of ``conv1``, ``conv2``, and the first input of ``cat`` are configured
     with the same configurations of ``QuantizationSpec``. The second input of ``cat`` is configured to use ``SharedQuantizationSpec``
     with the first input.
     
     .. code-block::
     
       conv1_out: qspec1(dtype=torch.int8, ...)
       conv2_out: qspec1(dtype=torch.int8, ...)
       cat_input0: qspec1(dtype=torch.int8, ...)
       cat_input1: SharedQuantizationSpec((conv1, cat))  # conv1 node is the first input of cat
     
     First of all, the output of ``conv1`` is implicitly sharing quantization parameters (and observer object)
     with the first input of ``cat``, and the same is true for the output of ``conv2`` and the second input of ``cat``.
     Therefore, since the user configures the two inputs of ``cat`` to share quantization parameters, by transitivity,
     ``conv2_out`` and ``conv1_out`` will also be sharing quantization parameters. In the observed graph, you
     will see the following:
     
     .. code-block::
     
         conv1 -> obs -> cat
         conv2 -> obs   /

     and both ``obs`` will be the same observer instance.


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

3. Annotate Operators with Fixed Quantization Parameters
---------------------------------------------------------

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

4. Annotate Tensors with Derived Quantization Parameters
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

A Note on IR for PT2E Quantization Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IR means the intermediate representation of the model, for example, ``torch`` IR (``torch.nn`` modules, ``torch.nn.functional`` ops) or ``aten`` IR (``torch.ops.aten.linear``, ...). PT2E Quantization Flow is using pre autograd aten IR (the output of `torch.export` API) so that we support training. As is shown before, we need to match the operator or operator patterns before we can attach annotations on them, So the question is how do we match the pattern?

Motivation: Problem of Matching ``aten`` IR directly
--------------------------------------------------------

The most straightforward way might be matching ``aten`` IR directly.

Example::

  for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        relu_node = n
        maybe_conv_node = n.args[0]
        if (
            not isinstance(maybe_conv_node, Node)
            or maybe_conv_node.op != "call_function"
            or maybe_conv_node.target
            not in [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
            ]
        ):
            continue

        # annotate conv and relu nodes
        ...

However one problem for using this IR is that the representation might change if the PyTorch implementation for modules or functional ops changed. But this could be unexpected since modeling users typically assume that when the eager mode model code doesn't change, they should get the same model representation after program capture as well. One concrete effect for this problem is that if a ``Quantizer`` do annotations based on recognizing ``aten`` IR patterns, then it may fail to recognzing the pattern after PyTorch version update, and the same eager mode floating point may be left unquantized.

Recommendation: Use ``SubgraphMatcherWithNameNodeMap`` for pattern matching
-----------------------------------------------------------------------------
Because of this, we recommend people to recognize the pattern through ``SubgraphMatcherWithNameNodeMap`` (an improved version of ``SubgraphMatcher`` that makes it easier to query the nodes that people want to annotate), through capturing a ``torch`` IR pattern (with the same program capture used for capturing the floating point model), instead of using the ``aten`` IR pattern directly.

Example::

  def conv_relu_pattern(input, weight, bias):
      conv = torch.nn.functional.conv2d(input, weight, bias)
      output = torch.nn.functional.relu(conv)
      # returns an additional dict that includes a map from name to node that we want to annotate
      return relu, {"input": input, "weight": weight, "bias": bias, "output": output}

  matcher = SubgraphMatcherWithNameNodeMap(conv_relu_pattern)
  matches = matcher.match(model)
  for match in matches:
      # find input and output of the pattern
      # annotate the nodes
      name_node_map = match.name_node_map
      input_node = name_node_map["input"]
      weight_node = name_node_map["weight"]
      bias_node = name_node_map["bias"]
      output_node = name_node_map["relu"]
      input_node.users[0].meta["quantization_annotation"] = ...
      weight_node.users[0].meta["quantization_annotation"] = ...
      bias_node.users[0].meta["quantization_annotation"] = ...
      output_node.meta["quantization_annotation"] = ...

With this, the ``Quantizer`` will still be valid even when the implementation for nn modules and functionals changes, the ``aten`` IR for floating point model will change, but since we capture the pattern again instead of hardcoding the ``aten`` IR for the pattern, we'll get the updated ``aten`` IR as well and will still be able to match the pattern.

One caveat is that if inputs of the pattern has multiple users, we don't have a good way to identify which user node we want to annotate except for checking the aten op target.

Another caveat is that we need to make sure we have an exhaustive list of examples (e.g. 2D, 3D, 4D inputs, real v.s. symbolic inputs, training=True v.s. training=False etc.) for the pattern to make sure cover different possible ``aten`` IR outcomes captured from the ``torch`` IR pattern.

Note: We may provide some (pattern, list of example_inputs) or some pre-generated matcher object so people can just use them directly in the future.

Conclusion
^^^^^^^^^^^^^^^^^^^

With this tutorial, we introduce the new quantization path in PyTorch 2. Users can learn about
how to define a ``BackendQuantizer`` with the ``QuantizationAnnotation API`` and integrate it into the PyTorch 2 Export Quantization flow.
Examples of ``QuantizationSpec``, ``SharedQuantizationSpec``, ``FixedQParamsQuantizationSpec``, and ``DerivedQuantizationSpec``
are given for specific annotation use case. You can use `XNNPACKQuantizer <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/xnnpack_quantizer.py>`_ as an example to start implementing your own ``Quantizer``. After that please follow `this tutorial <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html>`_ to actually quantize your model.
