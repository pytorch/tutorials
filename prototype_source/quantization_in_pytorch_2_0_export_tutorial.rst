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

-  Step1: Identify the original floating point ``add`` node in the FX graph. There are
   several ways to identify this node: 1. User may use a pattern matcher (e.g. SubgraphMatcher)
   to match the operator pattern. 2. User may go through the nodes from start to the end and compare
   the node's target type.
-  Step2: Define the ``QuantizationSpec`` for two inputs and one output of the ``add`` node to specify
   how to quantize input tensors and output tensor which includes parameters of ``observer type``,
   ``dtype``, ``quant_min``, and ``quant_max`` etc.
-  Step3: Annotate the inputs and output of the ``add`` node. User will create the ``QuantizationAnnotation``
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

After above annotation methods defined with ``QuantizationAnnotation API``, we can now put them together to construct a ``BackendQuantizer``
to run a example with Torchvision Resnet18. Here are some basic concepts before we move on to this example:

- `QuantizationSpec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L28-L66>`__
  defines the ``data type``, ``qscheme``, and other quantization parameters used to quantize a tensor.
- `QuantizationConfig <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L103-L109>`__
  consists of ``QuantizationSpec`` for activation, weight, and bias separately.
- When annotating the model, methods of
  `get_act_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L9>`__,
  `get_weight_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L26>`__, and
  `get_bias_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#LL42C5-L42C19>`__
  can be used to get the ``QuantizationSpec`` from ``QuantizationConfig`` for a specific node.

.. code:: ipython3

    import copy
    import itertools
    import operator
    from typing import Callable, Dict, List, Optional, Set, Any

    import torch
    import torch._dynamo as torchdynamo
    from torch.ao.quantization._pt2e.quantizer.utils import (
        _annotate_input_qspec_map,
        _annotate_output_qspec,
        get_input_act_qspec,
        get_output_act_qspec,
        get_bias_qspec,
        get_weight_qspec,
    )

    from torch.fx import Node

    from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

    from torch.ao.quantization._pt2e.quantizer.quantizer import (
        OperatorConfig,
        QuantizationConfig,
        QuantizationSpec,
        Quantizer,
        QuantizationAnnotation,
    )
    from torch.ao.quantization.observer import (
        HistogramObserver,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
    import torchvision
    from torch.ao.quantization._quantize_pt2e import (
        convert_pt2e,
        prepare_pt2e_quantizer,
    )

    def _mark_nodes_as_annotated(nodes: List[Node]):
        for node in nodes:
            if node is not None:
                if "quantization_annotation" not in node.meta:
                    node.meta["quantization_annotation"] = QuantizationAnnotation()
                node.meta["quantization_annotation"]._annotated = True

    def _is_annotated(nodes: List[Node]):
        annotated = False
        for node in nodes:
            annotated = annotated or (
                "quantization_annotation" in node.meta
                and node.meta["quantization_annotation"]._annotated
            )
        return annotated

    class BackendQuantizer(Quantizer):

        def __init__(self):
            super().__init__()
            self.global_config: QuantizationConfig = None  # type: ignore[assignment]
            self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

        def set_global(self, quantization_config: QuantizationConfig):
            """set global QuantizationConfig used for the backend.
            QuantizationConfig is defined in torch/ao/quantization/_pt2e/quantizer/quantizer.py.
            """
            self.global_config = quantization_config
            return self

        def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
            """annotate nodes in the graph with observer or fake quant constructors
            to convey the desired way of quantization.
            """
            global_config = self.global_config
            self.annotate_symmetric_config(model, global_config)

            return model

        def annotate_symmetric_config(
            self, model: torch.fx.GraphModule, config: QuantizationConfig
        ) -> torch.fx.GraphModule:
            self._annotate_linear(model, config)
            self._annotate_conv2d(model, config)
            self._annotate_maxpool2d(model, config)
            return model

        def _annotate_conv2d(
            self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
        ) -> None:
            conv_partitions = get_source_partitions(
                gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
            )
            conv_partitions = list(itertools.chain(*conv_partitions.values()))
            for conv_partition in conv_partitions:
                if len(conv_partition.output_nodes) > 1:
                    raise ValueError("conv partition has more than one output node")
                conv_node = conv_partition.output_nodes[0]
                if (
                    conv_node.op != "call_function"
                    or conv_node.target != torch.ops.aten.convolution.default
                ):
                    raise ValueError(f"{conv_node} is not an aten conv2d operator")
                # skip annotation if it is already annotated
                if _is_annotated([conv_node]):
                    continue

                input_qspec_map = {}
                input_act = conv_node.args[0]
                assert isinstance(input_act, Node)
                input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

                weight = conv_node.args[1]
                assert isinstance(weight, Node)
                input_qspec_map[weight] = get_weight_qspec(quantization_config)

                bias = conv_node.args[2]
                if isinstance(bias, Node):
                    input_qspec_map[bias] = get_bias_qspec(quantization_config)

                conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    output_qspec=get_output_act_qspec(quantization_config),
                    _annotated=True,
                )

        def _annotate_linear(
            self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
        ) -> None:
            module_partitions = get_source_partitions(
                gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
            )
            act_qspec = get_input_act_qspec(quantization_config)
            weight_qspec = get_weight_qspec(quantization_config)
            bias_qspec = get_bias_qspec(quantization_config)
            for module_or_fn_type, partitions in module_partitions.items():
                if module_or_fn_type == torch.nn.Linear:
                    for p in partitions:
                        act_node = p.input_nodes[0]
                        output_node = p.output_nodes[0]
                        weight_node = None
                        bias_node = None
                        for node in p.params:
                            weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                            if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                                weight_node = node
                            if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                                bias_node = node
                        if weight_node is None:
                            raise ValueError("No weight found in Linear pattern")
                        # find use of act node within the matched pattern
                        act_use_node = None
                        for node in p.nodes:
                            if node in act_node.users:  # type: ignore[union-attr]
                                act_use_node = node
                                break
                        if act_use_node is None:
                            raise ValueError(
                                "Could not find an user of act node within matched pattern."
                            )
                        if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                            _annotate_input_qspec_map(
                                act_use_node,
                                act_node,
                                act_qspec,
                            )
                        if bias_node and _is_annotated([bias_node]) is False:
                            _annotate_output_qspec(bias_node, bias_qspec)
                        if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                            _annotate_output_qspec(weight_node, weight_qspec)
                        if _is_annotated([output_node]) is False:
                            _annotate_output_qspec(output_node, act_qspec)
                        nodes_to_mark_annotated = list(p.nodes)
                        _mark_nodes_as_annotated(nodes_to_mark_annotated)

        def _annotate_maxpool2d(
            self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
        ) -> None:
            module_partitions = get_source_partitions(
                gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d]
            )
            maxpool_partitions = list(itertools.chain(*module_partitions.values()))
            for maxpool_partition in maxpool_partitions:
                output_node = maxpool_partition.output_nodes[0]
                maxpool_node = None
                for n in maxpool_partition.nodes:
                    if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                        maxpool_node = n
                if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
                    continue

                input_act = maxpool_node.args[0]  # type: ignore[union-attr]
                assert isinstance(input_act, Node)

                act_qspec = get_input_act_qspec(quantization_config)
                maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                    input_qspec_map={
                        input_act: act_qspec,
                    },
                    _annotated=True,
                )
                output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=act_qspec,
                    _input_output_share_observers=True,
                    _annotated=True,
                )

        def validate(self, model: torch.fx.GraphModule) -> None:
            """validate if the annotated graph is supported by the backend"""
            pass

        @classmethod
        def get_supported_operators(cls) -> List[OperatorConfig]:
            return []

    def get_symmetric_quantization_config():
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

        bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
        bias_quantization_spec = QuantizationSpec(
            dtype=torch.float,
            observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
        )
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            act_quantization_spec,
            weight_quantization_spec,
            bias_quantization_spec,
        )
        return quantization_config

    if __name__ == "__main__":
        example_inputs = (torch.randn(1, 3, 224, 224),)
        m = torchvision.models.resnet18().eval()
        m_copy = copy.deepcopy(m)
        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )    
        quantizer = BackendQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        m = prepare_pt2e_quantizer(m, quantizer)
        after_prepare_result = m(*example_inputs)
        m = convert_pt2e(m)
        print("converted module is: {}".format(m), flush=True)

6. Conclusion
---------------------

With this tutorial, we introduce the new quantization path in PyTorch 2.0. Users can learn about
how to define a ``BackendQuantizer`` with the ``QuantizationAnnotation API`` and integrate it into the quantization 2.0 flow.
Examples of ``QuantizationSpec``, ``SharedQuantizationSpec``, ``FixedQParamsQuantizationSpec``, and ``DerivedQuantizationSpec``
are given for specific annotation use case.
