(prototype) PyTorch Quantization 2.0 Tutorial
==========================================

Today we have `FX Graph Mode
Quantization <https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization>`__
which uses symbolic_trace to capture the model into a graph, and then
perform quantization transformations on top of the captured model. In a
similar way, for Quantization 2.0 flow, we will now use the PT2 Export
workflow to capture the model into a graph, and perform quantizations
transformations on top of the ATen dialect graph. This is expected to
have significantly higher model coverage, better programmability, and
a simplified UX.

Suppose we are a backend developer and we wish to integrate our backend
with PyTorch's quantization 2.0 flow. We only need to define the backend
specific quantizer. The high level arch of quantization 2.0 with quantizer could be: 

.. image:: /_static/img/quantization/pytorch_quantization_2_0_diagram.png

An existing quantizer object defined for QNNPack/XNNPack is here 
`QNNPackQuantizer <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py>`__.
Taking QNNPackQuantizer as example, the overall Quantization 2.0 flow could be:

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

Inside the Quantizer, we will use the `QuantizationAnnotation API <https://docs.google.com/document/d/1tjIsL7-uVgm_1bv_kUK7iovP6G1D5zcbzwEcmYEG2Js/edit#>`__
to convey user's intent for what quantization spec to use and how to
observe certain tensor values in the prepare step. Now, we will have a step by step
tutorial for how to use the `QuantizationAnnotation API` to create a quantizer.

1. Define QuantizationConfig
--------------------------------------------------------

QuantizationConfig defines the data type and qscheme for activation, weight and bias.
`QuantizationConfig <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L103-L109>`__ is defined here.
It consists of `QuantizationSpec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L28-L66>`__ defined for activation, weight and bias.
When annotating the model, methods of `get_act_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L9>`__,
`get_weight_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#L26>`__,
`get_bias_qspec <https://github.com/pytorch/pytorch/blob/73fd7235ad25ff061c087fa4bafc6e8df4d9c299/torch/ao/quantization/_pt2e/quantizer/utils.py#LL42C5-L42C19>`__
are used to get the `QuantizationSpec` from `QuantizationConfig` for the specific node. Then corresponding observer will been created
based on the node's `QuantizationSpec`.
Suppose we want to define:

-  Activation: `int8` data type, `per_tensor_affine` quantization, `HistogramObserver`
-  Weight    : `int8` data type, `per_channel_symmetric` quantization, `PerChannelMinMaxObserver`
-  Bias      : `float` data type, `PlaceholderObserver`

::

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
            act_quantization_spec, weight_quantization_spec, bias_quantization_spec
        )
        return quantization_config

2. Define the BackendQuantizer
--------------------------------------------------------

Then we will define the skeleton of a BackendQuantizer. The annotatation methods for each operation will be
defined later.

::

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
            for node in reversed(model.graph.nodes):
                # The annotation methods for each op will defined later
                pass
            return model

        def validate(self, model: torch.fx.GraphModule) -> None:
            """validate the annotated graph is supported by the backend"""
            pass

        @classmethod
        def get_supported_operators(cls) -> List[OperatorConfig]:
            """return the operator list which is supported by the backend"""
            return []

3. Annotate common operator patterns
--------------------------------------------------------

Now we will start to define the annotatation methods inside quantizer. For common operators like `conv2d`, we can use `QuantizationSpec` to
annotate the input, weight, bias and output.

::

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
            input_qspec_map[input_act] = get_act_qspec(quantization_config)

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = get_weight_qspec(quantization_config)

            bias = conv_node.args[2]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)

            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_act_qspec(quantization_config),
                _annotated=True,
            )

4. Annotate sharing qparams operators
--------------------------------------------------------

For operator such as `add` and `cat`, which we want the two inputs sharing
quantization parameters, we can use the `SharedQuantizationSpec` to make the two inputs
sharing the same quantization parameters.

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
            act_qspec = get_act_qspec(quantization_config)

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

5. Annotate fixed qparams operators
--------------------------------------------------------

For operator such as `sigmoid`, which has predefined and fixed scale/zero_point,
we can use fixed parameters for it with `FixedQParamsQuantizationSpec`.

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

6. Annotate tensor with derived quantization parameters
--------------------------------------------------------

`DerivedQuantizationSpec` is the quantization spec for the Tensors whose quantization parameters are derived from other Tensors.
For example, we want to define the scale, zp for bias derived from activation and weight of convolution node.

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
            act_qspec = get_act_qspec(quantization_config)
            weight_qspec = get_weight_qspec(quantization_config)

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

7. A Toy Example with Resnet18 
--------------------------------------------------------

After above annotation methods defined with `QuantizationAnnotation API`, we can now put them together for the BackendQuantizer
to run a example with Torchvision Resnet18.

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
        get_act_qspec,
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
            self.global_config = quantization_config
            return self

        def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
            """just handling global spec for now"""
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
                input_qspec_map[input_act] = get_act_qspec(quantization_config)

                weight = conv_node.args[1]
                assert isinstance(weight, Node)
                input_qspec_map[weight] = get_weight_qspec(quantization_config)

                bias = conv_node.args[2]
                if isinstance(bias, Node):
                    input_qspec_map[bias] = get_bias_qspec(quantization_config)

                conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    output_qspec=get_act_qspec(quantization_config),
                    _annotated=True,
                )

        def _annotate_linear(
            self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
        ) -> None:
            module_partitions = get_source_partitions(
                gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
            )
            act_qspec = get_act_qspec(quantization_config)
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

                act_qspec = get_act_qspec(quantization_config)
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
            act_quantization_spec, weight_quantization_spec, bias_quantization_spec
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
