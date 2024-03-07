(prototype) Graph Mode Dynamic Quantization on BERT
===================================================


**Author**: `Supriya Rao <https://github.com/supriyar>`_

Introduction
------------

This tutorial introduces the steps to do post training Dynamic Quantization with Graph Mode Quantization. Dynamic quantization converts a float model to a quantized model with static int8 data types for the weights and dynamic quantization for the activations. The activations are quantized dynamically (per batch) to int8 while the weights are statically quantized to int8. Graph Mode Quantization flow operates on the model graph and requires minimal user intervention to quantize the model. To be able to use graph mode, the float model needs to be either traced or scripted first.

Advantages of graph mode quantization are:

- In graph mode, we can inspect the code that is executed in forward function (e.g. aten function calls) and quantization is achieved by module and graph manipulations.
- Simple quantization flow, minimal manual steps.
- Unlocks the possibility of doing higher level optimizations like automatic precision selection.

For additional details on Graph Mode Quantization please refer to the `Graph Mode Static Quantization Tutorial <https://pytorch.org/tutorials/prototype/graph_mode_static_quantization_tutorial.html>`_.

tl;dr The Graph Mode Dynamic `Quantization API <https://pytorch.org/docs/master/quantization.html#torch-quantization>`_:

.. code:: python

    import torch
    from torch.quantization import per_channel_dynamic_qconfig
    from torch.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(float_model) # or torch.jit.trace(float_model, input)

    quantized = quantize_dynamic_jit(ts_model, {'': per_channel_dynamic_qconfig})

1. Quantizing BERT Model
------------------------

The installaion steps and details about the model are identical to the steps in the Eager Mode Tutorial. Please refer to the tutorial `here <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html#install-pytorch-and-huggingface-transformers>`_ for more details.

1.1 Setup
^^^^^^^^^
Once all the necesessary packages are downloaded and installed we setup the code. We first start with the necessary imports and setup for the model.

.. code:: python

    import logging
    import numpy as np
    import os
    import random
    import sys
    import time
    import torch

    from argparse import Namespace
    from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                  TensorDataset)
    from tqdm import tqdm
    from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
    from transformers import glue_compute_metrics as compute_metrics
    from transformers import glue_output_modes as output_modes
    from transformers import glue_processors as processors
    from transformers import glue_convert_examples_to_features as convert_examples_to_features
    from torch.quantization import per_channel_dynamic_qconfig
    from torch.quantization import quantize_dynamic_jit

    def ids_tensor(shape, vocab_size):
        #  Creates a random int32 tensor of the shape within the vocab size
        return torch.randint(0, vocab_size, shape=shape, dtype=torch.int, device='cpu')

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.WARN)

    logging.getLogger("transformers.modeling_utils").setLevel(
       logging.WARN)  # Reduce logging

    print(torch.__version__)

    torch.set_num_threads(1)
    print(torch.__config__.parallel_info())

1.2 Download GLUE dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
Before running MRPC tasks we download the GLUE data by running this script and unpack it to a directory glue_data.

.. code:: shell

    python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'

1.3 Set global BERT configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run this experiment we first need a fine tuned BERT model. We provide the fined-tuned BERT model for MRPC task `here <https://download.pytorch.org/tutorial/MRPC.zip>`_. To save time, you can download the model file (~400 MB) directly into your local folder $OUT_DIR.


.. code:: python

    configs = Namespace()

    # The output directory for the fine-tuned model, $OUT_DIR.
    configs.output_dir = "./MRPC/"

    # The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
    configs.data_dir = "./glue_data/MRPC"

    # The model name or path for the pre-trained model.
    configs.model_name_or_path = "bert-base-uncased"
    # The maximum length of an input sequence
    configs.max_seq_length = 128

    # Prepare GLUE task.
    configs.task_name = "MRPC".lower()
    configs.processor = processors[configs.task_name]()
    configs.output_mode = output_modes[configs.task_name]
    configs.label_list = configs.processor.get_labels()
    configs.model_type = "bert".lower()
    configs.do_lower_case = True

    # Set the device, batch size, topology, and caching flags.
    configs.device = "cpu"
    configs.per_gpu_eval_batch_size = 8
    configs.n_gpu = 0
    configs.local_rank = -1
    configs.overwrite_cache = False

    # Set random seed for reproducibility.
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    set_seed(42)

    tokenizer = BertTokenizer.from_pretrained(
        configs.output_dir, do_lower_case=configs.do_lower_case)

    model = BertForSequenceClassification.from_pretrained(configs.output_dir, torchscript=True)
    model.to(configs.device)

1.4 Quantizing BERT model with Graph Mode Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.4.1 Script/Trace the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The input for graph mode quantization is a TorchScript model, so you'll need to either script or trace the model first. Currently, scripting the BERT model is not supported so we trace the model here.

We first identify the inputs to be passed to the model. Here, we trace the model with the largest possible input size that will be passed during the evaluation.
We choose a batch size of 8 and sequence lenght of 128 based on the input sizes passed in during the evaluation step below. Using the max possible shape during inference while tracing is a limitation of the huggingface BERT model as mentioned `here <https://huggingface.co/transformers/v2.3.0/torchscript.html#dummy-inputs-and-standard-lengths>`_.

We trace the model using ``torch.jit.trace``.

.. code:: python

    input_ids = ids_tensor([8, 128], 2)
    token_type_ids = ids_tensor([8, 128], 2)
    attention_mask = ids_tensor([8, 128], vocab_size=2)
    dummy_input = (input_ids, attention_mask, token_type_ids)
    traced_model = torch.jit.trace(model, dummy_input)

1.4.2 Specify qconfig_dict
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    qconfig_dict = {'': per_channel_dynamic_qconfig}

qconfig is a named tuple of the observers for activation and weight. For dynamic quantization we use a dummy activation observer to mimic the dynamic quantization process that happens in the operator during runtime. For the weight tensors we recommend using per-channel quantization which helps improve the final accuracy.
``qconfig_dict`` is a dictionary with names of sub modules as key and qconfig for that module as value, empty key means the qconfig will be applied to whole model unless it’s overwritten by more specific configurations, the qconfig for each module is either found in the dictionary or fallback to the qconfig of parent module.

Right now qconfig_dict is the only way to configure how the model is quantized, and it is done in the granularity of module, that is, we only support one type of qconfig for each module, and the qconfig for sub module will override the qconfig for parent module. For example, if we have

.. code::

    qconfig = {
        '' : qconfig_global,
        'sub' : qconfig_sub,
        'sub.fc1' : qconfig_fc,
        'sub.fc2': None
    }

Module ``sub.fc1`` will be configured with ``qconfig_fc``, and all other child modules in ``sub`` will be configured with ``qconfig_sub`` and ``sub.fc2`` will not be quantized. All other modules in the model will be quantized with qconfig_global

.. code:: python

    qconfig_dict = {'': per_channel_dynamic_qconfig}

1.4.3 Quantize the model (one-line API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We call the one line API (similar to eager mode) to perform quantization as follows.

.. code:: python

    quantized_model = quantize_dynamic_jit(traced_model, qconfig_dict)

2. Evaluation
-------------

We reuse the tokenize and evaluation function from Huggingface.

.. code:: python

    def evaluate(args, model, tokenizer, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1]}
                    labels = batch[3]
                    if args.model_type != 'distilbert':
                        inputs['input'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    logits = outputs[0]
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results

    def load_and_cache_examples(args, task, tokenizer, evaluate=False):
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            features = convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_length=args.max_seq_length,
                                                    output_mode=output_mode,)
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def time_model_evaluation(model, configs, tokenizer):
        eval_start_time = time.time()
        result = evaluate(configs, model, tokenizer, prefix="")
        eval_end_time = time.time()
        eval_duration_time = eval_end_time - eval_start_time
        print(result)
        print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


2.1 Check Model Size
^^^^^^^^^^^^^^^^^^^^

We print the model size to account for wins from quantization

.. code:: python

    def print_size_of_model(model):
        if isinstance(model, torch.jit.RecursiveScriptModule):
            torch.jit.save(model, "temp.p")
        else:
            torch.jit.save(torch.jit.script(model), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    print("Size of model before quantization")
    print_size_of_model(traced_model)
    print("Size of model after quantization")

    print_size_of_model(quantized_model)

.. code::

    Size of model before quantization
    Size (MB): 438.242141
    Size of model after quantization
    Size (MB): 184.354759

2.2 Run the evaluation
^^^^^^^^^^^^^^^^^^^^^^
We evaluate the FP32 and quantized model and compare the F1 score. Note that the performance numbers below are on a dev machine and they would likely improve on a production server.

.. code:: python

    time_model_evaluation(traced_model, configs, tokenizer)
    time_model_evaluation(quantized_model, configs, tokenizer)

.. code::

    FP32 model results -
    'f1': 0.901
    Time taken - 188.0s

    INT8 model results -
    'f1': 0.902
    Time taken - 157.4s

3. Debugging the Quantized Model
--------------------------------

We can debug the quantized model by passing in the debug option.

.. code::

    quantized_model = quantize_dynamic_jit(traced_model, qconfig_dict, debug=True)

If debug is set to True:

- We can access the attributes of the quantized model the same way as in a torchscript model, e.g. model.fc1.weight (might be harder if you use a module list or sequential).
- The arithmetic operations all occur in floating point with the numerics being identical to the final quantized model, allowing for debugging.

.. code:: python

    quantized_model_debug = quantize_dynamic_jit(traced_model, qconfig_dict, debug=True)

Calling ``quantize_dynamic_jit`` is equivalent to calling ``prepare_dynamic_jit`` followed by ``convert_dynamic_jit``. Usage of the one-line API is recommended. But if you wish to debug or analyze the model after each step, the multi-line API comes into use.

3.1. Evaluate the Debug Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Evaluate the debug model
    time_model_evaluation(quantized_model_debug, configs, tokenizer)

.. code::

    Size (MB): 438.406429

    INT8 (debug=True) model results -
    'f1': 0.897

Note that the accuracy of the debug version is close to, but not exactly the same as the non-debug version as the debug version uses floating point ops to emulate quantized ops and the numerics match is approximate.
This is the case only for per-channel quantization (we are working on improving this). Per-tensor quantization (using default_dynamic_qconfig) has exact numerics match between debug and non-debug version.

.. code:: python

    print(str(quantized_model_debug.graph))

Snippet of the graph printed -

.. code::

    %111 : Tensor = prim::GetAttr[name="bias"](%110)
    %112 : Tensor = prim::GetAttr[name="weight"](%110)
    %113 : Float(768:1) = prim::GetAttr[name="4_scale_0"](%110)
    %114 : Int(768:1) = prim::GetAttr[name="4_zero_point_0"](%110)
    %115 : int = prim::GetAttr[name="4_axis_0"](%110)
    %116 : int = prim::GetAttr[name="4_scalar_type_0"](%110)
    %4.quant.6 : Tensor = aten::quantize_per_channel(%112, %113, %114, %115, %116)
    %4.dequant.6 : Tensor = aten::dequantize(%4.quant.6)
    %1640 : bool = prim::Constant[value=1]()
    %input.5.scale.1 : float, %input.5.zero_point.1 : int = aten::_choose_qparams_per_tensor(%input.5, %1640)
    %input.5.quant.1 : Tensor = aten::quantize_per_tensor(%input.5, %input.5.scale.1, %input.5.zero_point.1, %74)
    %input.5.dequant.1 : Float(8:98304, 128:768, 768:1) = aten::dequantize(%input.5.quant.1)
    %119 : Tensor = aten::linear(%input.5.dequant.1, %4.dequant.6, %111)

We can see that there is no ``quantized::linear_dynamic`` in the model, but the numerically equivalent pattern of ``aten::_choose_qparams_per_tensor`` - ``aten::quantize_per_tensor`` - ``aten::dequantize`` - ``aten::linear``.

.. code:: python

    # Get the size of the debug model
    print_size_of_model(quantized_model_debug)

.. code::

    Size (MB): 438.406429

Size of the debug model is the close to the floating point model because all the weights are in float and not yet quantized and frozen, this allows people to inspect the weight.
You may access the weight attributes directly in the torchscript model. Accessing the weight in the debug model is the same as accessing the weight in a TorchScript model:

.. code:: python

    print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.weight)

.. code::

    tensor([[-0.0157,  0.0257, -0.0269,  ...,  0.0158,  0.0764,  0.0548],
            [-0.0325,  0.0345, -0.0423,  ..., -0.0528,  0.1382,  0.0069],
            [ 0.0106,  0.0335,  0.0113,  ..., -0.0275,  0.0253, -0.0457],
            ...,
            [-0.0090,  0.0512,  0.0555,  ...,  0.0277,  0.0543, -0.0539],
            [-0.0195,  0.0943,  0.0619,  ..., -0.1040,  0.0598,  0.0465],
            [ 0.0009, -0.0949,  0.0097,  ..., -0.0183, -0.0511, -0.0085]],
            grad_fn=<CloneBackward>)

Accessing the scale and zero_point for the corresponding weight can be done as follows -

.. code:: python

    print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.getattr('4_scale_0'))
    print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.getattr('4_zero_point_0'))

Since we use per-channel quantization, we get per-channel scales tensor.

.. code::

    tensor([0.0009, 0.0011, 0.0010, 0.0011, 0.0034, 0.0013, 0.0010, 0.0010, 0.0013,
            0.0012, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0009, 0.0015,
            0.0016, 0.0036, 0.0012, 0.0009, 0.0010, 0.0014, 0.0008, 0.0008, 0.0008,
            ...,
            0.0019, 0.0023, 0.0013, 0.0018, 0.0012, 0.0031, 0.0015, 0.0013, 0.0014,
            0.0022, 0.0011, 0.0024])

Zero-point tensor -

.. code::

    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ..,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           dtype=torch.int32)

4. Comparing Results with Eager Mode
------------------------------------

Following results show the F1 score and model size for Eager Mode Quantization of the same model by following the steps mentioned in the `tutorial <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html#evaluate-the-inference-accuracy-and-time>`_. Results show that Eager and Graph Mode Quantization on the model produce identical results.

.. code::

    FP32 model results -
    Size (MB): 438.016605
    'f1': 0.901

    INT8 model results -
    Size (MB): 182.878029
    'f1': 0.902

5. Benchmarking the Model
-------------------------

We benchmark the model with dummy input and compare the Float model with Eager and Graph Mode Quantized Model on a production server machine.

.. code:: python

    def benchmark(model):
        model = torch.jit.load(model)
        model.eval()
        torch.set_num_threads(1)
        input_ids = ids_tensor([8, 128], 2)
        token_type_ids = ids_tensor([8, 128], 2)
        attention_mask = ids_tensor([8, 128], vocab_size=2)
        elapsed = 0
        for _i in range(50):
            start = time.time()
            output = model(input_ids, token_type_ids, attention_mask)
            end = time.time()
            elapsed = elapsed + (end - start)
        print('Elapsed time: ', (elapsed / 50), ' s')
        return
    print("Running benchmark for Float model")
    benchmark(args.jit_model_path_float)
    print("Running benchmark for Eager Mode Quantized model")
    benchmark(args.jit_model_path_eager)
    print("Running benchmark for Graph Mode Quantized model")
    benchmark(args.jit_model_path_graph)

.. code::

    Running benchmark for Float model
    Elapsed time: 4.49 s
    Running benchmark for Eager Mode Quantized model
    Elapsed time: 2.67 s
    Running benchmark for Graph Mode Quantized model
    Elapsed time: 2.69 s
    As we can see both graph mode and eager mode quantized model have a similar speed up over the floating point model.

Conclusion
----------

In this tutorial, we demonstrated how to convert a well-known state-of-the-art NLP model like BERT into dynamic quantized model using graph mode with same performance as eager mode.
Dynamic quantization can reduce the size of the model while only having a limited implication on accuracy.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.
