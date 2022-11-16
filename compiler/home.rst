Dynamo, Inductor and Compilers Tutorials
===========================================

TorchDynamo makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator ``torch._dynamo.optimize()``


.. _learn-dynamo:

Getting Started
---------

.. grid:: 2

     .. grid-item-card:: :octicon:`file-code;1em`
        Intro to TorchDynamo
        :link: https://pytorch.org/docs/stable/dynamo.html
        :link-type: url
        A step by step guide to installing and using dynamo and inductor
        +++
        :octicon:`code;1em` Code :octicon:`square-fill;1em` :octicon:`video;1em` Video

      .. grid-item-card:: :octicon:`file-code;1em`
        Troubleshooting guide
        :link: https://pytorch.org/tutorials/beginner/DynamoTroubleshooting.rst.html
        :link-type: url

        Troubleshooting guide covering tools that will help you debug the most common issues
        +++
        :octicon:`code;1em` Code :octicon:`square-fill;1em` :octicon:`video;1em` Video


.. _learn-internals:

Internals
----------

.. grid:: 2

     .. grid-item-card:: :octicon:`file-code;1em`
        A Deeper Dive into dynamo
        :link: https://pytorch.org/tutorials/advanced/DynamoDeeperDive.html
        :link-type: url

        This guide will help you better understand what torchdynamo is doing under the hood
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Guards Overview
        :link: https://pytorch.org/tutorials/advanced/DynamoGuardsOverview.html
        :link-type: url

        Guards are an important concept in dynamo that determine when something needs to be recompiled
        +++
        :octicon:`code;1em` Code

