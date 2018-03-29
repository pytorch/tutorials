Deep Learning for NLP with Pytorch
**********************************
**Author**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_

This tutorial will walk you through the key ideas of deep learning
programming using Pytorch. Many of the concepts (such as the computation
graph abstraction and autograd) are not unique to Pytorch and are
relevant to any deep learning toolkit out there.

I am writing this tutorial to focus specifically on NLP for people who
have never written code in any deep learning framework (e.g, TensorFlow,
Theano, Keras, Dynet). It assumes working knowledge of core NLP
problems: part-of-speech tagging, language modeling, etc. It also
assumes familiarity with neural networks at the level of an intro AI
class (such as one from the Russel and Norvig book). Usually, these
courses cover the basic backpropagation algorithm on feed-forward neural
networks, and make the point that they are chains of compositions of
linearities and non-linearities. This tutorial aims to get you started
writing deep learning code, given you have this prerequisite knowledge.

Note this is about *models*, not data. For all of the models, I just
create a few test examples with small dimensionality so you can see how
the weights change as it trains. If you have some real data you want to
try, you should be able to rip out any of the models from this notebook
and use them on it.


.. toctree::
    :hidden:

    /beginner/nlp/pytorch_tutorial
    /beginner/nlp/deep_learning_tutorial
    /beginner/nlp/word_embeddings_tutorial
    /beginner/nlp/sequence_models_tutorial
    /beginner/nlp/advanced_tutorial


.. galleryitem:: /beginner/nlp/pytorch_tutorial.py
    :intro: All of deep learning is computations on tensors, which are generalizations of a matrix that can be 

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: Deep learning consists of composing linearities with non-linearities in clever ways. The introduction of non-linearities allows

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: At this point, we have seen various feed-forward networks. That is, there is no state maintained by the network at all. 

.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: Dynamic versus Static Deep Learning Toolkits. Pytorch is a *dynamic* neural network kit. 


.. raw:: html

    <div style='clear:both'></div>
