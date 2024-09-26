Deep Learning for NLP with Pytorch
----------------------------------

These tutorials will walk you through the key ideas of deep learning
programming using Pytorch. Many of the concepts (such as the computation
graph abstraction and autograd) are not unique to Pytorch and are
relevant to any deep learning toolkit out there.

They are focused specifically on NLP for people who have never written
code in any deep learning framework (e.g, TensorFlow,Theano, Keras, DyNet).
The tutorials assumes working knowledge of core NLP problems: part-of-speech
tagging, language modeling, etc. It also assumes familiarity with neural
networks at the level of an intro AI class (such as one from the Russel and
Norvig book). Usually, these courses cover the basic backpropagation algorithm
on feed-forward neural networks, and make the point that they are chains of
compositions of linearities and non-linearities. This tutorial aims to get
you started writing deep learning code, given you have this prerequisite
knowledge.

Note these tutorials are about *models*, not data. For all of the models,
a few test examples are created with small dimensionality so you can see how
the weights change as it trains. If you have some real data you want to
try, you should be able to rip out any of the models from this notebook
and use them on it.

1. pytorch_tutorial.py
	Introduction to PyTorch
	https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html

2. deep_learning_tutorial.py
	Deep Learning with PyTorch
	https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html

3. word_embeddings_tutorial.py
	Word Embeddings: Encoding Lexical Semantics
	https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

4. sequence_models_tutorial.py
	Sequence Models and Long Short-Term Memory Networks
	https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

5. advanced_tutorial.py
	Advanced: Making Dynamic Decisions and the Bi-LSTM CRF
	https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
