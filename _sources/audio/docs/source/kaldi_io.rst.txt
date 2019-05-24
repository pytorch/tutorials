.. role:: hidden
    :class: hidden-section

torchaudio.kaldi_io
======================

.. currentmodule:: torchaudio.kaldi_io

To use this module, the dependency kaldi_io_ needs to be installed.
This is a light wrapper around ``kaldi_io`` that returns :class:`torch.Tensors`.

.. _kaldi_io: https://github.com/vesis84/kaldi-io-for-python

Vectors
-------

:hidden:`read_vec_int_ark`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_int_ark

:hidden:`read_vec_flt_scp`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_flt_scp

:hidden:`read_vec_flt_ark`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_vec_flt_ark

Matrices
--------

:hidden:`read_mat_scp`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_mat_scp

:hidden:`read_mat_ark`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: read_mat_ark
