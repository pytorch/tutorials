torchaudio.datasets
====================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers. 
For example: ::
    
    yesno_data = torchaudio.datasets.YESNO('.', download=True)
    data_loader = torch.utils.data.DataLoader(yesno_data, 
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=args.nThreads)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: torchaudio.datasets 


VCTK
~~~~~

.. autoclass:: VCTK
  :members: __getitem__
  :special-members:

YESNO
~~~~~~~~~~~~~

.. autoclass:: YESNO
  :members: __getitem__
  :special-members:
