# PyTorch Tutorials


All the tutorials are now presented as sphinx style documentation at:

## [https://pytorch.org/tutorials](https://pytorch.org/tutorials)



# Contributing

We use sphinx-gallery's [notebook styled examples](https://sphinx-gallery.github.io/stable/tutorials/index.html) to create the tutorials. Syntax is very simple. In essence, you write a slightly well formatted python file and it shows up as documentation page.

Here's how to create a new tutorial or recipe:
1. Create a notebook styled python file. If you want it executed while inserted into documentation, save the file with suffix `tutorial` so that file name is `your_tutorial.py`.
2. Put it in one of the beginner_source, intermediate_source, advanced_source based on the level. If it is a recipe, add to recipes_source.
2. For Tutorials (except if it is a prototype feature), include it in the TOC tree at index.rst
3. For Tutorials (except if it is a prototype feature), create a thumbnail in the [index.rst file](https://github.com/pytorch/tutorials/blob/master/index.rst) using a command like `.. customcarditem:: beginner/your_tutorial.html`. For Recipes, create a thumbnail in the [recipes_index.rst](https://github.com/pytorch/tutorials/blob/master/recipes_source/recipes_index.rst)

In case you prefer to write your tutorial in jupyter, you can use [this script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe) to convert the notebook to python file. After conversion and addition to the project, please make sure the sections headings etc are in logical order.

## Building

- Start with installing torch, torchvision, and your GPUs latest drivers. Install other requirements using `pip install -r requirements.txt`

> If you want to use `virtualenv`, make your environment in a `venv` directory like: `virtualenv ./venv`, then `source ./venv/bin/activate`.

- Then you can build using `make docs`. This will download the data, execute the tutorials and build the documentation to `docs/` directory. This will take about 60-120 min for systems with GPUs. If you do not have a GPU installed on your system, then see next step.
- You can skip the computationally intensive graph generation by running `make html-noplot` to build basic html documentation to `_build/html`. This way, you can quickly preview your tutorial.

> If you get **ModuleNotFoundError: No module named 'pytorch_sphinx_theme' make: *** [html-noplot] Error 2** from /tutorials/src/pytorch-sphinx-theme or /venv/src/pytorch-sphinx-theme (while using virtualenv), run `python setup.py install`. 


## About contributing to PyTorch Documentation and Tutorials
* You can find information about contributing to PyTorch documentation in the 
PyTorch Repo [README.md](https://github.com/pytorch/pytorch/blob/master/README.md) file. 
* Additional information can be found in [PyTorch CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md).