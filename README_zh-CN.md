# PyTorch 教程

[English](README.md) | 简体中文

所有的教程以Sphinx风格的文档形式呈现在以下地址：

## [https://pytorch.org/tutorials](https://pytorch.org/tutorials)



# 贡献

我们使用 sphinx-gallery 的 [notebook styled examples](https://sphinx-gallery.github.io/stable/tutorials/index.html) 来创建教程，语法非常简单。实质上，您编写一个稍微格式良好的 Python 文件，它将显示为一个 HTML 页面。此外，会自动生成一个 Jupyter 笔记本，并可在 Google Colab 上运行。

以下是您创建新教程的步骤（详细描述请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)）：

1. 创建一个 Python 文件，如果您希望在插入到文档中时执行该文件，请将文件保存为带有后缀 `tutorial` 的形式，文件名为 `your_tutorial.py`；
2. 根据难度级别，将其放入 `beginner_source`、`intermediate_source`、`advanced_source` 目录之一。如果是一个示例，将其添加到 `recipes_source` 中。对于展示不稳定原型功能的教程，请添加到 `prototype_source` 中；
3. 对于教程（除非是原型功能），将其包含在 `toctree` 指令中，并在 [index.rst](./index.rst) 中创建一个 `customcarditem`；
4. 对于教程（除非是原型功能），在 [index.rst file](https://github.com/pytorch/tutorials/blob/main/index.rst) 中创建一个缩略图，使用类似于 `.. customcarditem:: beginner/your_tutorial.html` 的命令。对于示例，创建一个缩略图在 [recipes_index.rst](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes_index.rst) 中。

如果您从 Jupyter notebook 开始，可以使用[此脚本](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe)将笔记本转换为 Python 文件。在转换并添加到项目后，请确保部分标题和其他内容按照逻辑顺序排列。

## 本地构建

教程构建非常庞大，需要使用 GPU。如果您的计算机没有 GPU 设备，您可以在不实际下载数据和运行教程代码的情况下预览 HTML 构建：

1. 运行以下命令安装所需的依赖项：`pip install -r requirements.txt`。

> 如果您想使用 `virtualenv`，在存储库的根目录下运行：`virtualenv venv`，然后运行 `source venv/bin/activate`。

- 如果您有一台配备 GPU 的笔记本电脑，可以使用 `make docs` 进行构建。这将下载数据，执行教程并构建文档到 `docs/` 目录。对于配备 GPU 的系统，这可能需要大约 60-120 分钟。如果您的系统没有安装 GPU，则请参考下一步；
- 您可以通过运行 `make html-noplot` 跳过计算密集型的图形生成，将基本 HTML 文档构建到 `_build/html`，这样，您可以快速预览教程。

> 如果您在使用 virtualenv 时从 /tutorials/src/pytorch-sphinx-theme 或 /venv/src/pytorch-sphinx-theme 处获得 **ModuleNotFoundError: No module named 'pytorch_sphinx_theme' make: [html-noplot] Error 2** 的错误，请运行 `python setup.py install`。

## 构建单个教程

您可以使用 `GALLERY_PATTERN` 环境变量构建单个教程。例如，要仅运行 `neural_style_transfer_tutorial.py`，请运行以下命令：

```
GALLERY_PATTERN="neural_style_transfer_tutorial.py" make html
```
或者

```
GALLERY_PATTERN="neural_style_transfer_tutorial.py" sphinx-build . _build
```

`GALLERY_PATTERN` 变量支持正则表达式。

## 关于为 PyTorch 文档和教程做贡献的说明

* 您可以在 PyTorch 存储库的 [README.md](https://github.com/pytorch/pytorch/blob/master/README.md) 文件中找到有关为 PyTorch 文档做贡献的信息；
* 附加信息可以在 [PyTorch CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md) 中找到。
