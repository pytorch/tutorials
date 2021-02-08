from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
import re
import os
import sphinx_gallery

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class IncludeDirective(Directive):
    """Include source file without docstring at the top of file.

    Implementation just replaces the first docstring found in file
    with '' once.

    Example usage:

    .. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py

    """

    # defines the parameter the directive expects
    # directives.unchanged means you get the raw value from RST
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False
    add_index = False

    docstring_pattern = r'"""(?P<docstring>(?:.|[\r\n])*?)"""\n'
    docstring_regex = re.compile(docstring_pattern)

    def run(self):
        document = self.state.document
        env = document.settings.env
        rel_filename, filename = env.relfn2path(self.arguments[0])

        try:
            text = open(filename).read()
            text_no_docstring = self.docstring_regex.sub('', text, count=1)

            code_block = nodes.literal_block(text=text_no_docstring)
            return [code_block]
        except FileNotFoundError as e:
            print(e)
            return []


class GalleryItemDirective(Directive):
    """
    Create a sphinx gallery thumbnail for insertion anywhere in docs.

    Optionally, you can specify the custom figure and intro/tooltip for the
    thumbnail.

    Example usage:

    .. galleryitem:: intermediate/char_rnn_generation_tutorial.py
        :figure: _static/img/char_rnn_generation.png
        :intro: Put your custom intro here.

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'figure': directives.unchanged,
                   'intro': directives.unchanged}
    has_content = False
    add_index = False

    def run(self):
        args = self.arguments
        fname = args[-1]

        env = self.state.document.settings.env
        fname, abs_fname = env.relfn2path(fname)
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)

        try:
            if 'intro' in self.options:
                intro = self.options['intro'][:195] + '...'
            else:
                _, blocks = sphinx_gallery.gen_rst.split_code_and_text_blocks(abs_fname)
                intro, _ = sphinx_gallery.gen_rst.extract_intro_and_title(abs_fname, blocks[0][1])

            thumbnail_rst = sphinx_gallery.backreferences._thumbnail_div(
                dirname, basename, intro)

            if 'figure' in self.options:
                rel_figname, figname = env.relfn2path(self.options['figure'])
                save_figname = os.path.join('_static/thumbs/',
                                            os.path.basename(figname))

                try:
                    os.makedirs('_static/thumbs')
                except OSError:
                    pass

                sphinx_gallery.gen_rst.scale_image(figname, save_figname,
                                                   400, 280)
                # replace figure in rst with simple regex
                thumbnail_rst = re.sub(r'..\sfigure::\s.*\.png',
                                       '.. figure:: /{}'.format(save_figname),
                                       thumbnail_rst)

            thumbnail = StringList(thumbnail_rst.split('\n'))
            thumb = nodes.paragraph()
            self.state.nested_parse(thumbnail, self.content_offset, thumb)

            return [thumb]
        except FileNotFoundError as e:
            print(e)
            return []


GALLERY_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="{tooltip}">

.. only:: html

    .. figure:: {thumbnail}

        {description}

.. raw:: html

    </div>
"""


class CustomGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.

    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.

    Example usage:

    .. customgalleryitem::
        :tooltip: I am writing this tutorial to focus specifically on NLP for people who have never written code in any deep learning framework
        :figure: /_static/img/thumbnails/babel.jpg
        :description: :doc:`/beginner/deep_learning_nlp_tutorial`

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'tooltip': directives.unchanged,
                   'figure': directives.unchanged,
                   'description': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'tooltip' in self.options:
                tooltip = self.options['tooltip'][:195] + '...'
            else:
                raise ValueError('tooltip not found')

            if 'figure' in self.options:
                env = self.state.document.settings.env
                rel_figname, figname = env.relfn2path(self.options['figure'])
                thumbnail = os.path.join('_static/thumbs/', os.path.basename(figname))

                try:
                    os.makedirs('_static/thumbs')
                except FileExistsError:
                    pass

                sphinx_gallery.gen_rst.scale_image(figname, thumbnail, 400, 280)
            else:
                thumbnail = '_static/img/thumbnails/default.png'

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not doc found')

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(tooltip=tooltip,
                                                thumbnail=thumbnail,
                                                description=description)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


class CustomCardItemDirective(Directive):
    option_spec = {'header': directives.unchanged,
                   'image': directives.unchanged,
                   'link': directives.unchanged,
                   'card_description': directives.unchanged,
                   'tags': directives.unchanged}

    def run(self):
        try:
            if 'header' in self.options:
                header = self.options['header']
            else:
                raise ValueError('header not doc found')

            if 'image' in self.options:
                image = "<img src='" + self.options['image'] + "'>"
            else:
                image = '_static/img/thumbnails/default.png'

            if 'link' in self.options:
                link = self.options['link']
            else:
                link = ''

            if 'card_description' in self.options:
                card_description = self.options['card_description']
            else:
                card_description = ''

            if 'tags' in self.options:
                tags = self.options['tags']
            else:
                tags = ''

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        card_rst = CARD_TEMPLATE.format(header=header,
                                        image=image,
                                        link=link,
                                        card_description=card_description,
                                        tags=tags)
        card_list = StringList(card_rst.split('\n'))
        card = nodes.paragraph()
        self.state.nested_parse(card_list, self.content_offset, card)
        return [card]


CARD_TEMPLATE = """
.. raw:: html

    <div class="col-md-12 tutorials-card-container" data-tags={tags}>

    <div class="card tutorials-card" link={link}>

    <div class="card-body">

    <div class="card-title-container">
        <h4>{header}</h4>
    </div>

    <p class="card-summary">{card_description}</p>

    <p class="tags">{tags}</p>

    <div class="tutorials-image">{image}</div>

    </div>

    </div>

    </div>
"""

class CustomCalloutItemDirective(Directive):
    option_spec = {'header': directives.unchanged,
                   'description': directives.unchanged,
                   'button_link': directives.unchanged,
                   'button_text': directives.unchanged}

    def run(self):
        try:
            if 'description' in self.options:
                description = self.options['description']
            else:
                description = ''

            if 'header' in self.options:
                header = self.options['header']
            else:
                raise ValueError('header not doc found')

            if 'button_link' in self.options:
                button_link = self.options['button_link']
            else:
                button_link = ''

            if 'button_text' in self.options:
                button_text = self.options['button_text']
            else:
                button_text = ''

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        callout_rst = CALLOUT_TEMPLATE.format(description=description,
                                              header=header,
                                              button_link=button_link,
                                              button_text=button_text)
        callout_list = StringList(callout_rst.split('\n'))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]

CALLOUT_TEMPLATE = """
.. raw:: html

    <div class="col-md-6">
        <div class="text-container">
            <h3>{header}</h3>
            <p class="body-paragraph">{description}</p>
            <a class="btn with-right-arrow callout-button" href="{button_link}">{button_text}</a>
        </div>
    </div>
"""
