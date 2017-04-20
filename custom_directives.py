from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList 
from docutils import nodes
import re
import os
import sphinx_gallery

class IncludeDirective(Directive):

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
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)

        try:
            if 'intro' in self.options:
                intro = self.options['intro'][:195] + '...'
            else:
                intro = sphinx_gallery.gen_rst.extract_intro(fname)

            thumbnail_rst = sphinx_gallery.backreferences._thumbnail_div(dirname, basename, intro)

            if 'figure' in self.options:
                env = self.state.document.settings.env
                rel_figname, figname = env.relfn2path(self.options['figure'])
                save_figname = os.path.join('_static/thumbs/', os.path.basename(figname))

                try:
                    os.makedirs('_static/thumbs')
                except FileExistsError:
                    pass

                sphinx_gallery.gen_rst.scale_image(figname, save_figname, 400, 280)
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
