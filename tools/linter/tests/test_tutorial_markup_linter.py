import textwrap
import unittest

from tools.linter.adapters.tutorial_markup_linter import lint_source


def lint(prose: str):
    source = (
        '"""\n'
        "Example tutorial\n"
        "================\n\n"
        f"{textwrap.dedent(prose)}\n"
        '"""\n'
    )
    return lint_source("example_tutorial.py", source)


class TutorialMarkupLinterTest(unittest.TestCase):
    def test_reports_indented_list_without_blank_line(self):
        messages = lint(
            """\
            Parameters:
              - first value
              - second value"""
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "list missing a preceding blank line")

    def test_allows_indented_list_after_blank_line(self):
        messages = lint(
            """\
            Parameters:

              - first value
              - second value"""
        )

        self.assertEqual(messages, [])

    def test_allows_conventionally_indented_nested_list(self):
        messages = lint(
            """\
            - parent item
              - nested item
              - another nested item"""
        )

        self.assertEqual(messages, [])

    def test_allows_list_table_cells(self):
        messages = lint(
            """\
            .. list-table::

               * - Heading
                 - Value"""
        )

        self.assertEqual(messages, [])

    def test_allows_list_aligned_inside_directive(self):
        messages = lint(
            """\
            .. note::
                This is directive content.
                * first item
                * second item"""
        )

        self.assertEqual(messages, [])

    def test_reports_unindented_list_continuation(self):
        messages = lint(
            """\
            - first item starts here
            but its continuation is not indented
            - second item"""
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "unindented list continuation")

    def test_allows_indented_list_continuation(self):
        messages = lint(
            """\
            - first item starts here
              and its continuation is indented
            - second item"""
        )

        self.assertEqual(messages, [])

    def test_ignores_comments_outside_gallery_prose(self):
        source = textwrap.dedent(
            '''\
            """Example tutorial"""

            # Implementation details:
            #  - this is a code comment, not narrative prose
            value = 1
            '''
        )

        self.assertEqual(lint_source("example_tutorial.py", source), [])


if __name__ == "__main__":
    unittest.main()
