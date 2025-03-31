import os

import sphinx.addnodes

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective


logger = logging.getLogger(__name__)


class AddCardsDirective(SphinxDirective):
    """
    Directive to automatically add cards based on toctree entries.
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        env = self.env
        toctrees = env.tocs.get(env.docname, None)

        if not toctrees:
            logger.warning(f"No toctrees found in document {env.docname}")
            return []

        # Find all toctrees in the document
        all_cards_container = nodes.container()
        all_cards_container["classes"] = ["all-tutorial-cards"]

        # Process each toctree
        for toctreenode in toctrees.traverse(addnodes.toctree):
            # Get caption
            caption = toctreenode.get("caption", "")

            # Create section container
            section_container = nodes.container()
            section_container["classes"] = ["tutorial-section"]

            # Add section title if caption exists
            if caption:
                title_node = nodes.paragraph()
                title_node["classes"] = ["tutorial-section-title"]
                title_node += nodes.Text(caption)
                section_container += title_node

            # Create cards container
            cards_container = nodes.container()
            cards_container["classes"] = ["tutorial-cards-container"]

            # Find all entries in this toctree
            for entry in toctreenode["entries"]:
                doc_name = entry[1]
                title = env.titles.get(doc_name, nodes.title()).astext() or doc_name

                # Try to get description from the document
                description = ""
                doc_path = os.path.join(self.env.srcdir, doc_name + ".rst")
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, "r") as f:
                            content = f.read()
                            # Extract first paragraph after title as description
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if i > 2 and line.strip() and not line.startswith(".."):
                                    description = line.strip()
                                    break
                    except Exception as e:
                        logger.warning(f"Error reading {doc_path}: {e}")

                # Create card
                card = nodes.container()
                card["classes"] = ["tutorial-card"]

                # Add link
                card_link = nodes.reference("", "")
                card_link["refuri"] = entry["refuri"]
                card_link["classes"] = ["card-link"]

                # Add title
                title_node = nodes.paragraph()
                title_node["classes"] = ["card-title"]
                title_node += nodes.Text(title)
                card_link += title_node

                # Add description if available
                if description:
                    desc_node = nodes.paragraph()
                    desc_node["classes"] = ["card-description"]
                    desc_node += nodes.Text(
                        description[:100] + "..."
                        if len(description) > 100
                        else description
                    )
                    card_link += desc_node

                card += card_link
                cards_container += card

            section_container += cards_container
            all_cards_container += section_container

        return [all_cards_container]


def setup(app):
    app.add_directive("add-cards", AddCardsDirective)
    app.add_css_file("tutorial_cards.css")

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
