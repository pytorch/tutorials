/**
 * Collapsible "Subclassed by" list behavior.
 * Finds paragraphs starting with "Subclassed by" that have more than 5 items,
 * collapses them to a single line, and adds a "See All" / "Hide" toggle button.
 *
 * Breathe/Doxygen renders "Subclassed by" in a <p> tag. The subclass names may
 * be <a> links (when targets exist) or plain text (when they don't). We count
 * both to determine whether to collapse.
 */
document.addEventListener('DOMContentLoaded', function() {
    var paragraphs = document.querySelectorAll('p');

    paragraphs.forEach(function(p) {
        var text = p.textContent.trim();
        if (!text.startsWith('Subclassed by')) return;

        // Count items: use <a> links if present, otherwise count comma-separated entries
        var links = p.querySelectorAll('a');
        var itemCount = links.length;
        if (itemCount <= 5) {
            // Links may not exist (plain text subclass names). Count comma-separated items.
            var afterLabel = text.replace(/^Subclassed by\s*/, '');
            var commaCount = afterLabel.split(',').filter(function(s) { return s.trim().length > 0; }).length;
            if (commaCount > itemCount) {
                itemCount = commaCount;
            }
        }

        if (itemCount > 5) {
            p.classList.add('subclassed-by-list', 'collapsed');

            var toggle = document.createElement('button');
            toggle.className = 'subclassed-by-toggle';
            toggle.textContent = 'See All (' + itemCount + ')';
            toggle.type = 'button';

            toggle.addEventListener('click', function(e) {
                e.preventDefault();

                if (p.classList.contains('collapsed')) {
                    p.classList.remove('collapsed');
                    p.classList.add('expanded');
                    toggle.textContent = 'Hide';
                } else {
                    p.classList.remove('expanded');
                    p.classList.add('collapsed');
                    toggle.textContent = 'See All (' + itemCount + ')';
                }
            });

            p.appendChild(toggle);
        }
    });
});
