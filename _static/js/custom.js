document.addEventListener("DOMContentLoaded", function() {
  // Select all <li> elements with the class "toctree-l1"
  var toctreeItems = document.querySelectorAll('li.toctree-l1');

  toctreeItems.forEach(function(item) {
    // Find the link within the item
    var link = item.querySelector('a');
    var nestedList = item.querySelector('ul');

    if (link && nestedList) {
      // Create a span element for the "[+]" or "[-]" sign
      var expandSign = document.createElement('span');
      expandSign.style.cursor = 'pointer'; // Make it look clickable

      // Use the link text as a unique key for localStorage
      var sectionKey = 'section_' + link.textContent.trim().replace(/\s+/g, '_');

      // Retrieve the saved state from localStorage
      var isExpanded = localStorage.getItem(sectionKey);

      // If no state is saved, default to expanded for "Learn the Basics" and collapsed for others
      if (isExpanded === null) {
        isExpanded = (link.textContent.trim() === 'Learn the Basics') ? 'true' : 'false';
        localStorage.setItem(sectionKey, isExpanded);
      }

      if (isExpanded === 'true') {
        nestedList.style.display = 'block'; // Expand the section
        expandSign.textContent = '[-] '; // Show "[-]" since it's expanded
      } else {
        nestedList.style.display = 'none'; // Collapse the section
        expandSign.textContent = '[+] '; // Show "[+]" since it's collapsed
      }

      // Add a click event to toggle the nested list
      expandSign.addEventListener('click', function() {
        if (nestedList.style.display === 'none') {
          nestedList.style.display = 'block';
          expandSign.textContent = '[-] '; // Change to "[-]" when expanded
          localStorage.setItem(sectionKey, 'true'); // Save state
        } else {
          nestedList.style.display = 'none';
          expandSign.textContent = '[+] '; // Change back to "[+]" when collapsed
          localStorage.setItem(sectionKey, 'false'); // Save state
        }
      });

      // Insert the sign before the link
      link.parentNode.insertBefore(expandSign, link);
    }
  });
});
