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

      // Check if this is the "Learn the Basics" section
      if (link.textContent.trim() === 'Learn the Basics') {
        nestedList.style.display = 'block'; // Expand the "Learn the Basics" section
        expandSign.textContent = '[-] '; // Show "[-]" since it's expanded
      } else {
        nestedList.style.display = 'none'; // Collapse other sections
        expandSign.textContent = '[+] '; // Show "[+]" since it's collapsed
      }

      // Insert the sign before the link
      link.parentNode.insertBefore(expandSign, link);

      // Add a click event to toggle the nested list
      expandSign.addEventListener('click', function() {
        if (nestedList.style.display === 'none') {
          nestedList.style.display = 'block';
          expandSign.textContent = '[-] '; // Change to "[-]" when expanded
        } else {
          nestedList.style.display = 'none';
          expandSign.textContent = '[+] '; // Change back to "[+]" when collapsed
        }
      });
    }
  });
});
