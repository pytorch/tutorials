document.addEventListener("DOMContentLoaded", function() {
  // Select all <li> elements with the class "toctree-l1"
  var toctreeItems = document.querySelectorAll('li.toctree-l1');
  toctreeItems.forEach(function(item) {
    // Check if the item has a nested <ul>
    var nestedList = item.querySelector('ul');
    if (nestedList) {
      // Display the nested list by default
      nestedList.style.display = 'block';
    }
  });
});
