// Override "main" with version variable
document.addEventListener('DOMContentLoaded', function() {
  // Check if this is the "docs" pytorch_project
  const metaElement = document.querySelector('meta[name="pytorch_project"]');
  console.log("PyTorch Project:", metaElement);
  if (!metaElement || metaElement.getAttribute('content') !== 'docs') {
    return; // Exit early if not the pytorch docs project
  }
  const version = document.documentElement.getAttribute('data-version');

  // Function to check and update buttons and dropdown items
  function updateElements() {
    // Update buttons
    const buttons = document.querySelectorAll('.version-switcher__button');
    let buttonFound = false;

    buttons.forEach(btn => {
      console.log("Found button:", btn.innerText);
      if (btn.innerText.includes('main')) {
        btn.innerText = version;
        if (btn.hasAttribute('data-active-version-name')) {
          btn.setAttribute('data-active-version-name', version);
        }
        buttonFound = true;
      }
    });

    // Update dropdown items
    const dropdownItems = document.querySelectorAll('.dropdown-item.list-group-item');
    let dropdownFound = false;

    dropdownItems.forEach(item => {
      if (item.getAttribute('data-version') === 'main') {
        // Update span text only
        const span = item.querySelector('span');
        if (span && span.innerText.includes('main')) {
          span.innerText = version;
          dropdownFound = true;
        }
      }
    });

    // If not found, try again after a delay
    if ((!buttonFound || !dropdownFound) && attempts < 10) {
      attempts++;
      setTimeout(updateElements, 500);
    }
  }

  let attempts = 0;
  updateElements();
});
