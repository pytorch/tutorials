// This code replaces the default sphinx gallery download buttons
// with the 3 download buttons at the top of the page

document.addEventListener('DOMContentLoaded', function() {
  var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
  if (downloadNote.length >= 1) {
      var tutorialUrlArray = $("#tutorial-type").text().split('/');
      tutorialUrlArray[0] = tutorialUrlArray[0] + "_source"

      var githubLink = "https://github.com/pytorch/tutorials/blob/main/" + tutorialUrlArray.join("/") + ".py";

      // Find the notebook download link by checking for .ipynb extension
      var notebookLinks = $(".reference.download");
      var notebookLink = "";
      for (var i = 0; i < notebookLinks.length; i++) {
          if (notebookLinks[i].href.endsWith('.ipynb')) {
              notebookLink = notebookLinks[i].href;
              break;
          }
      }
      // Fallback to first link if no .ipynb found
      if (!notebookLink && notebookLinks.length > 0) {
          notebookLink = notebookLinks[0].href;
      }

      var notebookDownloadPath = notebookLink.split('_downloads')[1],
          colabLink = "https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads" + notebookDownloadPath;

      $("#google-colab-link").wrap("<a href=" + colabLink + " data-behavior='call-to-action-event' data-response='Run in Google Colab' target='_blank'/>");
      $("#download-notebook-link").wrap("<a href=" + notebookLink + " data-behavior='call-to-action-event' data-response='Download Notebook'/>");
      $("#github-view-link").wrap("<a href=" + githubLink + " data-behavior='call-to-action-event' data-response='View on Github' target='_blank'/>");

      // Hide the original download links and signature
      $(".sphx-glr-footer").hide();
      $(".sphx-glr-signature").hide();
      $(".sphx-glr-footer, .sphx-glr-download").hide();
      $(".sphx-glr-signature").hide();

  } else {
      $(".pytorch-call-to-action-links").hide();
  }
});
