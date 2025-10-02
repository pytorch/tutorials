// This code replaces the default sphinx gallery download buttons
// with the 3 download buttons at the top of the page

document.addEventListener('DOMContentLoaded', function() {
  var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
  if (downloadNote.length >= 1) {
      var tutorialUrlArray = $("#tutorial-type").text().split('/');
      tutorialUrlArray[0] = tutorialUrlArray[0] + "_source"

      // Get configurable repository settings from conf.py

      // Default to PyTorch tutorials for backward compatibility
      var defaultGithubRepo = "pytorch/tutorials";
      var defaultGithubBranch = "main";
      var defaultColabRepo = "pytorch/tutorials";
      var defaultColabBranch = "gh-pages";

      // Use configured values from window.repoConfig or fallback to defaults
      // This ensures backward compatibility when tutorial_repo_config is not defined
      var githubRepo = (window.repoConfig && window.repoConfig.github_repo) || defaultGithubRepo;
      var githubBranch = (window.repoConfig && window.repoConfig.github_branch) || defaultGithubBranch;
      var colabRepo = (window.repoConfig && window.repoConfig.colab_repo) || defaultColabRepo;
      var colabBranch = (window.repoConfig && window.repoConfig.colab_branch) || defaultColabBranch;

      var githubLink = "https://github.com/" + githubRepo + "/blob/" + githubBranch + "/" + tutorialUrlArray.join("/") + ".py";

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
          colabLink = "https://colab.research.google.com/github/" + colabRepo + "/blob/" + colabBranch + "/_downloads" + notebookDownloadPath;

      $("#colab-link").attr("href", colabLink);
      $("#notebook-link").attr("href", notebookLink);
      $("#github-link").attr("href", githubLink);

      // Hide the original download links and signature
      $(".sphx-glr-footer").hide();
      $(".sphx-glr-signature").hide();
      $(".sphx-glr-footer, .sphx-glr-download").hide();
      $(".sphx-glr-signature").hide();

  } else {
      $(".pytorch-call-to-action-links").hide();
  }
});
