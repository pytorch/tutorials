# PyTorch Tutorial Submission Policy

This policy outlines the criteria and process for submitting new tutorials to the PyTorch community.
Our goal is to ensure that all tutorials are of high quality, relevant, and up-to-date, supporting
both the growth of our users and the evolution of the PyTorch framework itself. By following these
guidelines, contributors can help us maintain a robust and informative educational environment.

## Acceptance Criteria For New Tutorials
We are committed to enhancing the learning experience for PyTorch users by providing high-quality
tutorials. We accept new tutorials that adhere to one of the following use cases:

* **Support for New PyTorch Features:** Tutorials that support new features for upcoming PyTorch
releases are typically authored by the engineers who are developing these features. These tutorials
are crucial for showcasing the latest advancements in PyTorch.

* **Tutorials from PyTorch Partners:** We welcome tutorials from our partners who are actively
  collaborating with us to grow the PyTorch ecosystem.
  
* **Direct Requests from PyTorch Maintainers:** If a PyTorch maintainer has directly reached out
  to you to request a tutorial or otherwise, endorsed a tutorial, this indicates a specific need
  within the community or the project.
  
* **Tutorials from the open-source community members:** We accept tutorials from the community
  members. However, as we can't accept all of them, we have identified the process below.
  Please read below before sending a PR.

The first three use cases do not require following the submission process described below.
If your tutorial falls under the fourth category, please read and follow the instructions in
the **Submission Process For Community-Contributed Tutorials** section.

## Submission Process For Community-Contributed Tutorials

To maintain the quality and relevance of tutorials, community-contributed tutorials must undergo
a review process. If you are interested in contributing a tutorial, please follow these steps:

1. **Create an issue:**
   * Open an issue in the pytorch/tutorials repository proposing the new tutorial. Clearly
     explain the importance of the tutorial and confirm that there is no existing tutorial
     covering the same or similar topic. A tutorial can't endorse one technology over another.
   * These issues will be triaged by PyTorch maintainers on a case-by-case basis. 
   * Link any supporting materials including discussions in other repositories".
     
1. **Await Approval:**
   * Wait for a response from the PyTorch maintainers. They will review your proposal and
     determine whether a tutorial on the proposed topic is desirable. **Do not submit a PR with
     your tutorial** before you receive an approval.
     
1. **Adhere to writing and styling guidelines:**
   * Once approved, follow the guidelines outlined in [CONTRIBUTING.md](https://github.com/pytorch/tutorials/blob/main/CONTRIBUTING.md)
     and use the provided [template](https://github.com/pytorch/tutorials/blob/main/beginner_source/template_tutorial.py) for creating your tutorial.
   * Link the issue where you received an approval for your tutorial in the PR.
   * We accept tutorials in both ``.rst`` (ReStructuredText) and ``.py`` (Python)
     formats. However, unless your tutorial involves using multiple GPU, parallel/distributed
     training, or requires extended execution time (25 minutes or more), we prefer submissions
     in Python file format versus ``.rst``.
     
1. **Maintenance:**
   * By submitting a new tutorial, you agree to maintain it up-to-date with the latest PyTorch
     updates and new features, review issues and pull requests and bugs submitted against the
     tutorials and overall take proactive actions to ensure quality of the tutorial.
   * Each tutorial must be updated by its owners at least once a year. Tutorials that are
     abandoned by the owners will be eventually deprecated.

We reserve the right to decline any new tutorial submissions that do not follow this policy.
