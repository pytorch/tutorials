# PyTorch Tutorial Submission Policy

This policy outlines the criteria and process for submitting new
tutorials to the PyTorch community.
Our goal is to ensure that all tutorials are of high quality,
relevant, and up-to-date, supporting both the growth of the PyTorch
users and the evolution of the PyTorch framework itself. By following
these guidelines, contributors can help us maintain a robust and
informative educational environment.

## Acceptance Criteria For New Tutorials

We accept new tutorials that adhere to one of the following use cases:

* **Support for New PyTorch Features:** Tutorials that support new features
  for upcoming PyTorch releases are typically authored by the engineers who
  are developing these features. These tutorials are crucial for showcasing
  the latest advancements in PyTorch. We typically don't require more than
  one tutorial per feature.

* **Tutorials showcasing PyTorch usage with other tools and libraries:** We
  accept community-contributed tutorials that illustrate innovative uses of
  PyTorch alongside other open-source projects, models, and tools. Please
  ensure that your tutorial remains neutral and does not promote or endorse
  proprietary technologies over others.

The first use case does not require going through the submission
process outlined below. If your tutorial falls under the second category,
please read and follow the instructions in the
**Submission Process For Community-Contributed Tutorials** section.

## Submission Process For Community-Contributed Tutorials

To maintain the quality and relevance of tutorials, we request that
community-contributed tutorials undergo a review process. If you are
interested in contributing a tutorial, please follow these steps:

1. **Create an issue:**
   * Open an issue in the pytorch/tutorials repository proposing the
     new tutorial. Clearly explain the importance of the tutorial and
     confirm that there is no existing tutorial covering the same or
     similar topic. A tutorial should not disproportionately endorse
     one technology over another. Please consult with Core Maintainers
     to ensure your content adheres to these guidelines.
     Use the provided `ISSUE_TEMPLATE` for the new tutorial request.

     * If there is a tutorial on the existing topic that you would like
       to significantly refactor, you can just submit a PR. In the
       description of the PR explain why the changes are needed and
       how they improve the tutorial.

   * These issues will be triaged by PyTorch maintainers on a case-by-case basis. 
   * Link any supporting materials including discussions in other repositories.
     
1. **Await Approval:**
   * Wait for a response from the PyTorch maintainers. A PyTorch
     tutorial maintainer will review your proposal and
     determine whether a tutorial on the proposed topic is desirable.
     A comment and an **approved** label will be added to your tutorial
     by a maintainer. Review process for new tutorial PRs submitted
     without the corresponding issue will take longer.
     
1. **Adhere to writing and styling guidelines:**
   * Once approved, follow the guidelines outlined in [CONTRIBUTING.md](https://github.com/pytorch/tutorials/blob/main/CONTRIBUTING.md)
     and use the provided [template](https://github.com/pytorch/tutorials/blob/main/beginner_source/template_tutorial.py) for creating your tutorial.
   * Link the issue in which you received an approval for your tutorial
     in the PR.
   * We accept tutorials in both ``.rst`` (ReStructuredText) and ``.py``
     (Python) formats. However, unless your tutorial involves using
     multiple GPU, parallel/distributed training, or requires extended
     execution time (25 minutes or more), we prefer submissions
     in Python file format.
     
## Maintening Tutorials

When you submit a new tutorial, we encourage you to keep it updated
with the latest PyTorch updates and features. Additionally, we may
contact you to review any PRs, issues, and other related matters to
ensure the tutorial remains a valuable resource.

Please note the following: 

* If a tutorial is broken against the main branch, the tutorial will
  be excluded from the build and an issue will be filed against it
  with the author/maintainer notified about the issue. If the issue
  is not resolved within 90 days, the tutorial might be deleted from
  the repository.

* We recommend that each tutorial is reviewed at least once a year to
  ensure its relevance.

## Deleting Stale Tutorials

A tutorial might be considered stale when it has not been updated for
a significant period (more than a year) and no longer aligns with the
latest PyTorch updates, features, or best practices. Other indicators
of a stale tutorial include:

* The tutorial is no longer functional due to changes in PyTorch or
  its dependencies
* The tutorial has been superseded by a newer, more comprehensive, or
  more accurate tutorial
* The tutorial does not run successfully in the (CI), indicating
  potential compatibility or dependency issues.

If a tutorial is deemed stale, we will attempt to contact the code owner
or someone from the tutorial mainatainers might attempt to update it.
However, if despite those attempts, we fail to fix it, the tutorial
might be removed from the repository.
