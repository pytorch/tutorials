import os
from github import Github
import sys
import re

token = os.environ.get('GITHUB_TOKEN')

repo_owner = "pytorch"
repo_name = "tutorials"
pull_request_number = int(sys.argv[1])

g = Github(token)
repo = g.get_repo(f'{repo_owner}/{repo_name}')
pull_request = repo.get_pull(pull_request_number)
pull_request_body = pull_request.body

# get issue number from the PR body
if re.search(r'#\d{1,5}', pull_request_body):
    issue_number = int(re.findall(r'#(\d{1,5})', pull_request_body)[0])
    issue = repo.get_issue(issue_number)
    issue_labels = issue.labels
    docathon_label_present = any(label.name == 'docathon-h1-2023' for label in issue_labels)

    # if the issue has a docathon label, add all labels from the issue to the PR.
    if docathon_label_present:
        pull_request_labels = pull_request.get_labels()
        issue_label_names = [label.name for label in issue_labels]
        labels_to_add = [label for label in issue_label_names if label not in pull_request_labels]
        if labels_to_add:
            pull_request.set_labels(*labels_to_add)
            print("Labels added to the pull request!")
        else:
            print("The pull request already has the same labels.")
    else:
        print("The 'docathon-h1-2023' label is not present in the issue.")
else:
    print("The pull request does not mention an issue.")
