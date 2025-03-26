import json
import os
import subprocess
import sys
from datetime import datetime

from bs4 import BeautifulSoup

json_file_path = "tutorials-review-data.json"

# paths to skip from the post-processing script
paths_to_skip = [
    "beginner/examples_autograd/two_layer_net_custom_function",  # not present in the repo
    "beginner/examples_nn/two_layer_net_module",  # not present in the repo
    "beginner/examples_tensor/two_layer_net_numpy",  # not present in the repo
    "beginner/examples_tensor/two_layer_net_tensor",  # not present in the repo
    "beginner/examples_autograd/two_layer_net_autograd",  # not present in the repo
    "beginner/examples_nn/two_layer_net_optim",  # not present in the repo
    "beginner/examples_nn/two_layer_net_nn",  # not present in the repo
    "intermediate/coding_ddpg",  # not present in the repo - will delete the carryover
]
# Mapping of source directories to build directories
source_to_build_mapping = {
    "beginner": "beginner_source",
    "recipes": "recipes_source",
    "distributed": "distributed",
    "intermediate": "intermediate_source",
    "prototype": "prototype_source",
    "advanced": "advanced_source",
    "": "",  # root dir for index.rst
}


def get_git_log_date(file_path, git_log_args):
    try:
        result = subprocess.run(
            ["git", "log"] + git_log_args + ["--", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            date_str = result.stdout.splitlines()[0]
            return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
    except subprocess.CalledProcessError:
        pass
    raise ValueError(f"Could not find date for {file_path}")


def get_creation_date(file_path):
    return get_git_log_date(file_path, ["--diff-filter=A", "--format=%aD"]).strftime(
        "%b %d, %Y"
    )


def get_last_updated_date(file_path):
    return get_git_log_date(file_path, ["-1", "--format=%aD"]).strftime("%b %d, %Y")


# Try to find the source file with the given base path and the extensions .rst and .py
def find_source_file(base_path):
    for ext in [".rst", ".py"]:
        source_file_path = base_path + ext
        if os.path.exists(source_file_path):
            return source_file_path
    return None


# Function to process a JSON file and insert the "Last Verified" information into the HTML files
def process_json_file(build_dir, json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    for entry in json_data:
        path = entry["Path"]
        last_verified = entry["Last Verified"]
        status = entry.get("Status", "")
        if path in paths_to_skip:
            print(f"Skipping path: {path}")
            continue
        if status in ["needs update", "not verified"]:
            formatted_last_verified = "Not Verified"
        elif last_verified:
            try:
                last_verified_date = datetime.strptime(last_verified, "%Y-%m-%d")
                formatted_last_verified = last_verified_date.strftime("%b %d, %Y")
            except ValueError:
                formatted_last_verified = "Unknown"
        else:
            formatted_last_verified = "Not Verified"
        if status == "deprecated":
            formatted_last_verified += "Deprecated"

        for build_subdir, source_subdir in source_to_build_mapping.items():
            if path.startswith(build_subdir):
                html_file_path = os.path.join(build_dir, path + ".html")
                base_source_path = os.path.join(
                    source_subdir, path[len(build_subdir) + 1 :]
                )
                source_file_path = find_source_file(base_source_path)
                break
        else:
            print(f"Warning: No mapping found for path {path}")
            continue

        if not os.path.exists(html_file_path):
            print(
                f"Warning: HTML file not found for path {html_file_path}."
                "If this is a new tutorial, please add it to the audit JSON file and set the Verified status and todays's date."
            )
            continue

        if not source_file_path:
            print(f"Warning: Source file not found for path {base_source_path}.")
            continue

        created_on = get_creation_date(source_file_path)
        last_updated = get_last_updated_date(source_file_path)

        with open(html_file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        # Check if the <p> tag with class "date-info-last-verified" already exists
        existing_date_info = soup.find("p", {"class": "date-info-last-verified"})
        if existing_date_info:
            print(
                f"Warning: <p> tag with class 'date-info-last-verified' already exists in {html_file_path}"
            )
            continue

        h1_tag = soup.find("h1")  # Find the h1 tag to insert the dates
        if h1_tag:
            date_info_tag = soup.new_tag("p", **{"class": "date-info-last-verified"})
            date_info_tag["style"] = "color: #6c6c6d; font-size: small;"
            # Add the "Created On", "Last Updated", and "Last Verified" information
            date_info_tag.string = (
                f"Created On: {created_on} | "
                f"Last Updated: {last_updated} | "
                f"Last Verified: {formatted_last_verified}"
            )
            # Insert the new tag after the <h1> tag
            h1_tag.insert_after(date_info_tag)
            # Save back to the HTML.
            with open(html_file_path, "w", encoding="utf-8") as file:
                file.write(str(soup))
        else:
            print(f"Warning: <h1> tag not found in {html_file_path}")


def main():
    if len(sys.argv) < 2:
        print("Error: Build directory not provided. Exiting.")
        exit(1)
    build_dir = sys.argv[1]
    print(f"Build directory: {build_dir}")
    process_json_file(build_dir, json_file_path)
    print(
        "Finished processing JSON file. Please check the output for any warnings. "
        "Pages like `nlp/index.html` are generated only during the full `make docs` "
        "or `make html` build. Warnings about these files when you run `make html-noplot` "
        "can be ignored."
    )


if __name__ == "__main__":
    main()
