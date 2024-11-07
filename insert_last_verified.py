import json
import os
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime

# Path to the single JSON file
json_file_path = "output.json"

# Base directory for the generated HTML files
build_dir = "_build/html"

# Define the source to build path mapping
source_to_build_mapping = {
    "beginner": "beginner_source",
    "recipes": "recipes_source",
    "distributed": "distributed",
    "intermediate": "intermediate_source",
    "prototype": "prototype_source",
    "advanced": "advanced_source"
}

# Function to get the creation date of a file using git log
def get_creation_date(file_path):
    try:
        # Run git log to get the date of the first commit for the file
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%aD", "--", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        # Check if the output is not empty
        if result.stdout:
            creation_date = result.stdout.splitlines()[0]
            # Parse and format the date
            creation_date = datetime.strptime(creation_date, "%a, %d %b %Y %H:%M:%S %z")
            formatted_date = creation_date.strftime("%d %b, %Y")
        else:
            formatted_date = "Unknown"
        return formatted_date
    except subprocess.CalledProcessError:
        return "Unknown"

# Function to find the source file with any common extension
def find_source_file(base_path):
    for ext in ['.rst', '.py']:
        source_file_path = base_path + ext
        if os.path.exists(source_file_path):
            return source_file_path
    return None

# Function to process the JSON file
def process_json_file(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    # Process each entry in the JSON data
    for entry in json_data:
        path = entry["Path"]
        last_verified = entry["Last Verified"]

        # Format the "Last Verified" date
        try:
            last_verified_date = datetime.strptime(last_verified, "%Y-%m-%d")
            formatted_last_verified = last_verified_date.strftime("%d %b, %Y")
        except ValueError:
            formatted_last_verified = "Unknown"

        # Determine the source directory and file name
        for build_subdir, source_subdir in source_to_build_mapping.items():
            if path.startswith(build_subdir):
                # Construct the path to the HTML file
                html_file_path = os.path.join(build_dir, path + ".html")
                # Construct the base path to the source file
                base_source_path = os.path.join(source_subdir, path[len(build_subdir)+1:])
                # Find the actual source file
                source_file_path = find_source_file(base_source_path)
                break
        else:
            print(f"Warning: No mapping found for path {path}")
            continue

        # Check if the HTML file exists
        if not os.path.exists(html_file_path):
            print(f"Warning: HTML file not found for path {html_file_path}")
            continue

        # Check if the source file was found
        if not source_file_path:
            print(f"Warning: Source file not found for path {base_source_path}")
            continue

        # Get the creation date of the source file
        created_on = get_creation_date(source_file_path)

        # Open and parse the HTML file
        with open(html_file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Find the first <h1> tag and insert the "Last Verified" and "Created On" dates after it
        h1_tag = soup.find("h1")
        if h1_tag:
            # Create a new tag for the dates
            date_info_tag = soup.new_tag("p")
            date_info_tag['style'] = "color: #6c6c6d; font-size: small;"

            # Add the "Created On" and "Last Verified" information
            date_info_tag.string = f"Created On: {created_on} | Last Verified: {formatted_last_verified}"

            # Insert the new tag after the <h1> tag
            h1_tag.insert_after(date_info_tag)

            # Save the modified HTML back to the file
            with open(html_file_path, "w", encoding="utf-8") as file:
                file.write(str(soup))
        else:
            print(f"Warning: <h1> tag not found in {html_file_path}")

# Process the single JSON file
process_json_file(json_file_path)

print("Processing complete.")
