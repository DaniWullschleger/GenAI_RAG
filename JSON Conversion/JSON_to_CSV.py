import json
import csv


# Function to replace group abbreviations with full names
def replace_group_name(group):
    groups = {
        "RM": "Required Module",
        "GCEM": "General Core Elective Module",
        "CEM DE": "Core Elective Module Domain Experience",
        "CEM AAE": "Core Elective Module Advanced Topics"
    }
    return groups.get(group, group)  # Return the group name, or the original abbreviation if not found


# Load the JSON data
with open('skills.json', 'r') as file:
    data = json.load(file)

# Prepare the topics dictionary for replacement
topics_dict = {topic["id"]: topic["name"] for topic in data["topics"]}

# Open a CSV file to write the output
with open('module_requirements_table.csv', mode='w', newline='', encoding='utf-8') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)
    # Write the header row
    csv_writer.writerow(['Code', 'Name', 'ECTS', 'Semester', 'Group', 'Reqs', 'Topics'])

    # Iterate over the modules and write each row
    for module in data["modules"]:
        # Replace topics numbers with names
        topics_names = [topics_dict.get(topic_id, topic_id) for topic_id in module["topics"]]
        # Join the topics and requirements lists into strings
        topics_str = ", ".join(topics_names)
        reqs_str = ", ".join(module["reqs"])
        # Write the row
        csv_writer.writerow(
            [module['code'], module['name'], module['ects'], module['semester'], replace_group_name(module['group']),
             reqs_str, topics_str])
