import json


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

# Open a text file to write the output
with open('module_requirements_table.txt', 'w') as output_file:
    # Write the table header
    output_file.write(f"{'Code':<20}{'Name':<50}{'ECTS':<5}{'Semester':<9}{'Group':<35}{'Reqs':<20}{'Topics':<50}\n")
    output_file.write("-" * 160 + "\n")

    # Iterate over the modules and write each row
    for module in data["modules"]:
        # Replace topics numbers with names
        topics_names = [topics_dict.get(topic_id, topic_id) for topic_id in module["topics"]]
        # Join the topics and requirements lists into strings
        topics_str = ", ".join(topics_names)
        reqs_str = ", ".join(module["reqs"])
        # Write the row
        output_file.write(
            f"{module['code']:<20}{module['name'][:47]:<50}{module['ects']:<5}{module['semester']:<9}{replace_group_name(module['group']):<35}{reqs_str:<20}{topics_str[:47]:<50}\n")
