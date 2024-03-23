import json

# Load the original JSON data
with open('skills.json', 'r') as file:
    data = json.load(file)

# Prepare the topics dictionary for easy lookup
topics_dict = {topic["id"]: topic["name"] for topic in data["topics"]}

# Function to replace group abbreviations with their full names
def replace_group_name(group):
    groups = {
        "RM": "Required Module",
        "GCEM": "General Core Elective Module",
        "CEM DE": "Core Elective Module Domain Experience",
        "CEM AAE": "Core Elective Module Advanced Topics"
    }
    return groups.get(group, group)

# Create a new list to hold the transformed modules
transformed_modules = []

for module in data["modules"]:
    transformed_module = {
        "code": module["code"],
        "name": module["name"],
        "ects": module["ects"],
        "semester": module["semester"],
        "group": replace_group_name(module["group"]),
        "reqs": module["reqs"],
        "topics": [topics_dict.get(topic, topic) for topic in module["topics"]]
    }
    transformed_modules.append(transformed_module)

# Create a new dictionary to hold the transformed data
transformed_data = {"modules": transformed_modules}

# Save the transformed data to a new JSON file
with open('module_requirements_table_reduced.json', 'w') as outfile:
    json.dump(transformed_data, outfile, indent=4)
