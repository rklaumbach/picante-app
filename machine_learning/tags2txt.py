import json
import argparse
import sys

def json_to_tags_txt(json_input, txt_output):
    try:
        # Read the JSON data
        with open(json_input, 'r') as json_file:
            data = json.load(json_file)
        
        # Ensure the data is a list
        if not isinstance(data, list):
            print(f"Error: Expected a list of dictionaries in '{json_input}'.")
            sys.exit(1)
        
        # Extract the 'tag' from each dictionary
        tags = []
        for index, item in enumerate(data, start=1):
            if 'tag' in item:
                tag = item['tag']
                tags.append(tag)
            else:
                print(f"Warning: 'tag' key not found in item {index}. Skipping this item.")
        
        # Write the tags to the output text file
        with open(txt_output, 'w') as txt_file:
            for tag in tags:
                txt_file.write(f"{tag}\n")
        
        print(f"Successfully wrote {len(tags)} tags to '{txt_output}'.")
    
    except FileNotFoundError:
        print(f"Error: The file '{json_input}' does not exist.")
    except json.JSONDecodeError as jde:
        print(f"Error: Failed to parse JSON file '{json_input}'. {jde}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Extract tags from a JSON file and write them to a text file.')
    parser.add_argument('json_input', help='Path to the input JSON file.')
    parser.add_argument('txt_output', help='Path to the output text file.')
    
    args = parser.parse_args()
    
    json_to_tags_txt(args.json_input, args.txt_output)
