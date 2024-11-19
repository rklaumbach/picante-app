import json

def extract_tags(input_file, output_file):
    data = []
    try:
        with open(input_file, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                # Remove any leading/trailing whitespace
                line = line.strip()
                
                # Stop processing if line contains 'END'
                if 'END' in line:
                    print(f"Encountered 'END' at line {line_number}. Stopping processing.")
                    break
                
                # Skip empty lines
                if not line:
                    print(f"Skipping empty line at line {line_number}.")
                    continue
                
                # Split the line at the first comma
                if ',' in line:
                    tag, post_count = line.split(',', 1)
                    tag = tag.strip()
                    post_count = post_count.strip()
                    
                    # Add the dictionary to the list
                    data.append({
                        "tag": tag,
                        "post_count": post_count
                    })
                else:
                    print(f"Skipping line {line_number} (no comma found): {line}")
        
        # Write the data to a JSON file
        with open(output_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f"\nData successfully written to '{output_file}'")
        print(f"Total tags extracted: {len(data)}")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Specify the input text file and the output JSON file
    input_filename = 'danbooru_tags_post_count.txt'      # Replace with your input .txt file path
    output_filename = 'output.json'   # Replace with your desired output file path
    
    extract_tags(input_filename, output_filename)
