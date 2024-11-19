import openai
import os
import time
import sys

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from the environment variable.
    """
    api_key = "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"
    if not api_key:
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return api_key

def read_in_chunks(file_object, chunk_size=1000):
    """
    Generator to read a file in chunks of specified size.
    """
    chunk = []
    for line in file_object:
        tag = line.strip()
        if tag:  # Skip empty lines
            chunk.append(tag)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def create_prompt(tags, categories):
    """
    Creates a prompt for GPT-4 to filter tags based on categories.
    """
    categories_formatted = ', '.join([f'"{cat}"' for cat in categories])
    prompt = f"""
I have a list of tags. Please remove any tag that does not belong to one of the following categories: {categories_formatted}.

Return only the valid tags, one per line, exactly as they appear in the input. Do not include any additional text or explanations.

Here are the tags:

{chr(10).join(tags)}
"""
    return prompt

def filter_tags_with_gpt(api_key, tags, categories, model="gpt-4o", max_retries=5, backoff_factor=2):
    """
    Sends a prompt to GPT-4 to filter tags based on categories.
    Implements retry logic with exponential backoff.
    """
    openai.api_key = api_key
    prompt = create_prompt(tags, categories)

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that filters tags based on specified categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic output
                max_tokens=2048  # Adjust as needed
            )
            # Extract the content from the response
            filtered_tags = response.choices[0].message.content.strip().split('\n')
            # Remove any empty strings resulting from split
            filtered_tags = [tag.strip() for tag in filtered_tags if tag.strip()]
            return filtered_tags
        except AttributeError as ae:
            print(f"AttributeError: {ae}. Ensure that the OpenAI package is correctly installed and up-to-date.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Skipping this batch.")
            return []
    print("Max retries exceeded. Skipping this batch.")
    return []

def main(input_file_path, output_file_path):
    """
    Main function to process the input file and filter tags.
    """
    # Define the categories
    categories = [
        "body characteristics",
        "facial characteristics",
        "background scenery",
        "clothing",
        "body poses",
        "related to human anatomy"
    ]

    api_key = get_openai_api_key()

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'a', encoding='utf-8') as outfile:

            total_tags = 0
            total_filtered = 0
            batch_number = 1

            for chunk in read_in_chunks(infile, chunk_size=1000):
                print(f"Processing batch {batch_number} with {len(chunk)} tags...")
                filtered_tags = filter_tags_with_gpt(api_key, chunk, categories)

                # Write the filtered tags to the output file
                for tag in filtered_tags:
                    outfile.write(f"{tag}\n")

                batch_filtered_count = len(filtered_tags)
                total_filtered += batch_filtered_count
                total_tags += len(chunk)

                print(f"Batch {batch_number} processed. {batch_filtered_count} tags added.\n")
                batch_number += 1

            print(f"Processing complete. Total tags processed: {total_tags}. Total tags filtered: {total_filtered}.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter tags using GPT-4 based on specified categories.")
    parser.add_argument('input_file', help='Path to the input text file containing tags (one per line).')
    parser.add_argument('output_file', help='Path to the output text file to save filtered tags.')

    args = parser.parse_args()

    main(args.input_file, args.output_file)
