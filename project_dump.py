import os
from pathlib import Path

def dump_directory_contents(root_dir, output_file, exclude_dirs=None, exclude_files=None, exclude_names=None, verbose=False):
    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_files is None:
        exclude_files = []
    if exclude_names is None:
        exclude_names = []

    root_path = Path(root_dir).resolve()

    # Normalize exclusion paths
    exclude_paths = [ (root_path / Path(d)).resolve() for d in exclude_dirs ]
    exclude_file_paths = [ (root_path / Path(f)).resolve() for f in exclude_files ]

    with open(output_file, 'w', encoding='utf-8') as out:
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=True, followlinks=False):
            current_path = Path(dirpath).resolve()

            # Exclude directories by absolute path
            dirs_to_remove = []
            for dirname in dirnames:
                dir_full_path = (current_path / dirname).resolve()
                # Exclude based on absolute path
                if dir_full_path in exclude_paths:
                    dirs_to_remove.append(dirname)
                    if verbose:
                        print(f'Excluding directory by path: {dir_full_path}')
                # Exclude based on directory name
                elif dirname in exclude_names:
                    dirs_to_remove.append(dirname)
                    if verbose:
                        print(f'Excluding directory by name: {dir_full_path}')

            # Modify dirnames in-place to exclude specified directories
            for dirname in dirs_to_remove:
                dirnames.remove(dirname)

            # Exclude specific files by path or name
            for filename in filenames:
                file_full_path = (current_path / filename).resolve()
                # Exclude based on absolute path
                if file_full_path in exclude_file_paths:
                    if verbose:
                        print(f'Excluding file by path: {file_full_path}')
                    continue
                # Exclude based on file name
                if filename in exclude_names:
                    if verbose:
                        print(f'Excluding file by name: {file_full_path}')
                    continue
                # Write the file contents
                out.write(f'=== File: {file_full_path} ===\n')
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                    out.write(contents + '\n\n')
                except UnicodeDecodeError:
                    out.write('[Binary or non-text file cannot be displayed]\n\n')
                except Exception as e:
                    out.write(f'[Error reading file: {e}]\n\n')
    print(f'Directory contents have been dumped to "{output_file}"')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Recursively dump all files and their contents from a directory, excluding specified directories and files.')
    parser.add_argument('directory', help='Path to the directory to dump.')
    parser.add_argument('-o', '--output', default='directory_dump.txt', help='Output file name (default: directory_dump.txt).')
    parser.add_argument('-e', '--exclude', nargs='*', default=[], help='Directories to exclude from the dump (relative to the root directory). Separate multiple directories with spaces.')
    parser.add_argument('-f', '--exclude-file', nargs='*', default=[], help='Files to exclude from the dump (relative to the root directory). Separate multiple files with spaces.')
    parser.add_argument('-n', '--exclude-name', nargs='*', default=[], help='Names of files or directories to exclude, regardless of their location. Separate multiple names with spaces.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for debugging.')

    args = parser.parse_args()

    if not Path(args.directory).is_dir():
        print(f'Error: The directory "{args.directory}" does not exist or is not a directory.')
    else:
        dump_directory_contents(
            root_dir=args.directory,
            output_file=args.output,
            exclude_dirs=args.exclude,
            exclude_files=args.exclude_file,
            exclude_names=args.exclude_name,
            verbose=args.verbose
        )
