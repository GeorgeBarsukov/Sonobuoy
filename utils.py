import os

def find_wav_files(directory):
    """
    Recursively find all .wav files in a given directory and its subdirectories.
    
    Args:
        directory (str): The path to the directory to search.

    Returns:
        list: A list of paths to .wav files.
    """
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

# Example usage
if __name__ == "__main__":
    directory_to_search = input("Enter the directory to search: ").strip()
    wav_files = find_wav_files(directory_to_search)
    if wav_files:
        print("Found .wav files:")
        for wav_file in wav_files:
            print(wav_file)
    else:
        print("No .wav files found.")