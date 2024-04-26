import json
import os

def modify_audio_filepath(json_file):
    modified_entries = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            audio_filepath = entry.get("audio_filepath")
            # Normalize the file path to replace backslashes with forward slashes
            audio_filepath = os.path.normpath(audio_filepath)
            # Extract the filename from the normalized path
            filename = os.path.basename(audio_filepath)
            entry["audio_filepath"] = filename
            modified_entries.append(entry)

    # Write the modified data back to the JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        for entry in modified_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')  

# Example usage:
json_file = "manifest.json"  
modify_audio_filepath(json_file)
