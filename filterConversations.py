import os
import json

input_dir = './json_files'
output_file = 'filtered_conversations.json'

def filter_and_save_conversations():
    try:
        all_conversations = []
        
        # Get all files in the directory
        files = os.listdir(input_dir)
        
        for file in files:
            # Check if the file is a JSON file
            if file.endswith('.json'):
                file_path = os.path.join(input_dir, file)
                
                # Read and parse the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter conversations that have messages
                filtered_data = [convo for convo in data if convo.get('messages') and len(convo.get('messages')) > 0]
                
                # Add filtered conversations to our list
                all_conversations.extend(filtered_data)
        
        # Write the filtered conversations to a new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)
        
        print(f"Filtered conversations saved to {output_file}")
        print(f"Total number of conversations: {len(all_conversations)}")
    
    except Exception as error:
        print(f"Error filtering conversations: {error}")

if __name__ == "__main__":
    filter_and_save_conversations()