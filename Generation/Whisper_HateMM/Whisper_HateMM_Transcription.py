import os
import whisper
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

audio_folder = "/Users/jiaqiliu/Desktop/hate_wav/"
output_file = "/Users/jiaqiliu/Desktop/output.json" 
model_name = "base" 

# Load Whisper model
model = whisper.load_model(model_name)
# Dictionary to store transcriptions
transcriptions = {}
# Iterate through all .wav files in the folder
for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        print(f"Processing: {file_name}")

        try:
            # Transcribe the audio using Whisper
            result = model.transcribe(file_path)
            transcriptions[file_name] = result["text"]
            print(f"Finished: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            transcriptions[file_name] = None  # Record as None if transcription fails
# Save the results as a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=4)
print(f"All files processed. Transcriptions saved to {output_file}")






