import json
import os
import librosa
import torch
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# # Load the ASR model
# # model_name = "openai/whisper-medium"
# local_model_path = "odia_models\\whisper-medium-or_alldata_multigpu"
# model = Wav2Vec2ForCTC.from_pretrained(local_model_path)
# # model = Wav2Vec2ForCTC.from_pretrained(model_name)
# # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(local_model_path)
# tokenizer = Wav2Vec2Tokenizer.from_pretrained(local_model_path)

# Load dataset
dataset_path = "kathbath\\kathbath"
languages = os.listdir(dataset_path)
# print(languages)
total_wer = 0
total_count = 0
lang_code={'bengali':'bn', 'gujarati':'gu', 'hindi':'hi', 'kannada':'kn', 'malayalam':'ml', 'marathi':'mr', 'odia':'or', 'punjabi':'pa', 'sanskrit':'sa', 'tamil':'ta', 'telugu':'te', 'urdu':'ur'}
for language in languages:
    # Load the ASR model
    local_model_path = f"{language}_models\\whisper-medium-{lang_code[language]}_alldata_multigpu"
    print(local_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(local_model_path)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(local_model_path)



    # if (language!="odia"):
    #     print(language)
    #     continue
    # language_path = os.path.join(dataset_path, '\\')
    # language_path = os.path.join(language_path, language)
    language_path = os.path.join(dataset_path, language)
    print(language_path)
    if not os.path.isdir(language_path):
        print('no')
        continue

    audio_path = os.path.join(language_path, "wavs")        #changed (language_path, "audio") -> (language_path, "wavs")
    manifest_path = os.path.join(language_path, "manifest.json")
    print(audio_path)
    print(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            print(entry)
            audio_file = os.path.join(audio_path, entry["audio_filepath"])
            print(audio_file)

            # Transcribe audio
            audio_input, _ = librosa.load(audio_file, sr=16000)
            input_values = tokenizer(audio_input, return_tensors="pt").input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Convert IDs to text
            transcription = tokenizer.batch_decode(predicted_ids)[0]

            # Calculate WER
            reference = entry["text"]
            current_wer = wer(reference, transcription)
            total_wer += current_wer
            total_count += 1

    

avg_wer = total_wer / total_count
print(f"Average WER: {avg_wer}")
