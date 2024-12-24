import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re

def generate_caption(midi_data: dict) -> str:
    # Load base model and tokenizer
    model_name = "facebook/opt-350m"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load LoRA configuration and create PeftModel
    config = PeftConfig.from_pretrained("checkpoints/best_model")
    model = PeftModel.from_pretrained(base_model, "checkpoints/best_model")
    
    # Prepare input prompt
    prompt = f"Generate a caption for a MIDI file with the following properties:\nKey: {midi_data['key']}\nTempo: {midi_data['tempo']}\nGenre: {', '.join(midi_data['genre'])}\nMood: {', '.join(midi_data['mood'])}\n\nCaption:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate caption with better parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            min_length=40,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            top_p=0.92,
            early_stopping=True
        )
    
    raw_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    caption = raw_caption.split("Caption:")[-1].strip()
    return clean_caption(caption, midi_data)

def clean_caption(caption: str, midi_data: dict) -> str:
    """Clean and standardize model outputs"""
    
    # 1. Initial cleaning
    sentences = [s.strip() for s in caption.split('.') if s.strip()]
    cleaned_sentences = []
    
    # Track if we've mentioned key and tempo
    key_mentioned = False
    tempo_mentioned = False
    
    for sentence in sentences:
        # Skip incomplete sentences
        if len(sentence.split()) < 3:
            continue
            
        # Skip sentences that end abruptly
        if sentence.rstrip().endswith(('maintains a', 'features a', 'consists of a')):
            continue
            
        # Clean up chord progressions
        if 'chord progression' in sentence:
            if not any(chord in sentence for chord in ['Bb7', 'Dm7', 'Cmaj7']):
                continue
            if sentence.count('chord progression') > 1:
                sentence = sentence.replace('chord progression', 'harmony', 1)
                
        # Track key and tempo mentions
        if midi_data['key'] in sentence:
            key_mentioned = True
        if str(midi_data['tempo']) in sentence:
            tempo_mentioned = True
            
        cleaned_sentences.append(sentence)
    
    # 2. Ensure key and tempo are mentioned
    if not key_mentioned and cleaned_sentences:
        cleaned_sentences.insert(0, f"This piece is in the key of {midi_data['key']}.")
    if not tempo_mentioned and cleaned_sentences:
        cleaned_sentences.insert(1, f"It maintains a steady tempo of {midi_data['tempo']} BPM.")
    
    # 3. Join sentences and final cleaning
    output = '. '.join(cleaned_sentences)
    
    # Remove technical jargon
    output = re.sub(r'[A-G]b?#?\d*\/[A-G]b?#?\d*\/[A-G]b?#?\d*', 'complex harmonies', output)
    
    # Ensure proper ending
    if not output.endswith('.'):
        output += '.'
        
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--input_midi", type=str, required=True)
    args = parser.parse_args()
    
    print("\nTesting multiple musical styles:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Genre: {test_case['genre']}")
        print(f"Mood: {test_case['mood']}")
        print(f"Key: {test_case['key']}")
        print(f"Tempo: {test_case['tempo']}")
        print("\nGenerated Caption:")
        caption = generate_caption(test_case)
        print(caption)
        print("-" * 50)

# Test different musical styles
test_cases = [
    # Test Case 1: Simple Classical
    {
        'key': 'C major',
        'tempo': 120,
        'genre': ['classical'],
        'mood': ['peaceful']
    },
    # Test Case 2: Complex Jazz
    {
        'key': 'D minor',
        'tempo': 140,
        'genre': ['jazz', 'fusion'],
        'mood': ['energetic', 'complex']
    }
]

if __name__ == "__main__":
    main() 