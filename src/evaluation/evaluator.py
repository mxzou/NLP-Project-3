import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

class CaptionEvaluator:
    def __init__(self, checkpoint_path: str):
        print("Initializing evaluator...")
        self.model_name = "facebook/opt-350m"
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load LoRA model
        config = PeftConfig.from_pretrained(checkpoint_path)
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("✓ Model loaded successfully")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
    def generate_caption(self, midi_data: Dict) -> str:
        # Prepare input prompt
        prompt = f"Generate a caption for a MIDI file with the following properties:\nKey: {midi_data['key']}\nTempo: {midi_data['tempo']}\nGenre: {', '.join(midi_data['genre'])}\nMood: {', '.join(midi_data['mood'])}\n\nCaption:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=50,
                temperature=0.8,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Clean up the caption
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        caption = caption.split("Caption:")[-1].strip()
        
        # Remove repetitive phrases
        sentences = caption.split('.')
        unique_sentences = []
        for s in sentences:
            s = s.strip()
            if s and s not in unique_sentences:
                unique_sentences.append(s)
        
        return '. '.join(unique_sentences) + '.'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    evaluator = CaptionEvaluator(args.checkpoint_path)
    
    test_cases = [
        {
            'key': 'C major',
            'tempo': 120,
            'genre': ['classical'],
            'mood': ['peaceful']
        },
        {
            'key': 'D minor', 
            'tempo': 140,
            'genre': ['jazz', 'fusion'],
            'mood': ['energetic', 'complex']
        },
        {
            'key': 'F# major',
            'tempo': 200,
            'genre': ['experimental'],
            'mood': ['intense', 'avant-garde']
        }
    ]
    
    results = []
    print("\nGenerating captions for test cases:")
    for i, test_case in enumerate(tqdm(test_cases), 1):
        caption = evaluator.generate_caption(test_case)
        results.append({
            'test_case': test_case,
            'generated_caption': caption
        })
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
