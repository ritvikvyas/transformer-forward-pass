import torch
from transformers import BertTokenizer, BertForMaskedLM

def main():
    print("=== SIMPLE MASKED LANGUAGE MODELING ===")
    
    # 1. Load model
    print("1. Loading BERT MLM model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    print("✓ Model loaded!")
    
    # 2. Simple example
    # sentence = "The capital of France is [MASK]."
    # sentence = "The sky is [MASK]."
    sentence = "He went to the [MASK] to buy some bread."
    print(f"\n2. Input: {sentence}")
    
    # 3. Get predictions
    print("\n3. Getting predictions...")
    inputs = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Find mask position
    mask_pos = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0][0]
    
    # Get top 3 predictions
    mask_logits = logits[0, mask_pos, :]
    top_3 = torch.topk(mask_logits, 3)
    
    print("\n4. TOP 3 PREDICTIONS:")
    for i, (idx, score) in enumerate(zip(top_3.indices, top_3.values)):
        word = tokenizer.convert_ids_to_tokens(idx.item())
        print(f"   {i+1}. '{word}'")
    
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
