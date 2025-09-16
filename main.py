import torch
from transformers import BertTokenizer, BertModel

def main():
    print("=== SIMPLE TRANSFORMER FORWARD PASS ===")
    
    # 1. Load model and tokenizer
    print("1. Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("✓ Model loaded!")
    
    # 2. Tokenize input
    sentence = "Transformers are amazing!"
    print(f"\n2. Tokenizing: '{sentence}'")
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"✓ Tokens: {tokens}")
    
    # 3. Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    print("✓ Forward pass complete!")
    
    # 4. Show results
    print(f"\n4. RESULTS:")
    print(f"   - Input shape: {inputs['input_ids'].shape}")
    print(f"   - Output shape: {outputs.last_hidden_state.shape}")
    print(f"   - Hidden size: {outputs.last_hidden_state.shape[-1]}")
    print(f"   - Sequence length: {outputs.last_hidden_state.shape[1]}")
    
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
