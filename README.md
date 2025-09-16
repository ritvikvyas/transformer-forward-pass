# Transformer Forward Pass Visualization

A simple implementation to demonstrate the forward pass through a pre-trained BERT transformer model.

## What This Does

1. **Loads BERT model** from HuggingFace (`bert-base-uncased`)
2. **Tokenizes input** sentence: "Transformers are amazing!"
3. **Runs forward pass** through the model
4. **Shows tensor shapes** and basic data flow

## Quick Start

### 1. Run the Implementation
```bash
python3 main.py
```

**Note**: First run will download ~440MB BERT model.

## Installation

```bash
pip3 install -r requirements.txt
```

## Expected Output

```
=== SIMPLE TRANSFORMER FORWARD PASS ===
1. Loading BERT model...
✓ Model loaded!

2. Tokenizing: 'Transformers are amazing!'
✓ Tokens: ['[CLS]', 'transformers', 'are', 'amazing', '!', '[SEP]']

3. Running forward pass...
✓ Forward pass complete!

4. RESULTS:
   - Input shape: torch.Size([1, 6])
   - Output shape: torch.Size([1, 6, 768])
   - Hidden size: 768
   - Sequence length: 6

=== DONE ===
```

## What Happens

1. **Input**: "Transformers are amazing!" → `[CLS] transformers are amazing ! [SEP]`
2. **Model**: 12 transformer layers process the sequence
3. **Output**: Hidden states with shape `(batch=1, sequence=6, hidden=768)`
4. **Result**: Each token gets a 768-dimensional contextual representation

## MLM Extension

A simple extension that shows masked language modeling:

```bash
python3 mlm_extension.py
```

## Files

- `main.py` - The core implementation
- `mlm_extension.py` - Simple masked language modeling
- `requirements.txt` - Just PyTorch and Transformers
- `README.md` - This documentation

## Model Details

- **BERT Base**: 12 layers, 12 attention heads, 768 hidden size
- **Vocabulary**: 30,522 tokens
- **Parameters**: ~110 million

That's it! Simple and focused on the core concept, with a basic MLM extension.
