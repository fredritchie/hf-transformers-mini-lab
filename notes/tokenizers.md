# Tokenizers

## Focus points

- Tokenization converts text into tokens.
- `input_ids` are numeric token identifiers.
- `attention_mask` tells the model which tokens are real and which are padding.
- Padding makes sequences the same length in a batch.
- Truncation cuts long text to fit the model context limit.
- Fast tokenizers provide offset mapping between tokens and original text.

## Important concepts

- BPE
- WordPiece
- Unigram
- Offset mapping
- Padding
- Truncation
- Batch tokenization
