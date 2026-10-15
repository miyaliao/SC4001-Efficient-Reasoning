"""CPU-only Qwen3 chat-template smoke test: no model weights loaded."""
from transformers import AutoTokenizer

MODEL_PATH = "/data1/lmy/models/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
messages = [{"role": "user", "content": "What is 15% of 200?"}]
for thinking in (False, True):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    print(f"\n=== thinking={thinking} ===")
    print(prompt[-400:])
