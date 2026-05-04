import dspy
print("Loading LM...")
lm = dspy.LM("huggingface/meta-llama/Llama-2-7b-hf", max_tokens=1)
print("LM initialized!")
