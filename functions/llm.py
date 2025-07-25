def list_llm_models():
    llms = {
        "GPT-4": "OpenAI's most advanced LLM used in ChatGPT.",
        "Claude 3": "Anthropic's LLM known for safety and reasoning.",
        "Gemini 1.5": "Google DeepMind's powerful multimodal model.",
        "LLaMA 3": "Meta's open-source LLM series for research use.",
        "Mistral 7B": "Efficient, fast open-weight LLM by Mistral AI.",
        "Mixtral 8x7B": "Mixture-of-experts model by Mistral for high throughput."
    }

    print("📌 Top Large Language Models:\n")
    for name, desc in llms.items():
        print(f"{name}: {desc}")