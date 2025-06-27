from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import random
import torch

# Preferred model (GPT-2.5 equivalent)
MODEL_CANDIDATES = [
    "gpt2-large",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2"  # fallback
]

# Load model and tokenizer with fallback
for model_name in MODEL_CANDIDATES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Loaded model: {model_name}")
        break
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")

# Genre presets
GENRES = {
    "Fantasy": "In a realm where dragons soar and magic flows, ",
    "Sci-Fi": "In the year 2478, humanity had reached the edge of the galaxy. ",
    "Mystery": "The room was quiet, but the blood on the floor whispered secrets. ",
    "Romance": "Under the Parisian sky, their eyes met for the first time. ",
    "Adventure": "With only a compass and a map, she stepped into the unknown. "
}

# Story generator with prompt
def generate_story(genre, custom_prompt, max_length=150, temperature=0.9, top_p=0.95):
    base_prompt = GENRES.get(genre, "Once upon a time, ")
    final_prompt = base_prompt + custom_prompt.strip()

    inputs = tokenizer(final_prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
light_css = """
body, .gradio-container {
    font-family: 'Poppins', sans-serif;
    background: #fefae0;
    color: #1e293b;
}
textarea, input, .output-textbox {
    background-color: #ffffff !important;
    color: #1e293b !important;
    font-size: 16px;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px;
    padding: 12px;
}
button {
    background-color: #7c3aed !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 16px;
    transition: background 0.3s ease;
}
button:hover {
    background-color: #5b21b6 !important;
}
.output-textbox {
    background-image: url('https://www.transparenttextures.com/patterns/paper-fibers.png');
    background-color: #fefae0 !important;
    font-family: 'Georgia', serif;
    font-size: 18px;
    padding: 20px;
    border-left: 8px solid #d97706;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(css=light_css) as demo:
    gr.Markdown("# 🔮 Your Story Awaits")
    gr.Markdown("Craft immersive stories powered by GPT-2.5 with selectable genres and creative freedom.")

    with gr.Row():
        genre = gr.Dropdown(label="Select a Genre", choices=list(GENRES.keys()), value="Fantasy")
        prompt = gr.Textbox(label="Custom Prompt", placeholder="Add a twist or unique idea...")

    with gr.Row():
        max_len = gr.Slider(50, 300, step=10, value=150, label="Max Length")
        temp = gr.Slider(0.1, 2.0, step=0.1, value=0.9, label="Temperature")
        topp = gr.Slider(0.5, 1.0, step=0.05, value=0.95, label="Top-p")

    with gr.Row():
        generate_btn = gr.Button("✨ Generate Story")
        clear_btn = gr.Button("❌ Clear")

    output_box = gr.Textbox(label="Generated Story", lines=15)

    generate_btn.click(generate_story, inputs=[genre, prompt, max_len, temp, topp], outputs=output_box)
    clear_btn.click(fn=lambda: "", inputs=None, outputs=output_box)

demo.launch()