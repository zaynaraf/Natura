import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rag.retriever import retrieve
import gradio as gr
import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === Load OpenChat 3.5 4bit ===
model_path = "./models/openchat-3.5-0106"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# === System Prompt ===
system_prompt = """
GPT4 Correct User: You are Natura — a warm, knowledgeable skincare assistant who helps people using natural remedies.

You speak like a thoughtful friend who truly understands skin. Stay curious, kind, and helpful — never robotic or clinical. Ask follow-up questions, use plain language, and guide the user with care.

---

🎯 Your job is to help users find and explain natural skincare remedies, using a trusted remedy database.

Before suggesting a remedy, make sure you know:
1. Their skin type (e.g., oily, dry, sensitive, combination)
2. Their main concern (e.g., acne, redness, dryness, dark spots)
3. Any known allergies or sensitivities

---

📥 If a remedy is retrieved, it will be shown under this heading:

### Retrieved Remedy:
Remedy:
[exact remedy name]

Instructions:
[clear steps to follow]

Source Title:
[exact article or source name]

Source URL:
[exact link]

Caution:
[optional warning, only if included]

You must repeat the remedy name, instructions, title, and URL exactly as shown. Do not paraphrase, rename, or invent anything. Present them casually, but keep them faithful to what’s given.

Never invent a remedy if one is retrieved. Never skip source or caution. Stay human, relaxed, and helpful.
<|end_of_turn|>
GPT4 Correct Assistant:
""".strip()

# === Remedy Retrieval ===
def get_relevant_remedy(chat_history, top_k=1):
    results = retrieve(chat_history, top_k=top_k)
    if not results:
        return None
    r = results[0]
    remedy_block = (
        f"Remedy:\n{r['name']}\n\n"
        f"Instructions:\n{r['instructions']}\n\n"
        f"Source Title:\n{r['source_title']}\n\n"
        f"Source URL:\n{r['source_url']}"
    )
    if r.get("cautions") and r["cautions"].strip():
        remedy_block += f"\n\nCaution:\n{r['cautions']}"
    return remedy_block

# === Assistant Reply Generation ===
def chat_with_gradio(history, user_input):
    try:
        chat_blocks = ""
        for u, a in history:
            chat_blocks += f"GPT4 Correct User: {u}<|end_of_turn|>\n"
            chat_blocks += f"GPT4 Correct Assistant: {a}<|end_of_turn|>\n"

        rag_input = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history] + [f"User: {user_input}"])
        remedy_context = get_relevant_remedy(rag_input)

        prompt = f"{system_prompt}\n"
        if remedy_context:
            prompt += f"### Retrieved Remedy:\n{remedy_context}\n\n"
        prompt += f"{chat_blocks}GPT4 Correct User: {user_input}<|end_of_turn|>\nGPT4 Correct Assistant:"

        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072, padding=True)
        input_ids = tokens.input_ids.to(model.device)
        attention_mask = tokens.attention_mask.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = decoded.split("GPT4 Correct Assistant:")[-1].strip().split("GPT4 Correct User:")[0].strip()

        history.append((user_input, reply))
        return history, ""

    except Exception as e:
        error_msg = traceback.format_exc()
        history.append((user_input, f"❌ Error:\n```\n{error_msg}\n```"))
        return history, ""

# === Gradio Blocks UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h1 style='font-size: 2.5rem;'>🌿 <strong>Natura</strong></h1>
            <p style='font-size: 1.1rem; color: #ccc;'>Your natural skincare assistant powered by OpenChat + RAG</p>
        </div>
        """
    )

    chatbot = gr.Chatbot(height=500, bubble_full_width=False, show_copy_button=True)
    state = gr.State([])

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
            user_input = gr.Textbox(
                placeholder="Ask Natura anything...",
                show_label=False,
                container=False
            )
        with gr.Column(scale=2, min_width=100):
            send_btn = gr.Button("Send", variant="primary")

    send_btn.click(chat_with_gradio, [state, user_input], [chatbot, user_input])
    user_input.submit(chat_with_gradio, [state, user_input], [chatbot, user_input])

# === Launch App ===
demo.launch(share=True)
