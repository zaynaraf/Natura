📌 OpenChat Model Setup (Required for Natura Chatbot)

This folder is expected to contain the OpenChat 3.5 7B model (4-bit quantized version).

Instructions to download:

1. Go to the Hugging Face model page:
👉 https://huggingface.co/openchat/openchat-3.5-0106

2. Download the model files (you must be logged in to Hugging Face and have accepted the model license).

3. Place all downloaded files into a folder named:
./models/openchat-3.5-0106

This folder must match the `model_path` in `main.py`.

✅ Example file structure after download:
models/
└── openchat-3.5-0106/
    ├── config.json
    ├── generation_config.json
    ├── pytorch_model-00001-of-00003.bin
    ├── tokenizer.model
    └── ...