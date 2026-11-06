#!/usr/bin/env bash
set -euo pipefail

export PATH="/ssd2/jinzili/.local/bin:$PATH"
export UV_CACHE_DIR="/ssd2/jinzili/.cache/uv"
export HF_HOME="/ssd2/jinzili/.cache/huggingface"

echo "=== [1/5] Extracting Mambo voice models ==="
mkdir -p /ssd2/jinzili/MamboTTS/models
unzip -j -o /ssd2/jinzili/MamboTTS/MamboTTS-v1.2.1-full.zip \
    "*manbo_e8_s168.pth" "*manbo-e10.ckpt" "*refer.wav" \
    -d /ssd2/jinzili/MamboTTS/models/
ls -lh /ssd2/jinzili/MamboTTS/models/

echo "=== [2/5] Preparing Python 3.10 venv (/ssd2/jinzili/envs/gptsovits) ==="
if [ ! -d "/ssd2/jinzili/envs/gptsovits" ]; then
    uv venv /ssd2/jinzili/envs/gptsovits --python 3.10
fi
export VIRTUAL_ENV="/ssd2/jinzili/envs/gptsovits"
PY="/ssd2/jinzili/envs/gptsovits/bin/python"

# Ensure PyTorch is compiled for CUDA 12.x (driver on node is CUDA 12.9 / 12090)
if ! "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" >/dev/null 2>&1; then
    echo "Installing PyTorch for CUDA 12.6..."
    uv pip install --python "$PY" torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
fi

if ! "$PY" -c "import librosa, transformers, funasr, matplotlib, g2p_en, pyopenjtalk, fast_langdetect" >/dev/null 2>&1; then
    echo "Installing GPT-SoVITS inference dependencies..."
    uv pip install --python "$PY" \
        "numpy<2.0" scipy tensorboard "librosa==0.10.2" soundfile numba \
        "pytorch-lightning>=2.4" ffmpeg-python onnxruntime-gpu tqdm \
        cn2an pypinyin "pyopenjtalk>=0.4.1" g2p_en modelscope sentencepiece \
        "transformers>=4.41,<5" "peft<0.18.0" chardet PyYAML psutil \
        jieba_fast jieba split-lang "fast_langdetect>=0.3.1" wordsegment \
        rotary_embedding_torch ToJyutping g2pk2 ko_pron opencc \
        "fastapi>=0.115.2" uvicorn x_transformers "torchmetrics<=1.5" \
        "pydantic<=2.10.6" einops huggingface_hub nltk requests \
        matplotlib "av>=11" "ctranslate2>=4.0,<5" "funasr>=1.3.7"
fi

echo "=== [3/5] Downloading pretrained BERT, CNHubert, G2PW, NLTK, OpenJTalk ==="
cd /ssd2/jinzili/GPT-SoVITS
mkdir -p /ssd2/jinzili/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect

if [ ! -f "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin" ] || \
   [ ! -f "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin" ]; then
    "$PY" -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lj1995/GPT-SoVITS",
    allow_patterns=["chinese-hubert-base/*", "chinese-roberta-wwm-ext-large/*"],
    local_dir="/ssd2/jinzili/GPT-SoVITS/GPT_SoVITS/pretrained_models",
)
'
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Downloading G2PWModel.zip..."
    curl -L -sS "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip" -o G2PWModel.zip
    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -f G2PWModel.zip
fi

if [ ! -d "/ssd2/jinzili/envs/gptsovits/nltk_data" ]; then
    echo "Downloading nltk_data.zip..."
    curl -L -sS "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip" -o nltk_data.zip
    unzip -q -o nltk_data.zip -d /ssd2/jinzili/envs/gptsovits
    unzip -q -o nltk_data.zip -d /root
    rm -f nltk_data.zip
fi

PYOPENJTALK_DIR=$("$PY" -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")
if [ ! -d "$PYOPENJTALK_DIR/open_jtalk_dic_utf_8-1.11" ]; then
    echo "Downloading open_jtalk_dic..."
    curl -L -sS "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz" -o open_jtalk_dic_utf_8-1.11.tar.gz
    tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_DIR"
    rm -f open_jtalk_dic_utf_8-1.11.tar.gz
fi

echo "=== [4/5] Running GPT-SoVITS Mambo synthesis on GPU 1 ==="
export CUDA_VISIBLE_DEVICES=1
export NLTK_DATA="/ssd2/jinzili/envs/gptsovits/nltk_data"
"$PY" /ssd2/jinzili/synthesize_kl_mambo.py

echo "=== [5/5] Output files ==="
chmod -R a+rX /ssd2/jinzili/GPT-SoVITS/output_kl_mambo/
ls -lh /ssd2/jinzili/GPT-SoVITS/output_kl_mambo/
