#!/usr/bin/env python3
"""
Synthesize the Douyin 'Mambo' (曼波 / 待兼诗歌剧) voiceover commentary on the origin of KL
using RVC-Boss/GPT-SoVITS on GPU.
"""
import io
import os
import subprocess
import sys
import time
import numpy as np
import soundfile as sf

GPT_SOVITS_DIR = "/ssd2/jinzili/GPT-SoVITS"
MAMBO_MODELS_DIR = "/ssd2/jinzili/MamboTTS/models"
OUTPUT_DIR = "/ssd2/jinzili/GPT-SoVITS/output_kl_mambo"

SOVITS_MODEL = os.path.join(MAMBO_MODELS_DIR, "manbo_e8_s168.pth")
GPT_MODEL = os.path.join(MAMBO_MODELS_DIR, "manbo-e10.ckpt")
REF_WAV = os.path.join(MAMBO_MODELS_DIR, "refer.wav")
DEFAULT_REF_TEXT = "大家好，欢迎来到我的频道，今天给大家分享一个有趣的内容"
DEFAULT_REF_LANG = "zh"

# Paragraph-structured commentary lines (each item in a paragraph is a sentence unit)
PARAGRAPHS = [
    [
        "曼波，曼波，欧马自利曼波！",
        "家人们，如果我告诉你，《伪人降临》里最强大的神秘组织 KL，全书其实连一个活人都没有出现过，你敢信吗？",
        "今天曼波就带大家扒一扒，KL 背后最让人破防的终极伏笔！",
    ],
    [
        "在小说前期，主角拿到的所有保命神装，全都印着 KL 的标志。",
        "而救死扶伤的传奇医疗团 KLG，全称其实是 KL Goat，也就是 KL 的山羊分支。",
        "那这个神通广大的母体组织 KL，到底去哪了？",
    ],
    [
        "直到故事中后期，真相才彻底揭开。KL 的全称，是凯丽斯启明军团。",
        "当年后室通道在人类世界裂开，恶魔疯狂涌出。",
        "KL 的先驱们在裂缝上建起了一座八角笼，定下铁律：恶魔想进人间，必须先在笼子里打赢人类。",
        "但所有人都被骗了！八角笼根本不是什么擂台，而是一座双重封印！",
        "真正的规则是，只要格斗还在继续，地狱通道就会被死死锁住！",
    ],
    [
        "后来恶魔识破了计谋，不再派人上场。",
        "眼看封印就要失效，KL 的初代先驱们做了一个违背常理的决定：",
        "他们主动跳进裂缝，在后室里凭空建起了一座人类城市，逼迫恶魔不得不继续比赛。",
        "一九五八年，格斗彻底停摆，全体 KL 成员为了锁死通道，在地下五百米与恶魔决一死战，全员壮烈牺牲！",
        "这就是为什么，他们一夜之间从人间蒸发，再也没有回来。",
    ],
    [
        "更刀的是，这群先驱死后，依然在保护后来者。",
        "你以为 KL 雷达里那颗会跳动的肉球是怪物器官？不，那是先驱们用自己的大脑做成的活体探测器！",
        "你以为吃骨头的补给箱是恶魔科技？那是阵亡将士的遗骸化作的宝箱！",
        "还有最逆天的 KL 独眼复活药，为什么队友喝下复活药之后，会直接长出恶魔的眼睛、心脏和骨刺？",
        "因为那根本不是药，而是先驱们用自己的血肉和灵魂熬成的生命火种！",
    ],
    [
        "正如他们在补给箱上刻下的那句遗言：人人为我，我为人人。",
        "这群九死一生的孤勇者，哪怕死了，也把自己的骨肉化作武器，留给了素未谋面的后来人。",
        "看到这里的家人们，把致敬打在公屏上！曼波，我们下期再见，拜拜！",
    ],
]


def trim_trailing_silence(audio: np.ndarray, sr: int, keep_sec: float = 0.05, threshold: float = 0.003) -> np.ndarray:
    """Trim excessive trailing silence added by get_tts_wav so we can control pauses precisely."""
    abs_audio = np.abs(audio)
    non_silent = np.where(abs_audio > threshold)[0]
    if len(non_silent) == 0:
        return audio
    end_idx = min(len(audio), non_silent[-1] + int(sr * keep_sec))
    start_idx = max(0, non_silent[0] - int(sr * 0.02))
    return audio[start_idx:end_idx]


def synthesize_version(api_mod, speed: float, sent_pause_sec: float, para_pause_sec: float, out_prefix: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_chunks = []
    sr = 32000

    t_start = time.time()
    total_sents = sum(len(p) for p in PARAGRAPHS)
    sent_idx = 0

    for p_idx, para in enumerate(PARAGRAPHS):
        for s_idx, sentence in enumerate(para):
            sent_idx += 1
            print(f"[{out_prefix}] ({sent_idx}/{total_sents}) Synthesizing: {sentence}", flush=True)
            gen = api_mod.get_tts_wav(
                ref_wav_path=REF_WAV,
                prompt_text=DEFAULT_REF_TEXT,
                prompt_language=DEFAULT_REF_LANG,
                text=sentence,
                text_language="zh",
                top_k=15,
                top_p=0.8,
                temperature=0.6,
                speed=speed,
            )
            wav_bytes = b"".join(gen)
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = trim_trailing_silence(data, sr)
            all_chunks.append(data)

            # Add natural pause
            pause_sec = para_pause_sec if (s_idx == len(para) - 1) else sent_pause_sec
            all_chunks.append(np.zeros(int(sr * pause_sec), dtype=np.float32))

    full_audio = np.concatenate(all_chunks, axis=0)
    # Peak normalize to -1 dBFS (0.891) for clean, loud Douyin voiceover
    peak = np.abs(full_audio).max()
    if peak > 0:
        full_audio = full_audio * (0.89 / peak)

    wav_path = os.path.join(OUTPUT_DIR, f"{out_prefix}.wav")
    mp3_path = os.path.join(OUTPUT_DIR, f"{out_prefix}.mp3")
    sf.write(wav_path, full_audio, sr, format="WAV", subtype="PCM_16")

    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    duration = len(full_audio) / sr
    elapsed = time.time() - t_start
    print(
        f"[{out_prefix}] Done in {elapsed:.1f}s -> {wav_path} ({duration:.1f}s, sr={sr}Hz) & {mp3_path}",
        flush=True,
    )


def main():
    os.chdir(GPT_SOVITS_DIR)
    sys.path.insert(0, GPT_SOVITS_DIR)
    sys.path.insert(0, os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS"))

    sys.argv = [
        "api.py",
        "-s", SOVITS_MODEL,
        "-g", GPT_MODEL,
        "-dr", REF_WAV,
        "-dt", DEFAULT_REF_TEXT,
        "-dl", DEFAULT_REF_LANG,
        "-d", "cuda",
    ]

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}", flush=True)
    import api as api_mod

    if "default" not in api_mod.speaker_list:
        print("Explicitly loading GPT & SoVITS weights to surface any error...", flush=True)
        gpt = api_mod.get_gpt_weights(GPT_MODEL)
        sovits = api_mod.get_sovits_weights(SOVITS_MODEL)
        api_mod.speaker_list["default"] = api_mod.Speaker(name="default", gpt=gpt, sovits=sovits)

    # 1. Douyin commentary pacing (1.08x speed, compact pauses)
    synthesize_version(
        api_mod,
        speed=1.08,
        sent_pause_sec=0.18,
        para_pause_sec=0.38,
        out_prefix="kl_origin_mambo_douyin",
    )

    # 2. Standard pacing (1.00x speed, relaxed pauses)
    synthesize_version(
        api_mod,
        speed=1.00,
        sent_pause_sec=0.24,
        para_pause_sec=0.45,
        out_prefix="kl_origin_mambo_standard",
    )


if __name__ == "__main__":
    main()
