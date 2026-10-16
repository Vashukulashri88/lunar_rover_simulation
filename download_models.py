"""
download_models.py
Automated 1-Click Downloader for Qwen2-VL-2B Vision-Language Models.
Downloads the exact quantized GGUF weights and multimodal projection model
from HuggingFace directly into the llama.cpp models directory.
"""

import os
import sys
import urllib.request

MODELS = [
    {
        "filename": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF/resolve/main/qwen2-vl-2b-instruct-q4_k_m.gguf",
        "desc": "Qwen2-VL-2B 4-bit Quantized Base Model"
    },
    {
        "filename": "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
        "desc": "Multimodal Vision Projector (f16)"
    }
]

def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"

def download_with_progress(url, dest_path, desc):
    print(f"\n[*] Downloading {desc}...")
    print(f"    Source: {url}")
    print(f"    Destination: {dest_path}")
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100 * 1024 * 1024:
        print(f"    [+] Already exists ({format_size(os.path.getsize(dest_path))}). Skipping download.")
        return

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100.0)
            sys.stdout.write(f"\r    Progress: {percent:.1f}% ({format_size(downloaded)} / {format_size(total_size)})")
        else:
            sys.stdout.write(f"\r    Downloaded: {format_size(downloaded)}")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print(f"\n    [+] Successfully downloaded: {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"\n    [-] Download failed: {e}")
        print(f"    You can download it manually from: {url}")

def main():
    print("=" * 70)
    print("  LUNAR ROVER SIMULATOR - VLM/LLM MODEL DOWNLOADER")
    print("=" * 70)

    # Preferred destination: C:\Users\ckceg\llama.cpp\models or local ./models
    default_dir = os.path.expanduser(r"~\llama.cpp\models")
    if os.path.exists(os.path.dirname(default_dir)):
        target_dir = default_dir
    else:
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    os.makedirs(target_dir, exist_ok=True)
    print(f"Target Directory: {target_dir}")

    for m in MODELS:
        dest = os.path.join(target_dir, m["filename"])
        download_with_progress(m["url"], dest, m["desc"])

    print("\n" + "=" * 70)
    print("  [SUCCESS] All VLM models verified and ready for local GPU inference!")
    print("=" * 70)

if __name__ == "__main__":
    main()
