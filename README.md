# Ollama-Streaming-Voice-ChatAI
🎤 Ollama-Streaming-Voice-ChatAI: 실시간 음성 인식(STT) + AI 대화(Ollama) + 음성 합성(TTS)   Ollama 기반 AI와 GPT-SoVITS TTS를 활용한 **실시간 음성 스트리밍 대화 시스템**  

------------------------
CUDA 12.3
PyTorch 2.2.0 cuda121
Python 3.9+
Ollama gemma2 9B
------------------------


Ollama에서 사용할 모델을 다운로드.
GPT-SoVITS-v2를 다운로드후 GPT모델과 SoVITS학습하여 tts모델 생성. 
GPT-SoVITS-v2에서 tts_infer.yaml을 찾고 아래처럼 수정.

쿠다를 사용하는 경우
device: cuda
  is_half: true
  t2s_weights_path: GPT_weights_v2 .ckpt 확장자 명의 학습된 본인의 모델파일 이름.
  version: v2
  vits_weights_path: SoVITS_weights_v2 .pth 확장자 명의 학습된 본인의 모델파일 이름.

CPU를 사용하는경우
device: cpu
  is_half: false
  t2s_weights_path: GPT_weights_v2 .ckpt 확장자 명의 학습된 본인의 모델파일 이름.
  version: v2
  vits_weights_path: SoVITS_weights_v2 .pth 확장자 명의 학습된 본인의 모델파일 이름.

-예시-
default_v2:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: GPT_weights_v2/MY_TTS_MODEL.ckpt
  version: v2
  vits_weights_path: SoVITS_weights_v2/MY_TTS_MODEL.pth
