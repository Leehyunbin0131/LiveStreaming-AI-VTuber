# Ollama-Streaming-Voice-ChatAI

🎤 **Ollama-Streaming-Voice-ChatAI**: 실시간 음성 인식(STT) + AI 대화(Ollama) + 음성 합성(TTS)

Ollama 기반 AI와 GPT-SoVITS TTS를 활용한 **실시간 음성 스트리밍 대화 시스템**

## 📌 시스템 구성
- **STT**: 실시간 음성 인식
- **AI**: Ollama 기반 AI 대화 (benedict/linkbricks-llama3.1-korean:8b 모델 사용)
- **TTS**: GPT-SoVITS TTS를 활용한 음성 합성

---

## 🖥️ 환경 요구 사항
```
CUDA 12.3
PyTorch 2.2.0 cuda121
Python 3.9+
Ollama benedict/linkbricks-llama3.1-korean:8b
```

---

## 📥 모델 다운로드 및 설정

### 1️⃣ Ollama 모델 다운로드
Ollama에서 사용할 모델을 다운로드합니다.

### 2️⃣ GPT-SoVITS-v2 다운로드 및 학습
GPT-SoVITS-v2를 다운로드하고 GPT 모델과 SoVITS를 학습하여 TTS 모델을 생성합니다.

### 3️⃣ `tts_infer.yaml` 설정 변경
`GPT-SoVITS-v2` 폴더에서 `tts_infer.yaml`을 찾아 아래와 같이 수정합니다.

#### ✅ CUDA를 사용하는 경우
```yaml
device: cuda
is_half: true
t2s_weights_path: GPT_weights_v2/MY_TTS_MODEL.ckpt
version: v2
vits_weights_path: SoVITS_weights_v2/MY_TTS_MODEL.pth
```

#### ✅ CPU를 사용하는 경우
```yaml
device: cpu
is_half: false
t2s_weights_path: GPT_weights_v2/MY_TTS_MODEL.ckpt
version: v2
vits_weights_path: SoVITS_weights_v2/MY_TTS_MODEL.pth
```

### 4️⃣ 예제 설정 (`default_v2`)
```yaml
default_v2:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: GPT_weights_v2/MY_TTS_MODEL.ckpt
  version: v2
  vits_weights_path: SoVITS_weights_v2/MY_TTS_MODEL.pth
```

---

## 🚀 실행 방법

### 1️⃣ **GPT-SoVITS API 실행**
```bash
cd GPT-SoVITS-v2-240821
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

### 2️⃣ **데모 실행**
```bash
python DEMO_test.py
```

---

## 📜 라이선스
이 프로젝트는 오픈소스로 제공됩니다. 사용 시 라이선스를 참고하세요.

---

## 📞 문의
궁금한 점이나 개선 사항이 있다면 이슈를 남겨주세요! 😊
