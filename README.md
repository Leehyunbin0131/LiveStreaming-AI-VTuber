# Live Streaming AI VTuber for VTubeStudio

🎤 **Live Streaming AI VTuber**: 실시간 AI 대화 + TTS 음성 합성 + VTubeStudio 연동

Ollama 기반 AI와 GPT-SoVITS TTS,RealtimeSTT를 활용한 **스트리밍 및 VTubeStudio 연동 음성 AI버튜버 시스템**

> ⚠️ **이 코드는 데모 버전으로 여러 문제가 있을 수 있습니다.**
> **계속 업데이트 중이므로 최신 버전을 유지해주세요.**

---

## 모든 기여자들에게 감사드립니다 :3

<a href="https://github.com/Leehyunbin0131/LiveStreaming-AI-VTuber/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=Leehyunbin0131/LiveStreaming-AI-VTuber" />
</a>

---

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
</p>


## 🖥️ 환경 요구 사항
```
CUDA 12.3
PyTorch 2.2.0 cuda121
Python 3.9+
Ollama 
```

---

## 📥 모델 다운로드 및 설정

### 1️⃣ GPT-SoVITS 다운로드
[GPT-SoVITS-v2 다운로드](https://github.com/RVC-Boss/GPT-SoVITS/releases/tag/20240821v2)

GPT-SoVITS-v2를 다운로드한 후, GPT 모델과 SoVITS를 학습하여 TTS 모델을 생성합니다.

`GPT-SoVITS-v2/configs/tts_infer.yaml` 파일을 수정합니다.

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

#### ✅ 예제 설정 (`default_v2`)
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

### 2️⃣ Ollama 모델 다운로드 및 설정
[Ollama 다운로드](https://ollama.com/download)
#### **Ollama 모델 다운로드**
[Ollama Models 다운로드](https://ollama.com/search)
1. `DEMO_test.py`에서 **Ollama 모델이름**을 다운로드한 모델의 이름으로 변경합니다.
2. 모델 이름은 아래 명령어로 확인할 수 있습니다.
```bash
ollama list
```

**예제 코드 (`DEMO_test.py` 수정 부분)**
```python
class OllamaChat:
    def __init__(self, model: str = "Ollama 모델이름") -> None:
        self.model = model
        self.system_message = {
            'role': 'system',
            'content': (
                "당신은 인터넷 AI 방송 크리에이터입니다. "
                "Ollama 기반의 인공지능 AI이며, 시청자들과 소통하는 것을 즐기고 털털한 성격을 가졌습니다. "
                "존댓말을 사용하지 말고, 대화는 짧고 간결하게 하며, 정확한 정보를 전달하세요."
            )
        }
        self.conversation_history = []
```

#### **참조 오디오 세팅**
```python
DEFAULT_REF_AUDIO = r"C:\Users\unit6\Documents\Test\My_tts\My_BaseTTS_v2.wav"  # 참조 오디오 경로
DEFAULT_PROMPT_TEXT = ""  # 참조 텍스트 (한국어는 빈칸으로)
DEFAULT_PROMPT_LANG = "ko"  # 참조 언어 설정
```

---

## 🎥 VTubeStudio 설정
VTubeStudio 설정에서 **VTubeStudio API 시작 옵션을 ON**으로 활성화합니다.

---

## 🚀 실행 방법

### 1️⃣ **VTubeStudio**
VTubeStudio 실행.

### 2️⃣ **GPT-SoVITS API 실행**
```bash
cd GPT-SoVITS-v2-240821
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

### 3️⃣ **데모 실행**
```bash
python DEMO_test.py
```

### 4️⃣ **VTubeStudio**
VTubeStudio 화면의 **권한 요청을 허용**합니다.

## 📹 영상 [https://youtu.be/XH1xBt59EGw](https://youtu.be/XH1xBt59EGw)
[![데모 영상](https://img.youtube.com/vi/XH1xBt59EGw/0.jpg)](https://youtu.be/XH1xBt59EGw)

---

## ⚠️ 문제 해결
만약 **VTubeStudio 모델이 움직이지 않는다면**, `DEMO_vts_api_helper.py`의 `param_id`가 실제 ID와 일치하는지 확인하세요.

```python
def inject_mouth_value(self, mouth_value: float, face_found: bool = True, param_id: str = "MouthOpen"):  # 실제 ID 확인
    if not self.authenticated:
        return
    mouth_value = max(0.0, min(1.0, mouth_value))
    req_id = str(uuid.uuid4())[:8]
    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": req_id,
        "messageType": "InjectParameterDataRequest",
        "data": {
            "faceFound": face_found,
            "mode": "set",
            "parameterValues": [
                {"id": param_id, "value": mouth_value}
            ]
        }
    }
    self.send_message(payload)
```

---

## 🔗 참고 자료
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [VTubeStudio](https://github.com/DenchiSoft/VTubeStudio)

---

## 📜 라이선스
이 프로젝트는 오픈소스로 제공됩니다. 사용 시 라이선스를 참고하세요.

---

## 📞 문의
궁금한 점이나 개선 사항이 있다면 이슈를 남겨주세요! 😊
