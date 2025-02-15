# VTubeStudio-Streaming-Voice-ChatAI

🎤 **VTubeStudio-Streaming-Voice-ChatAI**: 실시간 AI 대화 + TTS 음성 합성 + VTubeStudio 연동

Ollama 기반 AI와 GPT-SoVITS TTS를 활용한 **스트리밍 및 VTubeStudio 연동 음성 채팅 시스템**

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
Ollama에서 사용할 모델을 다운로드한 후, `DEMO_test.py` 파일의 **Ollama 모델이름**을 다운로드한 모델의 이름으로 변경합니다. 모델 이름은 아래 명령어로 확인할 수 있습니다.
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

### 2️⃣ GPT-SoVITS 세팅
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

## 🎥 VTubeStudio 설정
VTubeStudio 설정에서 **VTubeStudio API 시작 옵션을 ON**으로 활성화합니다.

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

### 3️⃣ **VTubeStudio 연동**
VTubeStudio 실행 후 **권한 요청을 허용**합니다.

---

## ⚠️ 문제 해결
만약 **VTubeStudio 모델이 움직이지 않는다면**, `DEMO_vts_api_helper.py`의 `param_id`가 실제 ID와 일치하는지 확인하세요.

```python
def inject_mouth_value(self, mouth_value: float, face_found: bool = True, param_id: str = "MouthOpen"):  # param_id: str = "MouthOpen"의 실제 ID 확인
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

## 📜 라이선스
이 프로젝트는 오픈소스로 제공됩니다. 사용 시 라이선스를 참고하세요.

---

## 📞 문의
궁금한 점이나 개선 사항이 있다면 이슈를 남겨주세요! 😊
