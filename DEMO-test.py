import concurrent.futures
import threading
import queue
import re
import logging
import copy
import requests
import struct
import pyaudio
import time
import math
import numpy as np
import random
import json
import uuid
from websocket import create_connection, WebSocketTimeoutException

# 외부/사용자 라이브러리
from RealtimeSTT import AudioToTextRecorder
from ollama import chat

# VTS API 클래스는 별도 파일로 분리됨
from DEMO_vts_api_helper import VTubeStudioAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

DEFAULT_REF_AUDIO = r"C:\Users\unit6\Documents\Test\My_tts\My_BaseTTS_v2.wav"
DEFAULT_PROMPT_TEXT = ""
DEFAULT_PROMPT_LANG = "ko"
DEFAULT_PROMPT_FOR_NO_INPUT = "시청자들과 간단한 짧은토크를 진행해줘."
SILENCE_THRESHOLD = 15.0  # 입력 없으면 15초 후 기본 프롬프트 실행
TICK_INTERVAL = 1.0         # 모니터링 주기 (1초)
ACTIVITY_BLOCK_TIME = 5.0   # 최근 활동 후 몇 초 동안 silence 카운트 중지

# 전역 변수: 녹음 중 여부 및 녹음 종료 시각
recording_lock = threading.Lock()
recording_in_progress = False
recording_stop_time = None

# TTS 작업이 진행 중임을 나타내는 플래그 (TTS 합성 및 재생 둘 다 고려)
tts_active_flag = threading.Event()

# 최근 활동(사용자 입력, TTS 작업 큐 추가 등)이 있었던 시각 (초 단위)
last_activity_time = time.time()


########################################################################
# silence_monitor: 모든 대기열과 TTS 재생/합성 진행 여부를 확인하여 기본 프롬프트 삽입
########################################################################
def silence_monitor(recognized_queue: queue.Queue, tts_queue: queue.Queue, playback_queue: queue.Queue,
                    default_prompt: str, silence_threshold: float, tick_interval: float,
                    stop_event: threading.Event, tts_play_lock: threading.Lock):
    """
    - 녹음 중 또는 녹음 종료 후 3초 이내, 혹은 TTS 재생/합성 작업이 진행 중이면 카운터를 리셋합니다.
    - 그 외에는 반드시 모든 대기열(인식, TTS 합성, 재생)이 비어있고, TTS 작업도 진행 중이지 않은 경우에만
      누적 시간을 증가시킵니다.
    - 단, 최근 활동(ACTIVITY_BLOCK_TIME 이내)이 있었으면 누적을 진행하지 않습니다.
    - 누적 시간이 threshold 이상이면 기본 프롬프트를 삽입합니다.
    """
    global recording_in_progress, recording_stop_time, last_activity_time
    silence_elapsed = 0.0
    while not stop_event.is_set():
        # 녹음 상태 확인
        with recording_lock:
            is_recording = recording_in_progress
            rec_stop_time = recording_stop_time

        # 녹음 중이거나 녹음 종료 후 3초 이내이면 누적 카운터 리셋
        if is_recording or (rec_stop_time is not None and (time.time() - rec_stop_time < 3)):
            silence_elapsed = 0.0
            time.sleep(tick_interval)
            continue

        # TTS 재생 중이거나 TTS 합성 작업 진행 중이면 누적 리셋
        if tts_play_lock.locked() or tts_active_flag.is_set():
            silence_elapsed = 0.0
            time.sleep(tick_interval)
            continue

        # 최근 활동이 ACTIVITY_BLOCK_TIME 이내이면 카운트 진행하지 않음
        if time.time() - last_activity_time < ACTIVITY_BLOCK_TIME:
            silence_elapsed = 0.0
            time.sleep(tick_interval)
            continue

        # 반드시 모든 대기열(인식, TTS 합성, 재생)이 비어있어야 누적 카운트 진행
        if recognized_queue.empty() and tts_queue.empty() and playback_queue.empty():
            silence_elapsed += tick_interval
            logging.info("[Silence Monitor] 입력 없음: 누적 %.1f초", silence_elapsed)
            if silence_elapsed >= silence_threshold:
                logging.info("[Silence Monitor] 입력 없음 %.1f초 경과 -> 기본 프롬프트 삽입", silence_elapsed)
                recognized_queue.put(default_prompt)
                silence_elapsed = 0.0
        else:
            # 하나라도 데이터가 있다면 즉시 카운터 리셋
            silence_elapsed = 0.0

        time.sleep(tick_interval)


########################################################################
# OllamaChat 클래스: 챗봇 API 호출 및 대화 내역 관리
########################################################################
class OllamaChat:
    def __init__(self, model: str = "benedict/linkbricks-llama3.1-korean:8b") -> None:
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

    def get_response(self, conversation_history: list) -> str:
        try:
            full_history = [self.system_message] + conversation_history
            response = chat(model=self.model, messages=full_history)
            return response.get('message', {}).get('content', "응답을 받지 못했습니다.")
        except Exception as e:
            logging.error("[Ollama] 모델 응답 에러: %s", e)
            return "모델 응답을 받지 못했습니다."

    def stream_response(self, conversation_history: list):
        """
        스트리밍 응답 제너레이터.
        Ollama API의 스트리밍 모드를 사용해 토큰(부분 결과)을 순차적으로 yield한다고 가정.
        """
        try:
            full_history = [self.system_message] + conversation_history
            stream = chat(model=self.model, messages=full_history, stream=True)
            for token in stream:
                yield token.get('message', {}).get('content', '')
        except Exception as e:
            logging.error("[Ollama] 스트리밍 응답 에러: %s", e)
            yield ""
            
    def add_user_message(self, message: str) -> None:
        self.conversation_history.append({'role': 'user', 'content': message})
        self.trim_history()

    def add_assistant_message(self, message: str) -> None:
        self.conversation_history.append({'role': 'assistant', 'content': message})
        logging.info("[Ollama] 모델 응답: %s", message)
        self.trim_history()

    def trim_history(self, max_messages: int = 12) -> None:
        non_system_msgs = [msg for msg in self.conversation_history if msg['role'] != 'system']
        if len(non_system_msgs) > max_messages:
            non_system_msgs = non_system_msgs[-max_messages:]
            self.conversation_history = [self.system_message] + non_system_msgs

def filter_response(response_text: str) -> str:
    if not response_text:
        return response_text
    return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

########################################################################
# TTS 합성 함수: 텍스트를 TTS API를 통해 음성 데이터(바이트)로 합성
########################################################################
def synthesize_tts_audio(text: str, vts_api: VTubeStudioAPI) -> tuple:
    base_url = "http://127.0.0.1:9880/tts"
    params = {
        "text": text,
        "text_lang": "ko",
        "ref_audio_path": DEFAULT_REF_AUDIO,
        "prompt_text": DEFAULT_PROMPT_TEXT,
        "prompt_lang": DEFAULT_PROMPT_LANG,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": "true"  # streaming_mode는 그대로 사용
    }
    logging.info("[TTS 합성] 요청 텍스트: %s", text)
    response = requests.get(base_url, params=params, stream=True)
    response.raise_for_status()

    # WAV 헤더 읽기 (44바이트)
    header = response.raw.read(44)
    if len(header) < 44:
        raise ValueError("WAV 헤더 길이가 44바이트 미만입니다.")
    riff, size, wave_, fmt = struct.unpack('<4sI4s4s', header[:16])
    if riff != b'RIFF' or wave_ != b'WAVE':
        raise ValueError("유효한 WAV 파일이 아닙니다.(RIFF/WAVE 오류)")
    audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack('<HHIIHH', header[20:36])
    data_chunk_id = header[36:40]
    if data_chunk_id != b'data':
        raise ValueError("data 청크가 없습니다.")
    data_size = struct.unpack('<I', header[40:44])[0]
    logging.info("[TTS 합성] sample_rate=%d, channels=%d, bits=%d, data_size=%d",
                 sample_rate, num_channels, bits_per_sample, data_size)
    # data_size 만큼 읽거나 남은 데이터를 모두 읽기
    audio_bytes = response.raw.read(data_size) if data_size > 0 else response.raw.read()
    return (audio_bytes, sample_rate, num_channels, bits_per_sample)

########################################################################
# TTS 재생 함수: 합성된 음성 데이터를 PyAudio로 재생
########################################################################
def play_tts_audio(audio_bytes: bytes, sample_rate: int, channels: int, bits_per_sample: int,
                   vts_api: VTubeStudioAPI, tts_stop_event: threading.Event):
    p = pyaudio.PyAudio()
    if bits_per_sample == 8:
        pa_format = pyaudio.paInt8
        sample_width = 1
        dtype = np.int8
    elif bits_per_sample == 16:
        pa_format = pyaudio.paInt16
        sample_width = 2
        dtype = np.int16
    elif bits_per_sample == 24:
        pa_format = pyaudio.paInt24
        sample_width = 3
        # 일부 환경에서는 별도 처리가 필요할 수 있음
        dtype = np.int32  
    elif bits_per_sample == 32:
        pa_format = pyaudio.paInt32
        sample_width = 4
        dtype = np.int32
    else:
        raise ValueError(f"지원되지 않는 샘플 폭: {bits_per_sample}")
    stream_out = p.open(format=pa_format, channels=channels, rate=sample_rate, output=True)
    chunk_size = 1024
    last_inject_time = time.time()
    inject_interval = 0.05
    offset = 0
    while offset < len(audio_bytes) and not tts_stop_event.is_set():
        chunk = audio_bytes[offset:offset+chunk_size]
        offset += len(chunk)
        stream_out.write(chunk)
        if vts_api and vts_api.authenticated:
            now = time.time()
            if now - last_inject_time >= inject_interval:
                if dtype is not None and len(chunk) >= sample_width:
                    data_array = np.frombuffer(chunk, dtype=dtype)
                    if channels > 1:
                        data_array = data_array.reshape(-1, channels).mean(axis=1)
                    rms = math.sqrt(np.mean(data_array.astype(np.float32) ** 2))
                    mouth_value = min(rms / 30000.0, 1.0)
                else:
                    mouth_value = 0.0
                vts_api.inject_mouth_value(mouth_value, face_found=True, param_id="MouthOpen")
                last_inject_time = now
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    if vts_api and vts_api.authenticated:
        vts_api.inject_mouth_value(0.0, face_found=True, param_id="MouthOpen")

########################################################################
# TTS 합성 워커: tts_queue의 텍스트 청크를 합성하여 playback_queue에 추가
########################################################################
def tts_synthesis_worker(tts_queue: queue.Queue, tts_executor: concurrent.futures.ThreadPoolExecutor,
                         playback_queue: queue.Queue, vts_api: VTubeStudioAPI):
    while True:
        try:
            text_chunk = tts_queue.get(timeout=1)
        except queue.Empty:
            continue
        # TTS 합성 작업 시작 플래그 설정
        tts_active_flag.set()
        future = tts_executor.submit(synthesize_tts_audio, text_chunk, vts_api)
        try:
            audio_result = future.result()
            playback_queue.put(audio_result)
            logging.info("[TTS 합성 완료] %s", text_chunk)
        except Exception as e:
            logging.error("TTS 합성 작업 오류: %s", e)
        finally:
            # TTS 합성 작업 완료 후 플래그 해제
            tts_active_flag.clear()
        tts_queue.task_done()

########################################################################
# TTS 재생 워커: playback_queue에 쌓인 오디오 데이터를 순서대로 재생
########################################################################
def tts_playback_worker(playback_queue: queue.Queue, tts_play_lock: threading.Lock,
                        vts_api: VTubeStudioAPI):
    while True:
        try:
            audio_result = playback_queue.get(timeout=1)
        except queue.Empty:
            continue
        with tts_play_lock:
            tts_stop_event = threading.Event()
            play_tts_audio(audio_result[0], audio_result[1], audio_result[2], audio_result[3], vts_api, tts_stop_event)
            logging.info("[TTS 재생 완료] 재생된 오디오 길이: %d bytes", len(audio_result[0]))
        playback_queue.task_done()

########################################################################
# STT 쓰레드 함수: 음성을 텍스트로 인식하여 recognized_queue에 추가
########################################################################
def continuous_stt_thread(recorder: AudioToTextRecorder,
                            recognized_queue: queue.Queue,
                            stop_event: threading.Event,
                            stt_pause_event: threading.Event) -> None:
    global recording_in_progress, recording_stop_time, last_activity_time
    with recorder:
        logging.info("[STT] 지속형 스트리밍 시작")
        while not stop_event.is_set():
            if stt_pause_event.is_set():
                time.sleep(0.1)
                continue
            try:
                text = recorder.text()
                if text and text.strip():
                    recognized_text = text.strip()
                    # 녹음 시작/종료 감지
                    if "recording started" in recognized_text.lower():
                        with recording_lock:
                            recording_in_progress = True
                        logging.info("[STT] 녹음 시작 감지")
                    elif "recording stopped" in recognized_text.lower():
                        with recording_lock:
                            recording_in_progress = False
                            recording_stop_time = time.time()
                        logging.info("[STT] 녹음 종료 감지")
                    logging.info("[STT] 인식됨: %s", recognized_text)
                    print(f"[입력 인식됨] {recognized_text}")
                    recognized_queue.put(recognized_text)
                    # 사용자 입력이 있었으므로 활동 시각 갱신
                    last_activity_time = time.time()
                else:
                    time.sleep(0.1)
            except Exception as e:
                logging.error("[STT] 음성 인식 오류: %s", e)
                time.sleep(0.5)
        logging.info("[STT] 지속형 스트리밍 종료")

########################################################################
# 눈 깜빡임 쓰레드 함수: 주기적으로 눈 파라미터 업데이트
########################################################################
def blink_thread_func(vts_api: VTubeStudioAPI, stop_event: threading.Event):
    while not stop_event.is_set():
        wait_time = random.uniform(3.0, 6.0)
        time.sleep(wait_time)
        if stop_event.is_set():
            break
        vts_api.inject_eye_blink(0.0, 0.0)
        time.sleep(0.1)
        vts_api.inject_eye_blink(1.0, 1.0)

########################################################################
# 메인 함수: STT/TTS 파이프라인 관리 및 VTS, 챗봇 통합
########################################################################
def main() -> None:
    global recording_in_progress, recording_stop_time, last_activity_time
    ollama_session = OllamaChat()
    recognized_queue: queue.Queue = queue.Queue()
    tts_queue: queue.Queue = queue.Queue()        # 텍스트 청크 대기열 (합성용)
    playback_queue: queue.Queue = queue.Queue()     # 합성된 오디오 대기열 (재생용)
    stop_event = threading.Event()
    stt_pause_event = threading.Event()  # TTS 작업 중 STT 일시정지

    # TTS 재생 동시 실행 방지용 락
    tts_play_lock = threading.Lock()
    # TTS 합성을 위한 스레드 풀
    tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    # VTS API 연결/인증
    vts_api = None
    try:
        vts_api = VTubeStudioAPI(
            plugin_name="My Python Plugin",
            plugin_developer="MyName",
            stored_token=None,
            host="localhost",
            port=8001
        )
        logging.info("VTubeStudioAPI 연결/인증 완료.")
    except Exception as e:
        logging.error(f"VTS 플러그인 연결 실패: {e}")

    # STT 준비 및 실행 (TTS 작업 중에는 stt_pause_event에 의해 일시정지됨)
    recorder = AudioToTextRecorder(**{
        "model": "large-v2",
        "language": "ko",
        "device": "cuda",
        "gpu_device_index": 0,
        "beam_size": 5,
        "input_device_index": 0,
        "handle_buffer_overflow": True,
        "ensure_sentence_starting_uppercase": True,
        "ensure_sentence_ends_with_period": True,
        "webrtc_sensitivity": 1,
        "post_speech_silence_duration": 1.0,
        "silero_sensitivity": 0.5,
        "silero_deactivity_detection": True,
        "min_length_of_recording": 1.0,
        "min_gap_between_recordings": 1.0,
        "level": logging.INFO,
        "debug_mode": False,
        "print_transcription_time": True,
        "enable_realtime_transcription": True,
        "use_main_model_for_realtime": True,
        "realtime_model_type": "large-v2",
        "realtime_processing_pause": 0.2,
    })
    stt_thread = threading.Thread(
        target=continuous_stt_thread,
        args=(recorder, recognized_queue, stop_event, stt_pause_event),
        daemon=True
    )
    stt_thread.start()

    # 눈 깜빡임 쓰레드 시작
    blink_stop_event = threading.Event()
    blink_thread = None
    if vts_api and vts_api.authenticated:
        blink_thread = threading.Thread(
            target=blink_thread_func,
            args=(vts_api, blink_stop_event),
            daemon=True
        )
        blink_thread.start()

    # SILENCE 모니터링 스레드 시작 (모든 대기열 및 TTS 재생/합성 진행 여부 모니터링)
    silence_monitor_thread = threading.Thread(
        target=silence_monitor,
        args=(recognized_queue, tts_queue, playback_queue,
              DEFAULT_PROMPT_FOR_NO_INPUT, SILENCE_THRESHOLD, TICK_INTERVAL,
              stop_event, tts_play_lock),
        daemon=True
    )
    silence_monitor_thread.start()

    # TTS 합성 워커 스레드 시작
    tts_synthesis_thread = threading.Thread(
        target=tts_synthesis_worker,
        args=(tts_queue, tts_executor, playback_queue, vts_api),
        daemon=True
    )
    tts_synthesis_thread.start()

    # TTS 재생 워커 스레드 시작
    tts_playback_thread = threading.Thread(
        target=tts_playback_worker,
        args=(playback_queue, tts_play_lock, vts_api),
        daemon=True
    )
    tts_playback_thread.start()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        logging.info("파이프라인 시작. (Ctrl+C로 종료)\n")
        try:
            while True:
                # 메인 루프에서는 recognized_queue에서 메시지를 꺼내 챗봇 요청 처리
                try:
                    user_text = recognized_queue.get(timeout=TICK_INTERVAL)
                except queue.Empty:
                    continue

                if not user_text.strip():
                    user_text = DEFAULT_PROMPT_FOR_NO_INPUT

                print(f"\n[사용자 입력] {user_text}\n")
                ollama_session.add_user_message(user_text)
                # 사용자 입력 발생 시 활동 시각 갱신
                last_activity_time = time.time()

                # 대화 내역 스냅샷 생성 및 TTS 작업 준비
                history_snapshot = copy.deepcopy(ollama_session.conversation_history)
                stt_pause_event.set()
                with recognized_queue.mutex:
                    recognized_queue.queue.clear()

                full_response = ""
                token_buffer = ""
                for token in ollama_session.stream_response(history_snapshot):
                    full_response += token
                    token_buffer += token
                    # 구두점 기준으로 문장을 분할하여 TTS 작업 큐에 추가
                    if token_buffer.endswith(('.', '!', '?', ',', ';')):
                        tts_queue.put(token_buffer)
                        print(f"[TTS 작업 큐 추가] {token_buffer}")
                        # 사용자 활동으로 간주하여 활동 시각 갱신
                        last_activity_time = time.time()
                        token_buffer = ""
                        time.sleep(0.1)
                if token_buffer:
                    tts_queue.put(token_buffer)
                    print(f"[TTS 작업 큐 추가] {token_buffer}")
                    last_activity_time = time.time()
                final_reply = filter_response(full_response)
                ollama_session.add_assistant_message(final_reply)
                print(f"[Ollama 응답] {final_reply}\n")

                # 입력 후, 모든 대기열(TTS 합성 및 재생 대기열)이 비어질 때까지 대기
                while not (tts_queue.empty() and playback_queue.empty() and not tts_play_lock.locked()):
                    time.sleep(0.1)
                stt_pause_event.clear()
                
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt -> 프로그램 종료 요청")
        finally:
            stop_event.set()
            stt_thread.join()
            tts_executor.shutdown(wait=True)
            if blink_thread:
                blink_stop_event.set()
                blink_thread.join()
            if silence_monitor_thread:
                silence_monitor_thread.join()
            if vts_api:
                vts_api.close()
            logging.info("프로그램 정상 종료.")

if __name__ == "__main__":
    main()
