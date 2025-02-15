import json
import time
import uuid
import logging
import threading
from websocket import create_connection, WebSocketTimeoutException

class VTubeStudioAPI:
    def __init__(
        self, 
        plugin_name: str = "My Python Plugin",
        plugin_developer: str = "My Name",
        stored_token: str = None,
        host: str = "localhost",
        port: int = 8001
    ):
        """
        VTS WebSocket API와 연결하여 플러그인 인증 및 파라미터 주입에 활용하는 헬퍼 클래스.
        """
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.stored_token = stored_token  # 이미 발급받은 토큰(있다면) 저장
        self.host = host
        self.port = port
        self.ws = None
        self.authenticated = False
        self.token = None  # 발급받거나 재인증에 성공한 토큰 저장

        self.logger = logging.getLogger("VTS_API")
        self.logger.setLevel(logging.INFO)

        self.connect()
        if self.stored_token:
            if self.authenticate(self.stored_token):
                self.logger.info("기존 토큰으로 재인증 성공.")
            else:
                self.logger.info("기존 토큰 재인증 실패, 새 토큰 요청 시도...")
                self.request_and_auth_new_token()
        else:
            self.request_and_auth_new_token()

    def connect(self):
        url = f"ws://{self.host}:{self.port}"
        self.logger.info(f"VTS에 연결 시도: {url}")
        self.ws = create_connection(url, timeout=5)
        self.logger.info("VTS WebSocket 연결 성공.")

    def close(self):
        if self.ws:
            self.ws.close()
            self.logger.info("VTS WebSocket 연결 종료.")
            self.ws = None
            self.authenticated = False

    def send_message(self, message: dict, max_retries: int = 1) -> dict:
        if not self.ws:
            raise ConnectionError("WebSocket이 연결되지 않았습니다.")
        try:
            self.ws.send(json.dumps(message))
            response_str = self.ws.recv()
            return json.loads(response_str)
        except (WebSocketTimeoutException, OSError, ConnectionError, ValueError) as e:
            self.logger.error(f"VTS WebSocket 오류: {e}")
            if max_retries > 0:
                self.logger.info("WebSocket 재연결 시도 중...")
                self.close()
                if self.try_reconnect():
                    self.logger.info("재연결/재인증 성공. 메시지 재전송 시도...")
                    return self.send_message(message, max_retries=max_retries - 1)
                else:
                    self.logger.error("재연결/재인증 실패. 메시지 전송 중단.")
                    return {}
            else:
                self.logger.error("재시도 횟수 초과. WebSocket 메시지 전송 실패.")
                return {}
        except Exception as e:
            self.logger.error(f"VTS WebSocket 전송/수신 예외: {e}")
            return {}

    def try_reconnect(self) -> bool:
        try:
            self.connect()
            if self.token:
                if self.authenticate(self.token):
                    return True
                else:
                    self.logger.info("기존 토큰 재인증 실패 -> 새 토큰 발급 시도")
                    return self.request_and_auth_new_token()
            else:
                return self.request_and_auth_new_token()
        except Exception as e:
            self.logger.error(f"재연결 시도 중 오류: {e}")
            return False

    def api_state_request(self) -> dict:
        req_id = str(uuid.uuid4())[:8]
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": req_id,
            "messageType": "APIStateRequest"
        }
        return self.send_message(payload)

    def request_and_auth_new_token(self) -> bool:
        req_id = str(uuid.uuid4())[:8]
        request_token_payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": req_id,
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
            }
        }
        resp = self.send_message(request_token_payload)
        if resp.get("messageType") == "APIError":
            self.logger.error(f"토큰 요청 거부: {resp}")
            return False
        new_token = resp.get("data", {}).get("authenticationToken", None)
        if not new_token:
            self.logger.error(f"토큰 요청 실패(응답에 토큰 없음): {resp}")
            return False
        self.logger.info(f"새 토큰 발급됨: {new_token}")
        return self.authenticate(new_token)

    def authenticate(self, token: str) -> bool:
        req_id = str(uuid.uuid4())[:8]
        auth_req_payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": req_id,
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": token
            }
        }
        resp = self.send_message(auth_req_payload)
        if resp.get("messageType") == "AuthenticationResponse":
            if resp.get("data", {}).get("authenticated", False):
                self.authenticated = True
                self.token = token
                self.logger.info("VTS 플러그인 인증 완료.")
                return True
            else:
                reason = resp.get("data", {}).get("reason", "알 수 없는 이유")
                self.logger.error(f"인증 실패: {reason}")
                return False
        elif resp.get("messageType") == "APIError":
            self.logger.error(f"인증 요청 에러: {resp}")
            return False
        else:
            self.logger.error(f"인증 응답이 이상함: {resp}")
            return False

    def inject_mouth_value(self, mouth_value: float, face_found: bool = True, param_id: str = "MouthOpen"):
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

    def inject_eye_blink(self, left_value: float, right_value: float, face_found: bool = True):
        if not self.authenticated:
            return
        left_value = max(0.0, min(1.0, left_value))
        right_value = max(0.0, min(1.0, right_value))
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
                    {"id": "EyeOpenLeft", "value": left_value},
                    {"id": "EyeOpenRight", "value": right_value}
                ]
            }
        }
        self.send_message(payload)

