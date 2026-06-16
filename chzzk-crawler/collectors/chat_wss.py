import json
import logging
import threading
import time
from datetime import datetime
import requests
import websocket

from collectors.base import BaseChatCollector, ChatEvent

logger = logging.getLogger(__name__)

CHZZK_CHAT_CMD = {
    'ping'                : 0,
    'pong'                : 10000,
    'connect'             : 100,
    'connect_ack'         : 10100,
    'request_recent_chat' : 5101,
    'chat'                : 93101,
    'donation'            : 93102,
}

class WssChatCollector(BaseChatCollector):
    def __init__(self, broad_no: str, user_id: str, sink):
        super().__init__(broad_no, user_id, sink)
        self.chat_channel_id = None
        self.access_token = None
        self.sid = None
        self.ws = None
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def _fetch_chat_info(self):
        # 1. Get chatChannelId
        url_detail = f'https://api.chzzk.naver.com/service/v2/channels/{self.user_id}/live-detail'
        resp = requests.get(url_detail, headers=self._headers, timeout=10)
        resp.raise_for_status()
        self.chat_channel_id = resp.json()['content']['chatChannelId']

        # 2. Get Access Token
        url_token = f'https://comm-api.game.naver.com/nng_main/v1/chats/access-token?channelId={self.chat_channel_id}&chatType=STREAMING'
        resp2 = requests.get(url_token, headers=self._headers, timeout=10)
        resp2.raise_for_status()
        token_data = resp2.json()['content']
        self.access_token = token_data['accessToken']

    def _on_ws_open(self, ws):
        logger.info(f"[{self.broad_no}] Connected to chat wss")
        # Send Connect Cmd
        msg = {
            "ver": "2",
            "svcid": "game",
            "cid": self.chat_channel_id,
            "cmd": CHZZK_CHAT_CMD['connect'],
            "tid": 1,
            "bdy": {
                "uid": None,
                "devType": 2001,
                "accTkn": self.access_token,
                "auth": "READ"
            }
        }
        ws.send(json.dumps(msg))

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            cmd = data.get('cmd')

            if cmd == CHZZK_CHAT_CMD['ping']:
                pong_msg = {
                    "ver": "2",
                    "cmd": CHZZK_CHAT_CMD['pong']
                }
                ws.send(json.dumps(pong_msg))
                return

            if cmd in (CHZZK_CHAT_CMD['connect'], CHZZK_CHAT_CMD['connect_ack']):
                self.sid = data.get('bdy', {}).get('sid')
                # Request recent chat after connected
                req_msg = {
                    "ver": "2",
                    "svcid": "game",
                    "cid": self.chat_channel_id,
                    "cmd": CHZZK_CHAT_CMD['request_recent_chat'],
                    "tid": 2,
                    "sid": self.sid,
                    "bdy": {
                        "recentMessageCount": 50
                    }
                }
                ws.send(json.dumps(req_msg))
                return

            if cmd == CHZZK_CHAT_CMD['chat'] or cmd == CHZZK_CHAT_CMD['donation']:
                for chat_data in data.get('bdy', []):
                    uid = chat_data.get('uid')
                    msg_text = chat_data.get('msg', '')
                    msg_time = chat_data.get('msgTime', int(datetime.now().timestamp()*1000))
                    
                    nickname = "Anonymous"
                    if uid != 'anonymous' and chat_data.get('profile'):
                        try:
                            profile = json.loads(chat_data['profile'])
                            nickname = profile.get('nickname', 'Unknown')
                        except:
                            pass

                    event_ts = datetime.fromtimestamp(msg_time / 1000.0)
                    
                    self.sink.push(ChatEvent(
                        event_ts=event_ts,
                        broad_no=self.broad_no,
                        user_id=uid,
                        user_nick=nickname,
                        message_raw=msg_text,
                        raw_json=chat_data
                    ))
        except Exception as e:
            logger.error(f"[{self.broad_no}] Error parsing WS message: {e}")

    def _on_ws_error(self, ws, error):
        logger.error(f"[{self.broad_no}] WS Error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        logger.info(f"[{self.broad_no}] WS Closed: {close_status_code} - {close_msg}")

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._fetch_chat_info()
                
                # Setup websocket
                self.ws = websocket.WebSocketApp(
                    "wss://kr-ss1.chat.naver.com/chat",
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close
                )

                # Run forever blocking call
                while not self.stop_event.is_set():
                    # Check connection alive. Run forever handles polling.
                    # We will run this inner loop but pass dispatch.
                    # websocket.WebSocketApp has an internal loop, we shouldn't wrap it tightly in while.
                    # We must run it outside.
                    break
                    
                # The blocking call:
                self.ws.run_forever(ping_interval=20, ping_timeout=10)

                # If connection closed and we are not stopping, it will retry
                if not self.stop_event.is_set():
                    time.sleep(2) # Retry delay

            except requests.exceptions.RequestException as e:
                logger.error(f"[{self.broad_no}] Failed to fetch chat API info: {e}")
                time.sleep(5)
            except Exception as e:
                logger.exception(f"[{self.broad_no}] Unexpected error in WssChatCollector")
                time.sleep(5)

    def stop(self) -> None:
        super().stop()
        if self.ws:
            self.ws.close()
