"""Text-producing Qwen API; local Qwen3-VL remains the feature encoder."""
from pathlib import Path
import base64
import io
import http.client
import json
import time
import urllib.error
import urllib.request

DEFAULT_CREDENTIALS = Path('/home/liuzhirui/.config/physgen/qwen_api.json')


class APIRequestError(RuntimeError):
    def __init__(self,status,message,headers):
        super().__init__(f'API HTTP {status}: {message[:1000]}')
        wanted={'content-type','server','x-request-id','x-dashscope-request-id','x-acs-request-id','retry-after'}
        self.detail=dict(status=status,message=message[:4000],
            headers={k.lower():v for k,v in headers.items() if k.lower() in wanted})


class APIAnswerError(ValueError):
    def __init__(self,message,trace):
        super().__init__(message)
        self.detail=dict(answer_trace=trace)


def image_url(image):
    stream = io.BytesIO()
    image.convert('RGB').save(stream, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


class Client:
    def __init__(self, credentials=DEFAULT_CREDENTIALS):
        self.config = json.loads(Path(credentials).read_text())

    def call(self, prompt, images, *, video=False, max_tokens=2200, thinking=False, times=None):
        urls = [image_url(im) for im in images]
        visual = ([{'type': 'video', 'video': urls}] if video else
                  [{'type': 'image_url', 'image_url': {'url': u}} for u in urls])
        if times is not None:
            if video or len(times) != len(images):
                raise ValueError('Explicit timestamps require one timestamp per image')
            visual = []
            for index, (stamp, url) in enumerate(zip(times, urls)):
                visual += [dict(type='text', text=f'Chronological frame {index+1}, t={stamp:.6f} seconds:'),
                           dict(type='image_url', image_url={'url': url})]
        body = dict(model=self.config['model'], messages=[dict(role='user', content=visual +
                    [dict(type='text', text=prompt)])], temperature=0.1,
                    max_tokens=max_tokens, enable_thinking=thinking,
                    response_format={'type': 'json_object'})
        request = urllib.request.Request(self.config['base_url'].rstrip('/')+'/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
                    'Authorization': 'Bearer '+self.config['api_key']})
        start = time.monotonic()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                message = error.read().decode(errors='replace').replace(self.config['api_key'], '[removed]')
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise APIRequestError(error.code,message,error.headers) from None
                time.sleep(3*(attempt+1))
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError):
                if attempt == 2:
                    raise RuntimeError('API connection failed after three attempts') from None
                time.sleep(3*(attempt+1))
        choice = result['choices'][0]
        record = dict(model=result.get('model'), request_id=result.get('id'),
                      usage=result.get('usage', {}), finish_reason=choice.get('finish_reason'),
                      raw=choice['message']['content'], seconds=time.monotonic()-start,
                      image_count=len(images), video_sequence=video, thinking=thinking,
                      explicit_times=times, prompt=prompt)
        if choice.get('finish_reason') != 'stop':
            raise APIAnswerError(f'Incomplete API answer: {choice.get("finish_reason")}',record)
        try:value = json.loads(record['raw'])
        except (ValueError,TypeError):
            raise APIAnswerError('API answer is not valid JSON',record) from None
        if not isinstance(value, dict):
            raise APIAnswerError('Expected a JSON object from Qwen',record)
        return value, record
