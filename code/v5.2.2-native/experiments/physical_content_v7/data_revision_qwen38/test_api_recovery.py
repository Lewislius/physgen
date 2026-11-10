from pathlib import Path
import http.client
import io
import json
import sys
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0,str(Path(__file__).resolve().parent))
from api import Client,APIRequestError,APIAnswerError
from annotate import input_reasoning


class Recovery(unittest.TestCase):
    def client(self):
        client=Client.__new__(Client)
        client.config=dict(model='test',base_url='https://example.invalid/v1',api_key='test-only')
        return client

    def response(self):
        return io.BytesIO(json.dumps(dict(id='test',model='test',usage={},choices=[dict(
            finish_reason='stop',message=dict(content='{"relations":["A valid relation."]}'))])).encode())

    def test_disconnected_request_is_retried(self):
        with patch('api.urllib.request.urlopen',side_effect=[http.client.RemoteDisconnected('closed'),self.response()]) as call,patch('api.time.sleep'):
            value,trace=self.client().call('test',[])
        self.assertEqual(call.call_count,2)
        self.assertEqual(value['relations'],['A valid relation.'])

    def test_403_preserves_evidence_and_is_not_blindly_retried(self):
        error=urllib.error.HTTPError('https://example.invalid/v1',403,'Forbidden',
            {'content-type':'text/plain','x-request-id':'diagnostic-id','set-cookie':'irrelevant'},io.BytesIO(b'Forbidden'))
        with patch('api.urllib.request.urlopen',side_effect=error) as call:
            with self.assertRaises(APIRequestError) as caught:self.client().call('test',[])
        self.assertEqual(call.call_count,1)
        self.assertEqual(caught.exception.detail['headers']['x-request-id'],'diagnostic-id')
        self.assertNotIn('set-cookie',caught.exception.detail['headers'])

    def test_invalid_relation_answer_is_preserved_for_repair(self):
        class Bad:
            def call(self,*args,**kwargs):return dict(relations=[]),dict(raw='{"relations":[]}',request_id='bad-example')
        with self.assertRaises(APIAnswerError) as caught:input_reasoning(object(),'An actual request.',3,Bad())
        self.assertEqual(caught.exception.detail['answer_trace']['request_id'],'bad-example')


if __name__=='__main__':unittest.main()
