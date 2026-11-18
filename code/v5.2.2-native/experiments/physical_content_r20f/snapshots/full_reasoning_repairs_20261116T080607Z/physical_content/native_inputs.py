"""Compatibility for the existing input-only I2V native word encodings."""
import html
import re
import ftfy


def check_i2v_native(value, evidence, words):
    # This legacy scope name refers to the INPUT first image/request reasoning,
    # not to a future-video target. Keep its actual stored provenance.
    if (value.get('scope') != 'first_image_request_reasoning_only'
            or value.get('sample') != evidence['sample']
            or value.get('teacher') != 'official frozen T5 then official Wan text_embedding'):
        raise ValueError('Native I2V condition has different input provenance')
    def clean(text):
        return re.sub(r'\s+', ' ', html.unescape(html.unescape(ftfy.fix_text(text)))).strip()
    if [clean(s) for s in value['word_strings']] != [clean(s) for s in words['word_strings']]:
        raise ValueError('Native I2V words differ from the input reasoning')
    if tuple(value['native_words'].shape) != (len(words['word_strings']), 3072):
        raise ValueError('Native I2V condition has an invalid shape')
    return value['native_words']
