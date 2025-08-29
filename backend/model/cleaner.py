from bs4 import BeautifulSoup
import re
from underthesea import word_tokenize

class Cleaner:
    def __init__(self):
        pass

    def __call__(self, text: str | list[str]):
        if isinstance(text, str):
            return self._clean_text(text)
        return [self._clean_text(t) for t in text]

    def _clean_text(self, text):
        text = text.lower()
        text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
        text = re.sub(r"[‘’`]", "'", text)
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[–—−]", "-", text)
        text = re.sub(r'(.)\1+', r'\1', text)
        emoji_groups = {
            "<happy>": ["🙂", "😊", "😄", "😁", "😃", ":)", ":-)", ":d", ":-d", ":p", ":v", "=)", "colonsmile", "colonsmilesmile", ":3", "coloncontemn", "colonbigsmile", "colonsmallsmile", "colonhihi"],
            "<love>":  ["😍", "❤️", "💕", "💖", "<3", "colonlove", ":\">", "colonlovelove"],
            "<sad>":   ["😢", "😭", "☹️", "🙁", ":(", ":-(", "=(", "colonsad", ":_", "coloncc", ":'(", "colonsadcolon",],
            "<angry>": ["😡", "🤬", "😠"],
            "<surprise>": ["😲", "😮", "😯", "😳", "colonsurprise", ":@", "colondoublesurprise"],
            "<thinking>": ["🤔"],
            "<neutral>": ["😐", "😑", "coloncolon"]
        }
        for key, icons in emoji_groups.items():
            for icon in icons:
                text = text.replace(icon, f' {key} ')
        text = text.replace('doubledot', ':')
        text = text.replace('vdotv', '.')
        text = text.replace('v.v', '.')
        text = text.replace('dotdotdot', '.')
        text = text.replace('fraction', '/')
        text = re.sub(r"[^a-zA-Z0-9À-ỹ\s.,;:-<>/()!?'\"-]", " ", text)
        for punc in '.,;:\'"()!-?/<>':
            text = text.replace(punc, f' {punc} ')
        text = word_tokenize(text, format='text')
        text = re.sub(r"<[_\s]happy[_\s]>", "<happy>", text)
        text = re.sub(r"<[_\s]love[_\s]>", "<love>", text)
        text = re.sub(r"<[_\s]sad[_\s]>", "<sad>", text)
        text = re.sub(r"<[_\s]angry[_\s]>", "<angry>", text)
        text = re.sub(r"<[_\s]surprise[_\s]>", "<surprise>", text)
        text = re.sub(r"<[_\s]thinking[_\s]>", "<thinking>", text)
        text = re.sub(r"<[_\s]neutral[_\s]>", "<neutral>", text)
        text = re.sub('\s+', ' ', text).strip()
        return text
