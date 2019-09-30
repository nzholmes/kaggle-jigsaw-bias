import re
import string
import json
from tqdm import tqdm
from unicodedata import category, normalize

# read mappings of irregular text mapping
from typing import List

with open("./preprocessing_dictionary.json", 'r') as f:
    preprocessing_dictionary = json.load(f)

spaces = preprocessing_dictionary['spaces']
contraction_mapping = preprocessing_dictionary['contraction_mapping']
mispell_dict = preprocessing_dictionary['mispell_dict']
special_punc_mappings = preprocessing_dictionary['special_punc_mappings']
empty_punc = preprocessing_dictionary['empty_punc']
rare_words_mapping = preprocessing_dictionary['rare_words_mapping']
extra_punct = preprocessing_dictionary['extra_punct']
bad_case_words = preprocessing_dictionary['bad_case_words']

# read embedding vocabulary
with open('../input/glove_vocab.json', 'r') as f:
    glove_embedding = json.load(f)

with open('../input/crawl_vocab.json', 'r') as f:
    crawl_embedding = json.load(f)


def preprocess_final(text_list):
    """
    Text preprocessing consists three steps.
    The first is regular text normalization for punctuations, spaces, numbers,
    contractions, misspelled and rare words.
    The second deals with '/' and '.' in url and numbers.
    The third deals with '-' and '.' in url and numbers on the basis of the second step.
    Note that the second and third step takes into account embedding words and try to
    keep words in embedding as a whole.

    :param text_list: list of unpreprocessed text
    :return: list of preprocessed text
    """
    new_txt = map(text_normalize, text_list)  # type: List[str]
    new_txt = _process_url_slash_period(new_txt) # type: List[str]
    new_txt = _process_url_num(new_txt) # type: List[str]
    return new_txt

def text_normalize(text):
    """Steps to preprocess irregular text. The order matters."""
    text = remove_space(text)
    text = " {} ".format(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = clean_rare_words(text)
    text = decontracted(text)
    text = correct_contraction(text, contraction_mapping)
    text = clean_misspell(text)
    text = spacing_punctuation(text)
    text = clean_bad_case_words(text)
    text = clean_repeat_words(text)
    text = remove_space(text)
    return text

def remove_space(text):
    """Remove extra spaces and ending space if any"""
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

def remove_diacritics(s):
    """Replace strange punctuations and raplace diacritics"""
    return ''.join(
        c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
        if category(c) != 'Mn')

def decontracted(text):
    """Correct general contraction"""
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)
    text = re.sub(r"(I|i)(\'|\’)m ", "I am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)
    return text

def correct_contraction(text, dic):
    """Correct contraction in mapping of contracted words"""
    for word in dic:
        if word in text:
            text = text.replace(word, dic[word])
        elif word.capitalize() in text:
            text = text.replace(word.capitalize(), dic[word].capitalize())
    return text

def clean_special_punctuations(text):
    """Clean special punctuations"""
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    # remove_diacritics don´t' ->  'don t'
    # text = remove_diacritics(text)
    return text

def clean_number(text):
    """Clean number such as 1 st->1st, 1,000->1000"""
    if re.search(r'(\d+)(th|st|nd|rd)', text) is not None:
        text = re.sub(r'(\d+)(th|st|nd|rd)', '\g<1>\g<2>', text)
    else:
        # text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
        pass
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'\s+?(\d+)(e)(\d+)\s+?', '\g<1> \g<3>', text)
    return text

def clean_rare_words(text):
    """Clean rare words in rare_words_mapping"""
    for rare_word in rare_words_mapping:
        if rare_word in text:
            text = text.replace(rare_word, rare_words_mapping[rare_word])
    return text

def clean_misspell(text):
    """Clean misspelled words in misspelled dict"""
    for bad_word in mispell_dict:
        if bad_word in text:
            text = text.replace(bad_word, mispell_dict[bad_word])
    p=r'(been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|\
    their|has|have|the|be|that|not|was|he|just|they|who)(how)'
    text = re.sub(p, '\g<1> \g<2>', text)
    return text

def clean_bad_case_words(text):
    """Clean bad-case words in the mapping of bad_case_words"""
    for bad_word in bad_case_words:
        if bad_word in text:
            text = text.replace(bad_word, bad_case_words[bad_word])
    return text

def spacing_punctuation(text):
    """
    For punctuations in url, first replace url with special str,
    then replace punctuation in nornal text and finally replace special str with url
    """
    regular_punct = list(string.punctuation)
    all_punct = list(set(regular_punct + extra_punct))
    # do not spacing - and . in the case of "3.2" and "all-mighty"
    all_punct.remove('-')
    all_punct.remove('.')

    # replace url
    text, replace_urls = replace_url(text)

    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    # replace url back
    for r in replace_urls:
        text = text.replace(*r)
    # deal with .
    if '.' in text:
        text = re.sub(r'(\S+?)(\.\s+?)', '\g<1> \g<2>', text)
    return text

def replace_url(text):
    """Replace url with REPLACE"""
    match_urls = find_url(text)
    replace_urls = []
    for i, url in enumerate(match_urls):
        r = ['REPLACE{}'.format(i), url]
        text = text.replace(*r[::-1])
        replace_urls.append(r)
    return text, replace_urls

def find_url(text):
    """Find urls in text"""
    url_pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]\
    |[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\)))'
    # url_pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+\
    # |\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    # url_pattern=r'((?<=[^a-zA-Z0-9])\(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)\(?:\w{1,}\.{1}){1,5}\
    # (?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)'
    match_url = re.findall(url_pattern, text)
    match_url = [j for i in match_url for j in i if j]
    return match_url

def spacing_some_connect_words(text):
    """'Whyare' -> 'Why are'"""
    mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']
    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

    mis_spell_mapping = {'whattsup': 'WhatsApp', 'whatasapp': 'WhatsApp', 'whatsupp': 'WhatsApp',
                         'whatcus': 'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat': 'what',
                         'Whwhat': 'What', 'whatshapp': 'WhatsApp', 'howhat': 'how that',
                         'Whybis': 'Why is', 'laowhy86': 'Foreigners who do not respect China',
                         'Whyco-education': 'Why co-education',
                         "Howddo": "How do", 'Howeber': 'However', 'Showh': 'Show',
                         "Willowmagic": 'Willow magic', 'WillsEye': 'Will Eye', 'Williby': 'will by',
                         'pretextt': 'pre text', 'aɴᴅ': 'and', 'amette': 'annette', 'aᴛ': 'at',
                         'Tridentinus': 'mushroom',
                         'dailycaller': 'daily caller'}
    for error in mis_spell_mapping:
        if error in text:
            text = text.replace(error, mis_spell_mapping[error])

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')
    text = remove_space(text)
    return text

def clean_repeat_words(text):
    """Clean repeated words such as iiing->ing, ---->-"""
    # this one is causing few issues(fixed via monkey patching in other dicts for now), need to check it..
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"(\.{2,}|ᴵᴵ{1,})", " ", text)
    return text

def in_embedding(word):
    return (word in glove_embedding or word.capitalize() in glove_embedding or word.lower() in glove_embedding) or \
        (word in crawl_embedding or word.lower() in crawl_embedding)

def _process_url_slash_period(text_list):
    """Process '/' and '.' in url and number"""
    all_sent = list()
    sentences = (t.split() for t in text_list)

    replace_slash = re.compile(r'\b/\b')
    replace_multi = re.compile(r'[?.&+=/#]')

    for s in tqdm(sentences):
        new_sent = list()
        for w in s:
            if in_embedding(w):
                new_sent.append(w)

            elif find_url(w):
                # deal with url
                # split https://en.wikipedia.org/wiki/Riparian-buffer-> https://en.wikipedia.org, wiki, Riparian-buffer
                # ? . & + = /
                w_split = replace_slash.sub(' ', w).split()
                w_split[0] = "URL_" + w_split[0]
                if 'URL_' not in w_split[-1]:
                    w_split = w_split[:-1] + replace_multi.sub(' ', w_split[-1]).split()
                new_sent += w_split
                continue

            elif "." in w:
                # deal with cases such "lose."->"lose ."
                if re.search(r'(\d+)\.(\d+)', w) is not None:
                    new_sent.append("NUM_" + w)
                    continue
                w = w.replace(".", " . ")
                w_split = [i for i in w.split() if i.strip() and i != "."]
                if any([in_embedding(i) for i in w_split]):
                    new_sent += w_split
                else:
                    new_sent.append(w)
            elif "-" in w:
                if in_embedding(w.replace("-", "")):
                    new_sent.append(w.replace("-", ""))
                    continue
                w1 = w.replace("-", " - ")
                w_split = [i for i in w1.split() if i.strip() and i != "-"]
                if any([in_embedding(i) for i in w_split]):
                    new_sent += w_split
                else:
                    new_sent.append(w)
            else:
                new_sent.append(w)

        new_sent = " ".join(new_sent)
        all_sent.append(new_sent)
    return all_sent


def _process_url_num(text_list):
    """Process '-' and '.' in url and number"""
    all_sent = []

    for s in tqdm(text_list):
        new_sent = []
        for t in s.split():
            if "-" in t and (not t.startswith("URL_")) and (not t.startswith('NUM_')) and (not in_embedding(t)):
                if in_embedding(t.replace("-", '')):
                    new_sent.append(t.replace("-", ''))
                else:
                    new_sent += t.split("-")
            elif "." in t and (not t.startswith("URL_")) and (not t.startswith('NUM_')) \
                    and re.search(r'(\d+)\.(\d+)', t) is None:
                if all([in_embedding(i.strip()) for i in t.split(".")]):
                    new_sent.append(t.replace(".", " "))
                else:
                    new_sent.append(t)
            else:
                new_sent.append(t)

        new_sent = " ".join(new_sent)
        all_sent.append(new_sent)
    return all_sent