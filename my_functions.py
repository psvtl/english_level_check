from collections import Counter
import nltk
import os
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
import language_tool_python
import re
import torch
import itertools
my_tool = language_tool_python.LanguageTool('en-US')


def dialog_processing(contents):
    """
    Убирает теги организатора, лишних действий
    """
    dialog = contents[contents.find('<task>')+len('<task>'):contents.find('</task>')]
    to_remove = ['\n', '']
    dialog = dialog.replace('\n', '')
    dialog = " ".join(dialog.split())
    a, ctxt = 0, 0
    len_a = 0
    a_speech = ''
    a = dialog.find('<A>')
    while a > -1 or ctxt > -1:
        ctxt = dialog.find('<ctxt>')
        dialog = dialog if ctxt== -1 else dialog.replace(dialog[ctxt:dialog.find('</ctxt>')+len('</ctxt>')], "")
        a_speech += dialog[a+len('<A>'):dialog.find('</A>')]
        dialog = dialog if a== -1 else dialog.replace(dialog[a:dialog.find('</A>')+len('</A>')], "")
        a = dialog.find('<A>')
    # f, ol, nvs, h,  = 0, 0, 0, 0
    list_of_counters = [0]*4
    tags = ['<F>', '<OL>', '<nvs>', '<H']
    closed_tags = ['</F>', '</OL>','</nvs>','</H>']
    while Counter(list_of_counters)[-1] != len(list_of_counters): 
        for i in range(len(tags)):
            t =  dialog.find(tags[i])
            list_of_counters[i] = t
            dialog = dialog if t== -1 else dialog.replace(dialog[t:dialog.find(closed_tags[i])+len(closed_tags[i])], "")
    # while f > -1 or ol>-1 or h > -1 or nvs > -1:
    #     f = a_speech.find('<F>')
    #     a_speech = a_speech if f== -1 else a_speech.replace(a_speech[f:a_speech.find('</F>')+len('</F>')], "")
    #     ol = a_speech.find('<OL>')
    #     a_speech = a_speech if ol==-1 else a_speech.replace(a_speech[ol:a_speech.find('</OL>')+len('</OL>')], "")
    #     h = a_speech.find('<H')
    #     a_speech = a_speech if h==-1 else a_speech.replace(a_speech[h:a_speech.find('</H>')+len('</H>')], "")
    #     nvs = a_speech.find('<nvs>')
    #     a_speech = a_speech if nvs==-1 else a_speech.replace(a_speech[nvs:a_speech.find('</nvs>')+len('</nvs>')], "")
    len_a = len(a_speech)
    return dialog, len_a


def analyse_dialog(dialog, need_print=False):
    """
    Подсчитывает число тегов в речи
    """
    repetitions = dialog.count('<R>')+dialog.count('<R?>')
    pauses_len = dialog.count('<.>')*2 + dialog.count('<..>')*4
    filled_pauses = dialog.count('<F>')
    self_correction = dialog.count('<SC>') + dialog.count('<SC?>')
    japans = dialog.count('<JP>')
    is_unfinished = dialog.count('<CO>')

    message = f"There are {repetitions} repetitions, \n{pauses_len} seconds of pauses, "\
    f"\n{filled_pauses} seconds of filled pauses, \n{self_correction} self corrections,"\
    f"\n{japans} japanese words,\nand dialog is unfinished: {is_unfinished}"
    if need_print:
        print(message)

    return repetitions, pauses_len, filled_pauses, self_correction, japans, is_unfinished


def words_extractor(dialog):
    """
    Убирает теги, описывающие речь испытуемого
    """
    to_remove = ['\n', '<B>', '</B>', '<.>', '</.>', '<..>', '</..>', '<CO>', '</CO>', '<?>', '</?>', '<??>', 
                 '</??>', '<SC?>', '</SC?>', '<R?>', '</R?>', '<laughter>', '</laughter>']
    for item in to_remove:
        dialog = dialog.replace(item, ' ')
    list_of_counters = [0]*7
    tags = ['<F>', '<SC>', '<R>', '<OL>','<H','<nvs>','<JP>']
    closed_tags = ['</F>', '</SC>', '</R>', '</OL>','</H>','</nvs>','</JP>']
    while Counter(list_of_counters)[-1] != len(list_of_counters): 
        for i in range(len(tags)):
            t =  dialog.find(tags[i])
            list_of_counters[i] = t
            dialog = dialog if t== -1 else dialog.replace(dialog[t:dialog.find(closed_tags[i])+len(closed_tags[i])], "")
        # f = dialog.find('<F>')
        # dialog = dialog if f== -1 else dialog.replace(dialog[f:dialog.find('</F>')+len('</F>')], "")
        # sc = dialog.find('<SC>')
        # dialog = dialog if sc== -1 else dialog.replace(dialog[sc:dialog.find('</SC>')+len('</SC>')], "")
        # r = dialog.find('<R>')
        # dialog = dialog if r== -1 else dialog.replace(dialog[r:dialog.find('</R>')+len('</R>')], "")
        # ol = dialog.find('<OL>')
        # dialog = dialog if ol==-1 else dialog.replace(dialog[ol:dialog.find('</OL>')+len('</OL>')], "")
        # h = dialog.find('<H')
        # dialog = dialog if h==-1 else dialog.replace(dialog[h:dialog.find('</H>')+len('</H>')], "")
        # nvs = dialog.find('<nvs>')
        # dialog = dialog if nvs==-1 else dialog.replace(dialog[nvs:dialog.find('</nvs>')+len('</nvs>')], "")
        # jp = dialog.find('<JP>')
        # dialog = dialog if jp==-1 else dialog.replace(dialog[jp:dialog.find('</JP>')+len('</JP>')], "")
    return dialog
    

def pretty_dialog(dialog):
    dialog = dialog.lower()
    dialog = dialog.replace(' .', '.').replace('. ', '.').replace('.', '. ').replace(',', ', ')
    dialog = re.sub(r'[.]{2,}', '.', " ".join(dialog.split()).replace(' .', ''))
    return dialog


def avg_sentence_len(dialog):
    """
    Определяет среднюю длину предложения и количество коротких предложений
    """
    sentences = dialog.split(".")
    count_of_one_short_sentences = 0
    stop_words = ['yeah', 'o k', '']

    for i in range(len(sentences)):
        if len(sentences[i].split(" ")) < 4 and sentences[i].lower() not in stop_words:
            count_of_one_short_sentences+=1
    words = dialog.split(" ") 

    if(sentences[len(sentences)-1]==""):
        average_sentence_length = len(words) / (len(sentences)-1)
    else:
        average_sentence_length = len(words) / len(sentences)
    return average_sentence_length, count_of_one_short_sentences


def dialog_clear(dialog):
    dialog = dialog.replace('.', '').replace(',', '').replace('?', '').replace('"', '')
    return dialog


def nouns_of_dialog(dialog, stop_words):
    """
    Возвращает существительные в речи
    """
    word_dict = {}
    dialog = " ".join(dialog.split())
    dialog = dialog_clear(dialog)

    for word in dialog.split(' '):
        word = word.lower()
        if word not in stop_words and word != 'picture' and len(word)>2:
            if word in word_dict:
                word_dict[word]+=1
            else:
                word_dict[word]=1

    sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
    is_noun = lambda pos: pos[:2] == 'NN'
    nouns = [word for (word, pos) in nltk.pos_tag(sorted_words) if is_noun(pos)] 
    return nouns


def get_unrich_of_speech(dialog):
    """
    Определяет уровень словарного запаса
    """
    dialog = " ".join(dialog.split())
    dialog = dialog_clear(dialog)
    wordlist = dialog.split()

    wordfreq = [wordlist.count(p) for p in wordlist]
    freq_dict =  dict(list(zip(wordlist, wordfreq)))
    to_remove = ['this', 'the', 'is', 'a', 'and', 'to', 'it\'s', 'there\'s', 'there', 'i']

    for word in to_remove:
        if word in freq_dict:
            del freq_dict[word]

    aux = [(freq_dict[key], key) for key in freq_dict]
    aux.sort(reverse=True)
    length = len(aux)
    clip = min(length, 5)
    return sum([aux[i][0] for i in range(clip)])/length


def get_theme_nouns(df_nat):
    """
    Определяет существительные, типичные для каждого изображения
    """
    lemmatizer = WordNetLemmatizer()
    theme_nouns = []
    themes = list(df_nat['theme'].unique())
    
    for theme in themes:
        theme_nouns.append(list(itertools.chain.from_iterable(df_nat[df_nat.theme == theme]['nouns'])))

    theme_nouns = [[lemmatizer.lemmatize(word) for word in theme] for theme in theme_nouns]
    theme_nouns = [list(dict.fromkeys(item)) for item in theme_nouns]
    return theme_nouns, themes


def get_meaning_coef(nouns, theme_nouns, themes, curr_theme):
    """
    Возвращает коэффициент соответствия описания изображению
    """
    lemmatizer = WordNetLemmatizer()
    nouns = list(set([lemmatizer.lemmatize(n) for n in nouns]))
    nouns_of_theme = theme_nouns[themes.index(curr_theme)]
    meaning_coef = 0

    for noun in nouns:
        if noun in nouns_of_theme:
            meaning_coef+=1
    return meaning_coef


def get_count_mistakes(words, count_one_word_sentences):
    """
    Возвращает количество ошибок в речи
    """
    mistakes_types = ['COLLOCATIONS', 'GRAMMAR', 'MISC', 'CONFUSED_WORDS']
    matches = my_tool.check(words) 
    count_mistakes = 0
    for i in range(len(matches)):
        if matches[i].category in mistakes_types:
            count_mistakes+=1
    count_mistakes += count_one_word_sentences
    return count_mistakes


def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")