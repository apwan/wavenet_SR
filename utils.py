
import nltk
from functools import lru_cache
from pathlib import Path
import os


def dict2obj(d: dict) -> object:
    obj = type('dict_obj', (object, ), d)
    for i,j in d.items():
        setattr(obj, i, j)  
    return obj  


ph2index = {
"iy": 1,
"ix": 2,
"ih": 2,
"eh": 3,
"ae": 4,
"ax": 5,
"ah": 5,
"axh": 5,
"uw": 6,
"ux": 6,
"uh": 7,
"ao": 8,
"aa": 8,
"ey": 9,
"ay": 10,
"oy": 11,
"aw": 12,
"ow": 13,
"er": 14,
"axr": 14,
"l": 15,
"el": 15,
"r": 16,
"w": 17,
"y": 18,
"m": 19,
"em": 19,
"n": 20,
"en": 20,
"nx": 20,
"ng": 21,
"eng": 21,
"v": 22,
"f": 23,
"dh": 24,
"th": 25,
"z": 26,
"s": 27,
"zh": 28,
"sh": 28,
"jh": 29,
"ch": 30,
"b": 31,
"p": 32,
"d": 33,
"dx": 34,
"t": 35,
"g": 36,
"k": 37,
"hh": 38,
"hv": 38,
"bcl": 0,
"pcl": 0,
"dcl": 0,
"tcl": 0,
"gcl": 0,
"kcl": 0,
"q": 0,
"epi": 0,
"pau": 0,
"!ENTER": 0,
"!EXIT": 0,
}


# IO

src_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, './')))

def abs_dir(relative_dir):
    if type(relative_dir) is not str:
        relative_dir = str(relative_dir)
    if relative_dir.startswith('/'):
        return relative_dir
    else:
        return os.path.abspath(os.path.join(src_dir, relative_dir))


def files_by_suffixes(dir, suffixes, sort=True):
    data_dir = Path(dir)
    if type(suffixes) is str:
        suffixes = [suffixes]
    ret = [f.name for suffix in suffixes for f in data_dir.glob(f"*.{suffix}")]
    if sort:
        return sorted(ret)
    else:
        return ret   


'''
Dataset preparation
'''


def data_install(dataset):
    try:
        nltk.find('corpora/{}'.format(dataset))
        print("{} already exists".format(dataset))
    except:
        print('downloading {}'.format(dataset))
        nltk.download(dataset)


ipa_table = None

def load_ipa_table(file=abs_dir('arrpa_ipa.json')):
    global ipa_table
    import json, codecs
    input_file = codecs.open(file, 'r', encoding='utf8')
    ipa_table = json.loads(input_file.read())
    

@lru_cache(maxsize=1)
def get_ipa_table():
    if ipa_table is None:
        load_ipa_table()


    values = list(ipa_table.values())


    if type(values[0]) is dict:    
        return {k:v for x in values for (k,v) in x.items()}
    else:
        return ipa_table



def fix_multikey(d):
    pp = [(x, x.index('(')) for x in d.keys() if '(' in x]
    #print(len (pp), len(d))
    for (x, i) in pp:
        k = x[:i]
        d[k].extend(d[x])
        d.pop(x)
    #print('{} items left.'.format(len(d)))
    return d



from nltk.corpus.reader.cmudict import CMUDictCorpusReader

# TODO: migrate manual overloading to class inheritance

def _read_cmudict_block(stream):
    entries = []
    while len(entries) < 100: # Read 100 at a time.
        line = stream.readline()
        if line == '': return entries # end of file.
        pieces = line.split()
        # overload due to different format
        entries.append( (pieces[0].lower(), pieces[1:]) )
    return entries


def _dict(self, overloaded=False):
    from nltk.util import Index
    if overloaded:
        _tmp = nltk.corpus.reader.cmudict.read_cmudict_block
        nltk.corpus.reader.cmudict.read_cmudict_block = _read_cmudict_block
        ent = self.entries()
        #print(ent)
        nltk.corpus.reader.cmudict.read_cmudict_block = _tmp # restore
        return fix_multikey(dict(Index(ent)))
    else:
        return dict(Index(self.entries()))
    


import string


CMUDictCorpusReader.dict = _dict

en_dict, cn_dict, cn_phnset, en_phnset = None, None, None, None


@lru_cache(maxsize=1)
def init_dicts():
    global en_dict, cn_dict, cn_phnset, en_phnset
    en_dict = CMUDictCorpusReader(abs_dir('./'), 'cmudict-en-us.dict').dict(overloaded=True)
    cn_dict = CMUDictCorpusReader(abs_dir('./'), 'zh_broadcastnews_utf8.dict').dict(overloaded=True)
    cn_phnset = set(j for x in cn_dict.values() for i in x for j in i)
    cn_phnset = sorted(list(cn_phnset))
    en_phnset = set(x for y in en_dict.values() for z in y for x in z)
    en_phnset = sorted(list(en_phnset))

    cn_dict.update({chr(i):cn_dict[chr(i)+'.'] for i in range(ord('a'), ord('z')+1)})
    cn_dict.update({chr(ord('ａ')+i):cn_dict[chr(ord('a')+i)] for i in range(26)})
    cn_num = '零 一 二 三 四 五 六 七 八 九'.split()
    cn_dict.update({str(i): cn_dict[cn_num[i]] for i in range(len(cn_num))})

    trans_dict = {c: '' for c in string.punctuation}
    trans_dict['妳'] = '你'
    trans_dict.update({str(i): cn_num[i] for i in range(len(cn_num))})

    trans_dict = str.maketrans(trans_dict)

    def good_string(sent):
        return sent.translate(trans_dict).lower()


    return good_string


# TODO: change to lazy init
good_string = init_dicts()



# build hashmap to speedup searching
def make_gp(d):
    gp = {}
    for i in d:
        k = i[0] + str(len(i))
        if k not in gp:
            gp[k] = []
        gp[k].append(i) 
    return gp

cn_gp = None
#cn_gp = make_gp(cn_dict)


def longest_match(word, gp):
    k = word[0]
    l = len(word)
    for i in range(l, 0, -1):
        ki = k+str(i)
        if ki in gp and word[:i] in gp[ki]:
            return word[:i], i
    return None, 0

def lr_tokenize(sent, gp, ahead=6):
    l = len(sent)
    cur = 0
    ret = []
    while cur < l:
        front = min(l, cur+ahead)
        mat, n = longest_match(sent[cur:front], gp)
        if n == 0:
            print("invalid "+sent[cur])
            ret.append(None)
            cur += 1
        else:
            ret.append(mat)
            cur += n
    return ret



def cn_phon(sent, sep='<br>', d=cn_dict, gp=cn_gp, ahead=6):
    global cn_gp
    sent = good_string(sent)
    if cn_gp is None:
         cn_gp = make_gp(cn_dict)
    if gp is None:
        gp = cn_gp

    tokens = lr_tokenize(sent, gp=gp, ahead=ahead)
    return [[sep] if x is None else d[x][0] for x in tokens]


def en_phon(sent, sep='<br>', d=en_dict):
    sent = good_string(sent)
    tokens = sent.split()

    return [d.get(x)[0] if x in d else [sep] for x in tokens]


def get_en_dict(ipa=False):
    if en_dict is None:
        init_dicts()
    if ipa:
        trans = get_ipa_table() # flattened
        ret = {i:list(map(lambda x:trans[x], j[0])) for i,j in en_dict.items()}
    else:
        return {i:j[0] for i,j in en_dict.items()}

@lru_cache(maxsize=2)
def get_en_phnset(ipa=False):
    if en_phnset is None:
        init_dicts()   
    if ipa:
        trans = get_ipa_table()
        return [trans[x] for x in en_phnset]
    else:
        return en_phnset
            

# for source separation evaluation

import librosa
import numpy as np
from config import ModelConfig
import soundfile as sf
from mir_eval.separation import bss_eval_sources

def get_wav(filename, sr=ModelConfig.SR):
    src1_src2 = librosa.load(filename, sr=sr, mono=False)[0]
    mixed = librosa.to_mono(src1_src2)
    src1, src2 = src1_src2[0, :], src1_src2[1, :]
    return mixed, src1, src2

def to_wav_file(mag, phase, len_hop=ModelConfig.L_HOP):
    stft_maxrix = get_stft_matrix(mag, phase)
    return np.array(librosa.istft(stft_maxrix, hop_length=len_hop))

def to_spec(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return librosa.stft(wav, n_fft=len_frame, hop_length=len_hop)

def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

def write_wav(data, path, sr=ModelConfig.SR, format='wav', subtype='PCM_16'):
    sf.write(path, data, sr, format=format, subtype=subtype)

def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:len]
    src2_wav = src2_wav[:len]
    mixed_wav = mixed_wav[:len]
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), compute_permutation=True)
    # sdr, sir, sar, _ = bss_eval_sources(src2_wav,pred_src2_wav, False)
    # sdr_mixed, _, _, _ = bss_eval_sources(src2_wav,mixed_wav, False)
    nsdr = sdr - sdr_mixed
    return nsdr, sir, sar, len

def bss_eval_sdr(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr, _, _, _ = bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr


