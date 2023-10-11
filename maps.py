from textgrid import textgrid
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import numpy as np
import os
from julia import Main
from tqdm import tqdm
import statistics
import tensorflow as tf
import python_speech_features as psf
from utils import align, collapse
import itertools
from pathlib import Path

FRAME_LENGTH = 0.025 # 25 ms expressed as seconds
FRAME_INTERVAL = 0.01 # 10 ms expressed as seconds

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()

num2phn = {i: p for i, p in enumerate(phones)}
phn2num = {p: i for i, p in enumerate(phones)}

class PhoneLabel:

    def __init__(self, phone, duration):
    
        self.phone = phone
        self.duration = duration
        
    def __str__(self):
    
        return str([self.phone, self.duration])
        
class WordString:

    def __init__(self, words, pronunciations):
        self.words = words
        self.pronunciations = pronunciations
        
        self.phone_string = itertools.chain(pronunciations)
        self.collapsed_string = itertools.chain(collapse(p) for p in pronunciations)
        
        self.did_collapse = len(self.phone_string) == self.collapsed_string)
        
    def __str__(self):
        return str([self.words, f'collapsed_diff={self.did_collapse}', self.pronunciations])
    
def force_align(collapsed, yhat):

    yhat = np.squeeze(yhat, 0)
    predictions = np.abs(np.log(yhat))
    a, M = align(collapsed, predictions)
    a = [num2phn[p] for p in a]
    seq = [PhoneLabel(phone=a[0], duration=1)]
    durI = 1
    for elem in a[1:]:
        if not seq[-1].phone == elem:
            pl = PhoneLabel(phone=elem, duration=1)
            seq.append(pl)
        else:
            seq[-1].duration += 1
    return seq, M
    
def make_textgrid(seq, tgname, maxTime, words, interpolate=True, symm=True, probs=None):
    '''
    Side-effect of writing TextGrid to disk
    '''
    
    if interpolate and np.all(probs == None):
    
        raise ValueError('If using interpolation, the alignment matrix must also be passed in through the probs argument')
        
    
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier()
    tier.name = 'phones'
    curr_dur = 0
    
    if len(seq) == 1:
        last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
        tier.intervals.append(last_interval)
        tg.tiers.append(tier)
        tg.write(tgname)
        return
    
    added_bits = []
    frame_durs = [s.duration for s in seq]
    cumu_frame_durs = [sum(frame_durs[0:i+1]) for i in range(len(frame_durs))]
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    
    if interpolate:
    
        additional = interpolated_part(seq[0].duration-1, 0, probs, symm=symm)
        if curr_dur + additional < maxTime:
            curr_dur += additional
        added_bits.append(additional)
    
    tier.intervals.append(textgrid.Interval(0, curr_dur, seq[0].phone))

    for i, s in enumerate(seq[:-1]):
    
        if i == 0: continue
    
        label = s.phone
        duration = s.duration
    
        beginning = curr_dur
        dur = FRAME_INTERVAL * duration
        
        if interpolate:
        
            endCur = cumu_frame_durs[i] - 1
        
            dur -= added_bits[-1]
            additional = interpolated_part(endCur, i, probs, symm=symm)
            if beginning + dur + additional < maxTime:
                dur += additional
            added_bits.append(additional)
        
        ending = beginning + dur
        
        interval = textgrid.Interval(beginning, ending, label)
        tier.intervals.append(interval)
        
        curr_dur = ending
    
    last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
    tier.intervals.append(last_interval)
    
    if words.did_collapse:
        unsplit_phones(tier, words)
    
    word_tier = make_word_tier(tier, words)
    tg.tiers.append(word_tier)
    tg.tiers.append(tier)
    
    tg.write(tgname)
    
def unsplit_phones(tier, words):

    unc_i = 0
    
    split_idxs = []
    
    for col_i, p in enumerate(words.collapsed_string):
    
        unc_p = words.phone_string[col_i]
        if p != unc_p:
            split_idxs.append(col_i-1)
            unc_i += 1
        
        unc_i += 1
    
    # work in reverse order so that in-place modification of list and concomitant
    # changes in later index numbers don't affect earlier ones
    for i in split_idxs[::-1]:
    
        midpoint = (tier[i].minTime + tier[i].maxTime) / 2
        new_interval = textgrid.Interval(minTime=midpoint, maxTime=tier[i.maxTime], tier[i].mark)
        tier[i].maxTime = midpoint
        tier.insert(i+1, new_interval)
        
    
def make_word_tier(segment_tier, words):

    phone_string = itertools.chain(words.phone_string)
    
    words_int = textgrid.IntervalTier()
    word_ends = np.cumsum([len(p) for p in words.pronunciations]) - 1
    maxTime = segment_tier[word_ends[0]].maxTime
    interval = textgrid.Interval(minTime=0, maxTime=maxTime, words[0])
    words_int.append(interval)
    
    for w, idx in zip(words[1:], word_ends[1:]):
        minTime = words_int[-1].maxTime
        maxTime = segment_tier[word_ends[idx]].maxTime
        interval = textgrid.Interval(minTime=minTime, maxTime=maxTime, w)
        words_int.append(interval)
    
    return words_int
    
def interpolated_part(endCur, phone_n, probs):

    phone1_curr = probs[endCur, phone_n]
    phone1_next = probs[endCur+1, phone_n]

    phone2_curr = probs[endCur, phone_n+1]
    phone2_next = probs[endCur+1, phone_n+1]
        
    m1 = (phone1_next - phone1_curr) / FRAME_INTERVAL
    m2 = (phone2_next - phone2_curr) / FRAME_INTERVAL
    
    A = np.array([[-m1, 1], [-m2, 1]])
    b = [phone1_curr, phone2_curr]
    
    try:
        time_point, intersection_probability = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0
    
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    if 0 <= time_point < FRAME_INTERVAL:
        return time_point
        
    return 0
    

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Required command line arguments missing. Command syntax:', file=sys.stderr)
        print('\tpython maps.py AUDIO_NAME TRANSCRIPTION_NAME DICTIONARY_NAME (USE_INTERPOLATION)', file=sys.stderr)
        print('Note that the parentheses indicate that the USE_INTERPOLATION variable is opitonal, not that it should have surrounding parentheses.')

    mod_name = 'timbuck_eng.tf'
    MODEL = load_model(mod_name, compile=False)
    
    wavname = Path(sys.argv[1])
    
    if not wavname.is_file():
        print(f'Could not find {wavname}. Please check the spelling and try again.', file=sys.stderr)
        sys.exit(1)
    
    transcription = Path(sys.argv[2])
    
    if not transcription.is_file():
        print(f'Could not find {transcription}. Please check the spelling and try again.', file=sys.stderr)
        sys.exit(1)
    
    d_path = Path(sys.argv[3])
    
    if not d_path.is_file():
        print(f'Could not find {d_path}. Please check the spelling and try again.', file=sys.stderr)
        sys.exit(1)
    
    word2phone = load_dictionary(d_path)
    
    if sys.argv[4]:
        if sys.argv[4].lower() == 'true':
            use_interp = True
        elif sys.argv[4].lower() == 'false':
            use_interp = False
        else:
            print(f'Unsupported value for USE_INTERPOLATION: {sys.argv[4]}. Please use "True", "False", or leave blank to use default of "True".', file=sys.stderr)
            sys.ext(1)
            
    else: use_interp=True
    
    tgname = wavname.with_suffix('.TextGrid')s

    sr, samples = wavfile.read(wavname)
    duration = samples.size / sr # convert samples to seconds
        
    mfcc = psf.mfcc(samples, sr, winstep=FRAME_INTERVAL)
    delta = psf.delta(mfcc, 2)
    deltadelta = psf.delta(delta, 2)
        
    x = np.hstack((mfcc, delta, deltadelta))
    x = np.expand_dims(x, axis=0)
    yhat = MODEL.predict(x, verbose=0)
    
    with open(transcription, 'r') as f:
        word_labels = f.read().split()
        
    word_chain = []
    ood_words = []
    for w in word_labels:
        try: word_chain.append(word2phone[w])
        except KeyError:
            ood_words.append(w)
            
    if ood_words:
        print('The following words were not found in the dictionary. Please add them to the dictionary and run the aligner again.', file=sys.stderr)
        print(', '.join(ood_words), file=sys.stderr)
        sys.exit(1)
        
    best_score = np.inf
    
    # Iterate through pronunciation variants to choose best alignment
    for c in itertools.product(word_chain*):
        w_string = WordString(word_chain, c)
        
        seq, M = force_align(w_string.collapsed_string, yhat)
        if M[-1, -1] < best_score:
            best_seq = seq
            best_M = M
            best_score = M[-1, -1]        

    make_textgrid(seq, tgname, duration, w_string, interpolate=use_interp, probs=M.T)
        