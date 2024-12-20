import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from textgrid import textgrid
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import numpy as np
import re, sys, itertools
from tqdm import tqdm
import tensorflow as tf
import python_speech_features as psf
from utils import align, collapse, load_dictionary
from pathlib import Path
from args import build_arg_parser
import statistics
import math
import warnings

EPS = 1e-8

FRAME_LENGTH = 0.025 # 25 ms expressed as seconds
FRAME_INTERVAL = 0.01 # 10 ms expressed as seconds

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()

num2phn = {i: p for i, p in enumerate(phones)}
phn2num = {p: i for i, p in enumerate(phones)}
phn2num['sil'] = phn2num['h#']

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
        
        self.phone_string = list(itertools.chain(*pronunciations))
        self.collapsed_string = collapse([re.sub(r'[0-9]', '', x) for x in self.phone_string])
        
        self.did_collapse = len(self.phone_string) != len(self.collapsed_string)
        
    def __str__(self):
        return str([self.words, f'collapsed_diff={self.did_collapse}', self.pronunciations])
    
def force_align(collapsed, yhat):

    yhat = np.squeeze(yhat, 0)
    predictions = np.abs(np.log(yhat))
    collapsed = [phn2num[x.lower()] for x in collapsed]
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
    
def make_textgrid(seq, tgname, maxTime, words, interpolate=True, probs=None):
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
        if words.did_collapse: unmerge_phones(tier, words)
        word_tier = make_word_tier(tier, words)
        tg.tiers.append(word_tier)
        tg.tiers.append(tier)
        tg.write(tgname)
        return
    
    added_bits = []
    frame_durs = [s.duration for s in seq]
    cumu_frame_durs = [sum(frame_durs[0:i+1]) for i in range(len(frame_durs))]
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    
    if interpolate:
    
        additional = interpolated_part(seq[0].duration-1, 0, probs)
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
            additional = interpolated_part(endCur, i, probs)
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
        unmerge_phones(tier, words)

    for i in range(len(tier.intervals)):
        x = words.phone_string[i]
        y = tier.intervals[i]
        if x != y.mark:
            tier.intervals[i].mark = x

        # Prevent small boundary errors from numerical instability
        if i > 0:
            prev_end = tier.intervals[i-1].maxTime
            curr_start = tier.intervals[i].minTime

            if math.isclose(prev_end, curr_start) and not prev_end == curr_start:
                tier.intervals[i-1].maxTime = curr_start

    word_tier = make_word_tier(tier, words)
    tg.tiers.append(word_tier)
    tg.tiers.append(tier)
    
    tg.write(tgname)

def to_bucket_fmt(s):

    s = [re.sub(r'[0-9]', '', x) for x in s]

    buckets = []
    prev = s[0]
    count = 1
    for x in s[1:]:
        if x != prev:
            buckets.append((prev, count))
            count = 0
        prev = x
        count += 1
    buckets.append((prev, count))
    return buckets
    
def unmerge_phones(tier, words):

    collapsed_bucket = to_bucket_fmt(words.collapsed_string)
    uncollapsed_bucket = to_bucket_fmt(words.phone_string)

    intervals = []

    for i, (c, u) in enumerate(zip(collapsed_bucket, uncollapsed_bucket)):

        dur = tier.intervals[i].maxTime - tier.intervals[i].minTime
        chunk_dur = dur / u[1]
        mint = tier.intervals[i].minTime

        for j in range(u[1]):
            low = mint + j * chunk_dur
            high = low + chunk_dur
            interv = textgrid.Interval(minTime=low, maxTime=high, mark=c[0])
            intervals.append(interv)

    tier.intervals = intervals
    return
    
def make_word_tier(segment_tier, words):

    phone_string = list(itertools.chain(words.phone_string))
    
    words_int = textgrid.IntervalTier()
    words_int.name = 'words'
    word_ends = np.cumsum([len(p) for p in words.pronunciations]) - 1
    maxTime = segment_tier[word_ends[0]].maxTime
    interval = textgrid.Interval(minTime=0, maxTime=maxTime, mark=words.words[0])
    words_int.intervals.append(interval)
    
    for w, w_end in zip(words.words[1:], word_ends[1:]):
        minTime = words_int[-1].maxTime
        maxTime = segment_tier[w_end].maxTime
        interval = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=w)
        words_int.intervals.append(interval)
    
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

    p = build_arg_parser()
    args = p.parse_args()
    args = vars(args)
    
    wavname_path = Path(args['audio'])
    
    if not wavname_path.is_file() and not wavname_path.is_dir():
        raise RuntimeError(f'Could not find {wavname_path}. Please check the spelling and try again.')
    elif wavname_path.is_dir():
        wavnames = [wavname_path / Path(x) for x in os.listdir(wavname_path) if x.lower().endswith('.wav')]
        wavnames.sort()
    else: wavnames = [wavname_path]
    
    model_path = Path(args['model'])
    if not model_path.suffix == '.tf':
        model_names = sorted([x for x in model_path.iterdir() if x.suffix == '.tf'])
        if not model_names:
            raise RuntimeError(f'Could not find a model named {model_path}, nor any models within that path. Please check spelling and file extensions and try again.')
    else:
        model_names = [model_path]
    
    use_ensemble = len(model_names) > 1
    rm_ensemble = args['rm_ensemble']
    ensemble_table = args['ensemble_table']
    
    transcription_path = Path(args['text'])
    
    if not transcription_path.is_file() and not transcription_path.is_dir():
        raise RuntimeError(f'Could not find {transcription_path}. Please check the spelling and try again.')
    elif transcription_path.is_dir():
        transcriptions = [transcription_path / Path(x.name).with_suffix('.txt') for x in wavnames]
    else: transcriptions = [transcription_path]
    
    w_set = set(x.stem for x in wavnames)
    t_set = set(x.stem for x in transcriptions)
    
    mismatched = []
    for w in wavnames:
        if w.stem not in t_set:
            mismatched.append(w)
    
    for t in transcriptions:
        if t.stem not in w_set:
            mismatched.append(t)
    
    if mismatched:
        raise RuntimeError(f'The following files did not have a corresponding WAV or txt match. Please add matches or remove the files. Note that name matching is case-sensitive.\n{",".join(str(x) for x in mismatched)}')
    
    d_path = Path(args['dict'])
    
    if not d_path.is_file():
        raise RuntimeError(f'Could not find {d_path}. Please check the spelling and try again.')
    
    word2phone = load_dictionary(d_path)
    
    # Parentheses to help visually distinguish the "=" and "==" in the same line
    use_interp = (args['interp'] == 'true')        
    add_sil = (args['sil'] == 'true')
    
    tgnames = [x.with_suffix('.TextGrid') for x in wavnames]
    
    word_list = []
    for t in transcriptions:
        with open(t, 'r') as f:
            w = f.read().upper().split()
            word_list += w
    
    ood_words = set([w for w in word_list if w not in word2phone])
    
    if ood_words:
        raise RuntimeError(f'The following words were not found in the dictionary. Please add them to the dictionary and run the aligner again.\n{", ".join(ood_words)}')

    quiet = args['quiet']
    
    filenames = list(zip(tgnames, wavnames, transcriptions))
    
    if not quiet:
        print('BEGINNING ALIGNMENT')

    overwrite = args['overwrite']

    for m_I, m_name in enumerate(model_names, start=1):

        # if m_name.suffix == '.tf':
        #     warnings.warn('TensorFlow has stopped supporting the tf format. Your models may need to be updated to the keras or h5 formats for long-term functionality.')
        #     m = tf.keras.layers.TFSMLayer(m_name, call_endpoint='serving_default')
        m = load_model(m_name, compile=False)
        
        print(f'USING MODEL {m_name.name} ({m_I}/{len(model_names)})', flush=True)
        
        if not quiet: filenames = tqdm(filenames)
    
        for tgname_base, wavname, transcription in filenames:

            if use_ensemble:
                tgname = tgname_base.parent / tgname_base.parts[-1].replace('.TextGrid', f'_{m_name.stem}.TextGrid')
            else:
                tgname = tgname_base

            if tgname.is_file() and not overwrite: continue
            
            sr, samples = wavfile.read(wavname)
            duration = samples.size / sr # convert samples to seconds
                
            mfcc = psf.mfcc(samples, sr, winstep=FRAME_INTERVAL)
            delta = psf.delta(mfcc, 2)
            deltadelta = psf.delta(delta, 2)
                
            x = np.hstack((mfcc, delta, deltadelta))
            x = np.expand_dims(x, axis=0)

            yhat = m.predict(x, verbose=0)
            
            with open(transcription, 'r') as f:
                word_labels = f.read().upper().split()
                
            if add_sil and duration >= 0.045:
                word_labels = ['sil'] + word_labels + ['sil']
            elif add_sil:
                warnings.warn(f'Silence segments not added to ends of transcription for {wavname} because duration of {duration} s is too short to have silence padding.')
            word_chain = [word2phone[w] for w in word_labels]
                
            best_score = np.inf

            check_variants = args['check_variants']
            best_w_string = 0
            
            # Iterate through pronunciation variants to choose best alignment
            # TODO: This iteration only checks segmental differences; stress differences won't get evaluated
            #   and may end up semi-randomly chosen (or choose only first option)
            #
            # This method will very quickly cause combinatoric explosion since function words have
            # several variants
            for c in itertools.product(*word_chain):
            
                # Remove empty 'sil' options
                if add_sil:
                    this_word_labels = [x for cI, x in zip(c, word_labels) if cI]
                    c = [x for x in c if x]
                else:
                    this_word_labels = word_labels

                w_string = WordString(this_word_labels, c)
                if add_sil and len(w_string.collapsed_string) > (duration - 0.015) / 0.01:
                    if this_word_labels[0] == 'sil':
                        this_word_labels = this_word_labels[1:]
                        c = c[1:]
                        warnings.warn(f'File {wavname} with duration {duration} too short for adding silence to transcription {w_string.collapsed_string}. Removing first silence label.')
                    if this_word_labels[-1] == 'sil':
                        this_word_labels = this_word_labels[:-1]
                        c = c[:-1]
                        warnings.warn(f'File {wavname} with duration {duration} too short for adding silence to transcription {w_string.collapsed_string}. Removing final silence label.')

                    w_string = WordString(this_word_labels, c)

                if best_w_string == 0: best_w_string = w_string

                seq, M = force_align(w_string.collapsed_string, yhat)
                if M[-1, -1] < best_score:
                    best_seq = seq
                    best_M = M
                    best_score = M[-1, -1]
                    best_w_string = w_string

                if not check_variants: break

            n_segs = len(best_w_string.collapsed_string)
            if n_segs > 1 and duration < 0.015 + (0.01 * n_segs):
                warnings.warn(f'File {wavname} with duration {duration} too short for collapsed {len(best_w_string.collapsed_string)}-segment best transcription {best_w_string.collapsed_string}. Assigning equal durations for each segment.')

                intervals = []
                for i, x in enumerate(best_w_string.collapsed_string):
                    d_min = i / len(best_w_string.collapsed_string) * duration
                    d_max = (i+1) / len(best_w_string.collapsed_string) * duration

                    intervals.append(textgrid.Interval(minTime=d_min, maxTime=d_max, mark=x))

                tier = textgrid.IntervalTier('segments')
                tier.intervals = intervals
                word_tier = make_word_tier(tier, best_w_string)
                tg = textgrid.TextGrid()
                tg.tiers.append(word_tier)
                tg.tiers.append(tier)
                tg.write(tgname)
                continue

            make_textgrid(best_seq, tgname, duration, best_w_string, interpolate=use_interp, probs=best_M.T)

    if use_ensemble:
        print('ENSEMBLING', flush=True)
        
        if ensemble_table:
            f_path = f'{"_".join(wavname_path.parts)}_{model_path.name}_alignment_results.tsv'
            col_names = ['file', 'word', 'word_mintime', 'word_maxtime', 'segment', 'segment_mintime', 'segment_maxtime', 'segment_se', 'segment_lo_ci', 'segment_hi_ci']
            with open(f_path, 'a') as w:
                w.write('\t'.join(col_names) + '\n')
        
        all_tg_names = list()
        for tgname_base, _, _ in tqdm(filenames):
            
            ensemble_tg_path = Path(tgname_base.parent, tgname_base.stem + '_ensemble.TextGrid')
            
            ens_intervals = textgrid.IntervalTier(name='segments')
            intervals = list()
            cis = list()
            
            tg_names = list()
            
            for m_name in model_names:
                tail = tgname_base.parts[-1].replace('.TextGrid', f'_{m_name.stem}.TextGrid')
                t = tgname_base.parent / tail
                tg_names.append(t)
                
            all_tg_names += tg_names
            if ensemble_tg_path.is_file() and not overwrite: continue
                
            tgs = [textgrid.TextGrid() for _ in tg_names]
            for tg, tg_name in zip(tgs, tg_names):
                tg.read(tg_name, round_digits=1000)
                
            n_tgs = len(tgs)

            n_intervals = len(tgs[0].tiers[1].intervals)
            duration = tgs[0].tiers[1].maxTime
            
            for i in range(n_intervals):
                lab = tgs[0].tiers[1].intervals[i].mark
                mintimes = [tgs[tier_I].tiers[1].intervals[i].minTime for tier_I in range(n_tgs)]
                maxtimes = [tgs[tier_I].tiers[1].intervals[i].maxTime for tier_I in range(n_tgs)]
                
                mintime = statistics.mean(mintimes)
                maxtime = statistics.mean(maxtimes)
                
                sd = statistics.stdev(maxtimes)
                se = sd / math.sqrt(len(model_names))
                ci_lo = max(0, maxtime - 1.96 * se)
                ci_hi = min(maxtime + 1.96 * se, duration)
                if ci_lo == ci_hi:
                    ci_lo -= EPS
                    ci_hi += EPS
                
                interval = textgrid.Interval(minTime=mintime, maxTime=maxtime, mark=lab)
                intervals.append(interval)
                
                if i < n_intervals - 1:
                    ci_lo_p = textgrid.Point(mark=f'{lab}_cilo', time=ci_lo)
                    ci_hi_p = textgrid.Point(mark=f'{lab}_cihi', time=ci_hi)
                    cis += [ci_lo_p, ci_hi_p]
                
            n_word_intervals = len(tgs[0].tiers[0].intervals)
            word_intervals = list()
            
            for i in range(n_word_intervals):
                lab = tgs[0].tiers[0].intervals[i].mark
                mintimes = [tgs[tier_I].tiers[0].intervals[i].minTime for tier_I in range(n_tgs)]
                maxtimes = [tgs[tier_I].tiers[0].intervals[i].maxTime for tier_I in range(n_tgs)]
                
                mintime = statistics.mean(mintimes)
                maxtime = statistics.mean(maxtimes)
                
                word_interval = textgrid.Interval(minTime=mintime, maxTime=maxtime, mark=lab)
                word_intervals.append(word_interval)
            
            # make word tier here
            
            ens_tg = textgrid.TextGrid(maxTime=tgs[0].maxTime)
            
            word_tier = textgrid.IntervalTier(name='words')
            word_tier.intervals = word_intervals
            ens_tg.tiers.append(word_tier)
            
            int_tier = textgrid.IntervalTier(name='segments')
            int_tier.intervals = intervals
            ens_tg.tiers.append(int_tier)
            
            ci_tier = textgrid.PointTier(name='95-CIs')
            ci_tier.points = cis
            ens_tg.tiers.append(ci_tier)
            
            ens_tg.write(ensemble_tg_path)
            
            all_tg_names += tg_names

            if ensemble_table:
                with open(f_path, 'a') as w:
                    
                    fname = ensemble_tg_path.name
                    
                    word_iter = iter(word_intervals)
                    word = next(word_iter)
                    
                    for x_I, x in enumerate(intervals):
                        if x_I == len(intervals) - 1:
                            segment_lo_ci = x.minTime
                            segment_hi_ci = x.maxTime
                            segment_se = 0
                        else:
                            segment_lo_ci = cis[x_I * 2].time
                            segment_hi_ci = cis[x_I * 2 + 1].time
                            segment_se = (segment_hi_ci - segment_lo_ci) / (2 * 1.96)
                        
                        s = [fname, word.mark, word.minTime, word.maxTime, x.mark, x.minTime, x.maxTime, segment_se, segment_lo_ci, segment_hi_ci]
                        s = '\t'.join([str(z) for z in s])
                        
                        w.write(s + '\n')
                        
                        if x.maxTime == word.maxTime and x_I < len(intervals) - 1:
                            word = next(word_iter)
            
        # Remove ensemble files if flagged to remove
        if rm_ensemble:
            for n in all_tg_names: n.unlink(missing_ok=True)
