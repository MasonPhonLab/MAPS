import numpy as np
import re

_FOLDINGS = {
    'AO': 'AA',
    'AX': 'AH',
    'AX-H': 'AH',
    'AXR': 'ER',
    'HV': 'HH',
    'IX': 'IH',
    'EL': 'L',
    'EM': 'M',
    'EN': 'N',
    'NX': 'N',
    'ENG': 'NG',
    'PCL': 'P',
    'TCL': 'T',
    'KCL': 'K',
    'BCL': 'B',
    'DCL': 'D',
    'GCL': 'G',
    'H#': 'SIL',
    'PAU': 'SIL',
    'EPI': 'SIL',
    'UX': 'UW'       
}

def align(seq, probs):

    def distance(r, c):
        category = seq[r]
        timestep = c
        return probs[timestep, category]
        
    nrow = len(seq)
    ncol = probs.shape[0] # time steps
    
    M = warp(range(nrow), range(ncol), distance)
    
    alignment = backtrack(M[1:, 1:], seq)
    return alignment, M[1:, 1:]
    
def warp(S, T, d):
    
    nrow = len(S)
    ncol = len(T)
    
    M = np.full((nrow+1, ncol+1), np.inf)
    
    M[0, 0] = 0
    
    for r in range(1, nrow+1):
        for c in range(1, ncol+1):
            # need -1 offset because of extra first row
            # and first column in M that isn't present in
            # the `probs` matrix
            cost = d(r-1, c-1)
            M[r, c] = cost + min(M[r, c-1], M[r-1, c-1])
            
    return M
    
def backtrack(M, sequence):

    nrow, ncol = M.shape
    seq = [nrow-1]
    
    r = nrow-1
    c = ncol-1
    
    curr_prob = M[r, c]
    
    while c >= 0:
        if r == 0:
            seq += [0 for x in range(ncol - len(seq))]
            break
        
        if M[r-1, c-1] > M[r, c-1]:
            seq.append(r)
            c -= 1
        else:
            c -= 1
            r -= 1
            
            seq.append(r)
            
    rs = seq[::-1]
    rs = [sequence[r] for r in rs]
    
    return rs
    
def collapse(s):
    a = [s[0]]
    for symbol in s[1:]:
        if a[-1] != symbol:
            a.append(symbol)
    return a
    
def fold_phone(p):

    stress_mark = re.findall(r'\d+', p)
    if stress_mark:
        stress_mark = stress_mark[0]
    else:
        stress_mark = ''
    phone_label = re.sub(r'\d', '', p)
    if phone_label in _FOLDINGS:
        phone_label = _FOLDINGS[phone_label]
    return phone_label + stress_mark

    
def load_dictionary(dname, cmuformat=True):

    

    mapping = dict()
    
    if cmuformat:

        with open(dname, 'r') as d:
            for line in d:
                all_items = line.split()
                word = all_items[0]
                word = re.sub(r'\(\d*\)', '', word)
                pronunciation = all_items[1:]
                pronunciation = [fold_phone(p) for p in pronunciation]
                
                if word not in mapping:
                    mapping[word] = [pronunciation]
                else:
                    mapping[word].append(pronunciation)
                    
        return mapping
        
    else:
        print('Custom dictionary formats not supported yet. Please format use CMU dict formatting and run again.')
        sys.exit(1)
                