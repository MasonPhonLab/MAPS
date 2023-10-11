import numpy as np

def align(seq, probs):

    def distance(r, c):
        category = seq[r]
        timestep = c
        return probs[timestep, category]
        
    nrow = len(seq)
    ncol = probs.shape(0) # time steps
    
    M = warp(range(nrow), range(ncol), distance)
    
    alignment = backtrack(M, seq)
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
            M[r, c] = cost + min(M[r, c-1], M[c-1, r-1])
            
    return M
    
def backtrack(M, sequence):

    nrow, ncol = M.shape
    seq = [nrow]
    
    r = nrow
    c = ncol
    
    curr_prob = M[r, c]
    
    while c >= 0:
        if r == 0:
            seq += [1 for x in range(c-1)]
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
    
def load_dictionary(dname):

    mapping = dict()

    with open(dname, 'r') as d:
        for line in d:
            all_items = line.split()
            word = all_items[0]
            pronunciation = all_items[1:]
            
            if word not in mapping:
                mapping[word] = [pronunciation]
            else:
                mapping[word].append(pronunciation)
                
    return mapping
                