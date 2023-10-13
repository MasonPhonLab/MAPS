import argparse

def build_arg_parser():

    p = argparse.ArgumentParser(
        description='''Mason-Alberta Phonetic Segmenter (MAPS). Phonetically segments a file or files in a directory given transcription(s) and a grapheme-to-phoneme dictionary. Passed in directories will be processed recursively to find all WAV and txt files contained therein.
        
        Arguments may use "=" or not. For example, both "--audio=s.wav" and "--audio s.wav" will work correctly.'''
    )

    p.add_argument('--audio', help='A WAV file or a directory containing WAV files', required=True)
    p.add_argument('--text', help='A txt file with an orthographic transcription or a directory with such files', required=True)
    p.add_argument('--dict', help='A grapheme-to-phoneme dictionary like the CMU Pronouncing Dictionary', required=True)
    p.add_argument('--interp', default='true', choices=['true', 'false'], help='Whether to use interpolation or not. Default is set to true.')
    p.add_argument('--sil', default='true', choices=['true', 'false'], help='Whether to add silences to beginning and end of transcription. Default is set to true')

    return p
