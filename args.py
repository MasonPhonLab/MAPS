import argparse

def build_arg_parser():

    p = argparse.ArgumentParser(
        description='''Mason-Alberta Phonetic Segmenter (MAPS). Phonetically segments a file or files in a directory given transcription(s) and a grapheme-to-phoneme dictionary. Passed in directories will be processed at the top level only and will not be processed recursively.
        
        Arguments that take values may use "=" or not. For example, both "--audio=s.wav" and "--audio s.wav" will work correctly.'''
    )

    p.add_argument('--audio', help='A WAV file or a directory containing WAV files', required=True)
    p.add_argument('--text', help='A txt file with an orthographic transcription or a directory with such files', required=True)
    p.add_argument('--model', help='A tf format TensorFlow model to use for alignment; if a folder with multiple models is given, all models will be used for ensemble alignment with confidence intervals', required=True)
    p.add_argument('--dict', help='A grapheme-to-phoneme dictionary like the CMU Pronouncing Dictionary', required=True)
    p.add_argument('--interp', default='true', choices=['true', 'false'], help='Whether to use interpolation or not. Default is set to true')
    p.add_argument('--rm-ensemble', action='store_true', help='Flags the program to remove the intermediate TextGrids created during ensemble alignment')
    p.add_argument('--sil', default='true', choices=['true', 'false'], help='Whether to add silences to beginning and end of transcription. Default is set to true')
    p.add_argument('--quiet', action='store_true', help='Suppresses info messages when flag is present; does not accept a value')
    p.add_argument('--check-variants', action='store_true', help='Checks pronunciation variants in the dictionary. This is currently VERY slow')
    p.add_argument('--overwrite', action='store_true', help='If flag is set, existing TextGrid files with the same stem as the WAV files will be overwritten')
    p.add_argument('--ensemble-table', action='store_true', help='Flags program to write ensemble results to a table. Appends to a filename based on path to the audio and the path to the model. You may need to delete rows from previous runs if they are not needed.')

    return p
