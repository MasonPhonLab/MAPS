# Simple demo

Until the documentation pages are written, this contents of this folder will demonstrate how the system is run.

## Relevant files contained

* **dark_suit_sentence_16khz.wav** The audio file to be segmented, recorded by @maetshju
* **dark_suit_sentence_16khz.wav** The orthogrpahic transcription of the audio file
* **dark_suit_sentence_16khz.TextGrid** The TextGrid that is created to view in Praat after running the aligner
* **sample_dictionary.txt** A sample dictionary using the CMU Pronouncing Dictionary format
* **alignment_sampe.png** Image of alignment with spectrogram and waveform, created from Praat


## Running the demo

From the root folder of this repo, the aligner can be run on the demo data with the following command:

```bash
python maps.py --audio=demo_files/dark_suit_sentence_16khz.wav --text=demo_files/dark_suit_sentence_16khz.txt --dict=demo_files/sample_dictionary.txt --model=timbuck_eng.tf
```

This command will create the dark_suit_sentence_16khz.TextGrid file, which can be viewed, along with the wav file, using [Praat](https://www.fon.hum.uva.nl/praat/).

## Limitations

As with many dictionaries, the transcriptions given in are so-called "canonical" pronunciations, which do not always match what is said in the recording. A pertinent example is that "had your" is transcribed from the dictionary as [HH AE1 D / Y ER1] in Arpabet or [hæd jɝ] in IPA, though what was actually said is more like [HH AE1 JH / Y ER1] or [hæd͡ʒ jɝ], maybe even missing the first sound of "your". This choice for the dictionary was intentional so as to realistically portray what the output will look like for data that uses dictionary transcriptions.

A crucial point is that the audio files **must** be sampled at 16 kHz. If not, the MFCC extraction will not produce the feature input the network is expecting, and the alignment quality will suffer. The resulting TextGrid can still be viewed with a recording that has a higher sample rate, however. If you need to downsample audio, to 16 kHz, both Praat and SoX are good options.
