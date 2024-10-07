---
title: Model card for MAPS acoustic model
author:
    - Matthew C. Kelley
    - Scott James Perry
    - Benjmain V. Tucker
date: October 2023
---

# Model details

## Model description

* Serves as the acoustic model in the Mason-Alberta Phonetic Segmenter (MAPS) tool
* Bidirectional LSTM-based network
* Operates over vectors of 13 mel frequency cepstral coefficients (MFCCs) along with their delta and delta-delta coefficients, for a total vector length of 39
* Yields posterior probabilities of American English phoneme classes given the acoustic feature vector
* **Developed by:** Matthew C. Kelley, Scott James Perry, and Benjamin V. Tucker
* **Language:** English
* **Model type:** Acoustic model / phone recognizer
* **License:** MIT
* **Parent model:** None
* **Resources for more information:** [MAPS GitHub repo](https://github.com/MasonPhonLab/MAPS), [arXiv pre-print](https://arxiv.org/abs/2310.15425), [MAPS_Paper_Code GitHub repo](https://github.com/MasonPhonLab/MAPS_Paper_Code/)

## Model sources

* **Repository:** [https://github.com/MasonPhonLab/MAPS_Paper_Code](https://github.com/MasonPhonLab/MAPS_Paper_Code)
* **Paper:** Kelley, M. C., Perry, S. J., & Tucker, B. V. (Submitted). The Mason-Alberta Phonetic Segmenter: A forced alignment system based on deep neural networks and interpolation.
* **Demo:** [https://github.com/MasonPhonLab/MAPS](https://github.com/MasonPhonLab/MAPS_Paper_Code)

# Uses

## Direct use

The model is designed to yield posterior probabilities of American English phones given an acoustic vector of length 39 that is made up of 12 mel frequency cepstral coefficients and the 0th coefficient replaced with log energy, in addition to the associated delta and delta-delta coefficients. The intended use of this model was for forced alignment purposes, to determine the boundaries between the phones given in a phonetic transcription of an utterance.

## Downstream use

The model could be used for other speech tasks, such as examining the overall entropy of the posterior of an American English phone recognizer when presented with a specific acoustic vector. It could also be used for varieties or languages that are not American English if their associated phone sets are coereced to "match" those of American English.

## Out-of-scope use

Outside of research directly on this topic, this model should not be used to determine acceptability or prototypicality of speech articulations since it was not trained on human judgments of acceptability, and we are not aware of any research that ties this kind of posterior to articulation acceptability. It also should not be used for non-speech data.

# Bias, risks, and limitations

By training on the TIMIT and Buckeye corpora, the training data consist majoritarily of speech from white speakers. TIMIT attempts to achieve some degree of dialectal balance, but there is not a balance for race or ethnicity. The Buckeye speech corpus contains only speech from white speakers around the Columbus, Ohio region, but it is stratified for gender and sex within that cross-section. The amount of data in the Buckeye corpus is several times greater than the amount in TIMIT, so it will produce the heaviest bias in the model.

## Bias

In more specific detail, the model will be most likely to work best on speech that is similar to that in the Midwest, and more specifically to
the speaking patterns of white speakers of English near Columbus, Ohio. Sociolphonetic variation in vowel production across the US or across social groups, for example, may cause the model to struggle with different vowel patterns.

## Risks

* It is conecivable that one could conclude from using this model on under-represented varieties of American English that said varieties are naturally more difficult to segment. In point of fact, though, any difficulties in segmentation are far more likely to be due to the model not being trained on specific kinds of data or the qualities of the recordings used not being commensurate to the recordings trained on.
* It is also conceivable that, counter to our indication of out-of-scope use, this model is used to determine what kind of speakers speak the "best" kind of English by comparing performance of the recognizer on different varieties of speech. This would lead to further stigmatization of varieties of English not represented in the model.
* In a related but more subtle scenario, a researcher who consistently uses this model for under-represented models may develop an implicit association between varieties of speech that weren't trained on and lower performance on speech tasks. This association may lead to subconscious bias against individuals with these speaking pattern.

## Limitations

* The number of phones in the model is difficult to change, so encountering additional phonemic contrasts like for individuals who have not merged /ɑ/ and /ɔ/ will be difficult to contend with without re-training the model or attempting to resolve this issue at the transcription level.
* The program using the model also does not yet have a bespoke user-facing training interface. A technologically sophisticated user could, of course, re-train the model using TensorFlow since the model artifact is provided. But, the requirement of knowledge of using TensorFlow may be enough of a hurdle to prevent some target users from being able to retrain or fine-tune the model.

## Recommendations

We believe the best practice is to fine-tune the model's acoustic model on at least a subset of the data that is to be aligned. For projects where the aligned data is for a language that has a very different phone inventory than English, fine-tuning with a new output layer or training an entirely new model from scratch are more likely to provide better solutions and reduce bias in the model than using it as-is.

# Training details

The model was trained to predict phones given acoustic vectors.

## Training data

The training data were composed of two sources:

1. The TIMIT speech corpus, which strictly contains sentence read aloud.
2. The Buckeye speech corpus, which contains spontaneous speech collected during sociolinguistic interviews.

Both of these data sets are phonetically transcribed and segmented, and the framewise training targets were determined from the transcription and segmentation. We note that speech data like these should not necessarily be considered to have an objective ``gold standard'' to compare against. Individual transcribers do not always agree, and because speech is continuous, there are infinite locations where a boundary could be placed that would be considered to be acceptable to many phoneticians.

## Training procedure

During training, the model optimized a categorical cross-entropy loss function, comparing the network's softmax output to a one-hot vector that indicated the ``target'' phone to output. The optimizer used was Adam, using the default values in Keras.

The model was trained for 50 epochs, and the epoch that had the highest associated validation accuracy was taken as the best version of the model.

The model was trained 10 separate times with separate random weight initializations so as to be able to provide estimates of uncertainty on the statistics and metrics being reported.

### Preprocessing

The training data were pre-processed in the 39 acoustic features as described above (12 MFCCs plus the log frame energy replacing the 0th coefficient, and delta and delta-delta coefficients associated with the MFCCs). The TIMIT sentence saved as entire training example, but the Buckeye data were pre-processed into phrases based on the intervals the annotators gave for phrase- and sentence-level boundaries.

The target phones ultimately were collapsed into a smaller subset that merges acoustically similar categories, such as merging the closure period for \[p, t, k\] into just a category for the stop that does not separate the closure from the release.

5% of the training data was randomly selected to be held out as a validation set during training. Speaker 4 from the Buckey data set was also held out for the validation set.

### Size

The model has 977,803 parameters.

# Evaluation

The model was evaluated both as a phone recognizer and within the context of a forced aligner.

## Testing data

The testing data consisted of the test data set from TIMIT and the phrases from speakers 27, 38, 39, and 40 from the Buckeye corpus.

## Factors

Based on the representativeness of the data and research on sociophonetic variation, the relevant social factors likely to cause performance variation in the model are 1) variety or dialect of English and 2) linguistic expressions and indexes of race, ethnicity, sex, gender identity, and sexual orientation. In addition, the intersections of these factors with each other and age are likely to be relevant factors.

## Metrics

The English model here was evaluated in terms of prediction accuracy. The downstream forced alignment was evaluated in terms of the absolute error of the boundaries placed during the alignment process.

## Results

The mean accuracy of the 10 runs of the model on each data set are given in the following table. The accuracy is presented as the average over the 10 runs, plus or minus 1.96 times the standard error to provide a 95% confidence interval.

Data set   | Accuracy (%)
-----------|--------------------
Train      | $0.74 \pm$ 0.008$
Validation | $0.73 \pm$ 0.002$
Test       | $0.71 \pm 0.002$

The forced alignment evaluation involved evaluating whether the alignment scores should be interpolated or not. The results for the mean and median absolute boundary error are prsented in the following table. Both the average of the 10 runs and a 95\% confidence intervals are presented.

Data set   | Interpolation | Mean abs. error (ms) | Median abs. error (ms)
-----------|---------------|----------------------|------------------------
Train      | Yes           | $15.45 \pm 0.09$     | $6.59 \pm 0.05$
Validation | Yes           | $14.82 \pm 0.15$     | $6.82 \pm 0.05$
Test       | Yes           | $17.80 \pm 0.18$     | $7.31 \pm 0.06$
Train      | No            | $16.55 \pm 0.07$     | $7.97 \pm 0.09$
Validation | No            | $15.90 \pm 0.17$     | $8.21 \pm 0.11$
Test       | No            | $18.91 \pm 0.2$      | $8.68 \pm 0.14$

# Environmental impact

We did not track the number of GPU hours we spent during the development of this model, but we will provide our best estimates, within an order of magnitude, of the training time. We note that we trained the model locally on a computer located at the University of Alberta in Edmonotn, Alberta, Canada.

We alternately used a Titan X Pascal and a Titan V, depending on availability. For simplification, we will assume using just the Titan X Pascal (and checking the calculator results, it seems that these cards are treated having an equal level of efficiency).

Using the calculator at \url{https://mlco2.github.io/impact/\#compute:

Using the calculator at [https://mlco2.github.io/impact/#compute](https://mlco2.github.io/impact/#compute):

Variable | Value
---------|---------
Hardware type | Titan X Pascal
Hours used    | 1000
Provider      | Private infrastructure
Carbon efficiency (kg/k/Wh) | 0.9
Offset bouth    | 0
Carbon emitted (kg CO_2) | 225

We note that the carbon efficiency factor was sourced from [Carbon Footprint's factor for Alberta](https://www.carbonfootprint.com/docs/2018_8_electricity_factors_august_2018_-_online_sources.pdf), as linked to on the calculator website.


# Technical specifications}

The model consisted of 3 bidirectional LSTM layers with 128 units each, and they were connected to a fully-connected layer with 61 output units that finished with a softmax activation. The model optimized categorical cross-entropy as its objective function, comparing the posterior distributions of the network to one-hot target vectors.

# Citation

Kelley, M. C., Perry, S. J., \& Tucker, B. V. (Submitted). The Mason-Alberta Phonetic Segmenter: A forced alignment system based on deep neural networks and interpolation.

# Model card authors

Matthew C. Kelley, Scott James Perry, Benjamin V. Tucker

# Model card contact

\href{mailto:mkelle21@gmu.edu}{Mattthew C. Kelley}

# How to get started with the model

The best way to get started with the model is to use the downstream MAPS aligner: [https://github.com/MasonPhonLab/MAPS](https://github.com/MasonPhonLab/MAPS).

