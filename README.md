## birdclef-2023-identify-bird-calls-in-soundscapes
## score at 5th position is achieved.
![birdclef-submission](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/1dd6c8fe-2956-4ebf-a6fd-2c67bfa7fb1e)

### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. collect_all_ogg.ipynb
3. train.ipynb
4. birdclef-submission.ipynb

<b>Code has been explained in the files "modules/preprocess.py", "modules/model.py" and "modules/dataset.py" along the code.</b>

### Dataset
-----
Download the audios from [BirdCLEF2023](https://www.kaggle.com/competitions/birdclef-2023/data), [BirdCLEF2022](https://www.kaggle.com/competitions/birdclef-2022/data), [BirdCLEF2021](https://www.kaggle.com/competitions/birdclef-2021/data) and [Extended BirdCLEF2020](https://www.kaggle.com/competitions/birdclef-2023/discussion/398318).

Put all the audios to ./inputs/train_audios/ using <b>collect_all_ogg.ipynb</b>.

Metadata(train.csv) can be downloaded from [here](https://www.kaggle.com/datasets/narender129/birdclef-complete-metadata).

Download the background-noise audios from [here](https://www.kaggle.com/datasets/honglihang/background-noise) and put all the audios to ./inputs/background_noise/.

### Training
-----
![sed_model](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/3c33fe75-9a2c-4817-8930-1c594c7e65a4)

Backbones which have been used in SED (sound event detection) models are:<br>
1. tf_efficientnetv2_s_in21k
2. seresnext26t_32x4d
3. tf_efficientnet_b3_ns
All of them are trained on 10sec clip.

For SED model, input has a size  [BS, 1 , 320000] i.e., audio file is sampled at the rate of 32KHz for 10 seconds. During validation the duration for sample collection has been kept to 60 seconds. The resulting length has been further broken down into shapes of (1,320000) and adjusted to batch size.

In the case of BCEWithLogitsLoss; input, target and output all have the same shape i.e., [BS, 834]. Output has been summed in the dimension 1 to get the shape [BS,].

![cnn_model](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/00f70093-cf1d-47e1-953f-d7777e9d8b24)

Backbones which have been used in CNN models are:<br>
1. tf_efficientnetv2_s_in21k
2. resnet34d
3. tf_efficientnet_b3_ns
4. tf_efficientnet_b0_ns
All except b0 are trained on 15sec clip. b0 is trained on 20sec clip.

Each model (out of 7) has gone through following stages are:
1. Pretrain on all data (834 species) with CrossEntropyLoss.
2. Pretrain on all data with BCEWithLogitsLoss using step 1 weights.
3. Train on 2023 species (264 species) with CrossEntropyLoss using step 2 weights.
4. Train on 2023 species with BCEWithLogitsLoss using step 3 weights.
5. Finetune (frozen backbone and certain layers) on 2023 species with BCEWithLogitsLoss using step 4 weights.

During training on 264 species, we have popped out those layers from model state_dict (which have size issues with 864 species).

Model converges faster with CrossEntropyLoss than BCEWithLogitsLoss, but BCEWithLogitsLoss gives better score. So, first get out of local minima then slowly reach the global minima.

Pseudo-labeling has been avoided as it not improved the accuracy.

Sklearn average precision score is computed as follows:
![sklearn_avg_precision_score](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/7da1b47b-a9dd-4160-8f28-1f499cc67706)

#### Audio in Time and Frequency Domain
-----
When a computer records digital audio, it measures the sound pressure level multiple times per second. These measurements are often called samples. Being digital, the samples are quantized -- that is, they can only take on certain discrete values as compared to the continuous range of possible values in the actual analog sound wave. 

The rate at which we sample the data can vary, but is most commonly 44.1kHz, or 44,100 samples per second. 
![audio_in_time](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/e2e19d66-348f-48e0-b46e-d4480c45555a)
[Image Reference](https://elvers.us/perception/soundWave/)


The FFT (fast fourier transform) is a widely used algorithm for transforming time-domain audio signals into the frequency domain.
![fft](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/e9e1391b-0288-4d44-a2da-11ba3a8ee038)
[Image Reference](https://mriquestions.com/fourier-transform-ft.html)

#### What is MelSpectrogram?
-----
The FFT (fast fourier transform) is computed on overlapping windowed segments of the signal, and we get what is called the spectrogram. A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale. The Mel Scale is a logarithmic transformation of a signal's frequency. The core idea of this transformation is that sounds of equal distance on the Mel Scale are perceived to be of equal distance to humans.

In a nutshell, a signal goes through a pre-emphasis filter; then gets sliced into (overlapping) frames and a window function is applied to each frame; afterwards, it do a Fourier transform on each frame (or more specifically a Short-Time Fourier Transform) and calculate the power spectrum; and subsequently compute the filter banks. 

![spectrogram](https://github.com/bishnarender/birdclef-2023-identify-bird-calls-in-soundscapes/assets/49610834/b0bdc37f-e8d0-4f01-95eb-b92189e91d1c)
[Image Reference](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

[Link to Article 1](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
[Link to Article 2](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

#### How sklearn computes label_ranking_average_precision_score?
-----
Label ranking average precision (LRAP) averages over the samples the answer to the following question: for each ground truth label, what fraction of higher-ranked labels were true labels? This performance measure will be higher if you are able to give better rank to the labels associated with each sample. The obtained score is always strictly greater than 0, and the best value is 1.

The sklearn label_ranking_average_precision_score metric is:

[1/num_samples] * [sum of per-sample average precisions]

where,<br>
per-sample average precision = [1/num_positives_among_labels] * sum_over_positive_labels ( [L in true/positive labels for this label] / [rank in all labels for this label] ). 

L is rank among only positive/true/relevant labels.

<b>Let understand it with an example.</b>

y_true = np.array([[1, 0, 0], [0, 1, 1]])<br>
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])

For the first sample (y_true[0] and y_score[0]): Predictions are ranked as [2, 3, 1] for the sample [1, 0, 0] since the lower scores are considered better for ranking.

Since there is only one true/positive label in the sample. num_positives_among_labels=1. And L=1 for this single label i.e., relevant rank is 1.
The average precision for the first sample is (1/1)(1/2) = 0.5.

For the second sample (y_true[1] and y_score[1]): Predictions are ranked as [1, 2, 3] for the sample [0, 1, 1] since the lower scores are considered better for ranking.

Since there is 2 true/positive label in the sample. num_positives_among_labels=2. 

relevant_labels = [NA,1,1].<br>
relevant_ranks = [NA,1,2]

The average precision for the second sample is (1/2)[(1/2)+(2/3)] = 0.58.

“label ranking average precision score” = (0.5 + 0.58) / 2 = 0.54.
