# EEGdecodingproject
REPORT

EEG Decoding!
Anandita Garg, Chinmayi Kalapatapu, Suhaani Sachdeva, Uma Gaba
We picked up the EEG Challenge (2025) for our Deep Learning project. This had 2 sub tasks:
Cross-Task Transfer Learning: Developing models that can effectively transfer knowledge from any cognitive EEG tasks to active task
Subject Invariant Representation: Creating robust representations that generalize across different subjects while predicting clinical factors.

<img width="1138" height="838" alt="unnamed" src="https://github.com/user-attachments/assets/e1cbbe03-a85b-4bda-a7ba-a2e2ee388a26" />

To go about this, we divided ourselves into pairs. Anandita and Suhaani did Task 1, Chinmayi and Uma did Task 2.
Our dataset consisted of EEG recordings collected from participants across six distinct cognitive tasks. Each recording contained 129 channels and was segmented into 2-second windows with a 0.5-second offset, sampled at 100 Hz. In addition to the EEG data, we had access to demographic information, including age, sex, and handedness. The dataset was organized into 11 full releases, each containing data from 200 participants, along with corresponding mini-release subsets, each comprising data from 20 participants.
Task 1
Task one was based on cross-task transfer learning. We had to predict behavioral performance metrics (response time) on active tasks, using transfer learning from passive tasks data.The motivation behind this, as mentioned on the official page was that EEG decoding faces significant challenges due to signal heterogeneity from various factors like non-stationarity, noise sensitivity, inter-subject morphological differences, varying experimental paradigms, and differences in sensor placement. While recent advances in machine learning have shown promise, there remains a critical need for models that can generalize across different subjects and tasks without expensive recalibration. For our task we used the mini-releases as our dataset.


Passive Tasks Used: 
Resting State (RS): Eyes open/closed conditions with fixation cross
Surround Suppression (SuS): Four flashing peripheral disks with contrasting background
Movie Watching (MW): Four short films with different themes
Active Task Used: Contrast Change Detection (CCD): Identifying dominant contrast in co-centric flickering grated disks
We had to pretrain and then finetune our model. For pretraining, we used the Masked Self Supervised technique using an encoder-decoder model. In this:
The model trains as an autoencoder on a Masked Signal Modeling task using passive EEG data.
A random time column (all channels at a single time point) is zeroed out in the input.
The Encoder-Decoder reconstructs the missing column based on the surrounding signal.
This teaches the GCN to extract robust, spatially-aware features, preparing it for the final supervised task.
Then we moved on to fine tuning.
Through literature review, we found that there are multiple architectures used for similar tasks in literature, ranging from basic MLPs and SVMs to GCN. Future innovation can also try Transformers/Attention.  We tried 2 model architectures: Vanilla CNNs and GCNs. Our results are as follows:
<img width="1265" height="222" alt="Screenshot 2025-12-03 214245" src="https://github.com/user-attachments/assets/dd84f619-0dcf-4984-a082-8606337953b4" />


Task 2
Task two is a supervised regression challenge involving the prediction of the externalizing factor, a psychopathological factor (externalizing) value from EEG data for an individual. The motivation for this task is to address the gap in current methods, self reported surveys,  employed to calculate such factors. To eliminate subjectivity and bias from this method, using EEG data to predict such factors opens up the possibility to create a more objective manner of evaluation. 
Through literature review and running of baseline codes, we identified that the top 3 architectures for this task were EEGNet, EEgNex, and EEGConformer out of which EEGNet gave the lowest RMSE. Following this, we tried to build upon the EEGNet architecture to introduce a novel architecture for this unexplored domain. The first change that was implemented was to convert the standard EEGNET version 4 to handle regression tasks over classification tasks. This involved adding a regression head to replace the softmax layer. Thus our baseline model was EEGNet version 4 adapted to regression tasks and trained on one mini release of data
Through a rigorous literature review, we learned about multi-head fusion and attention mechanisms that enable more efficient extraction of multi-scale features. We also explored incorporating the demographic data provided in the competition. The dataset includes information on participants’ sex, age, and handedness. However, our research identified reliable evidence of correlations only between age, sex, and EEG signals. As a result, we chose to include only these two demographic features and excluded handedness from our model.
Post establishing the baseline, the versions that we explored for our model  and their respective RMSE are as follows:


<img width="1278" height="600" alt="Screenshot 2025-12-03 214338" src="https://github.com/user-attachments/assets/34845db9-f437-4513-b8d3-3494214ac4a1" />

As seen from our results, nearly all versions yielded similar RMSE values. Although every version after Version 1 showed improvement, we were unable to achieve any further reduction in RMSE beyond Version 2. While several factors may have contributed to this, the primary bottleneck was the limited compute available to us. Due to these constraints, we were only able to train our model on one of the six tasks, and we were restricted to using the mini-release subsets of the dataset. Consequently, our final accessible dataset included data from only 20–60 participants out of the full set of 3,000.


Doing this project was a great learning experience for us. While we found it very hard and struggled a lot, we got an idea about dealing with EEG data as well as AI Biological applications. We struggled a lot with the server and resource constraints, due to which we were not able to experiment quite as much as we wanted so we aim to do that in the future. We express our gratitude to Professor Anupam Sobti and all the TAs of the course for their availability, timely support and encouragement.




OUR VIDEO LINK: https://plakshauniversity1-my.sharepoint.com/personal/uma_gaba_ug23_plaksha_edu_in/_layouts/15/stream.aspx?id=%2Fpersonal%2Fuma%5Fgaba%5Fug23%5Fplaksha%5Fedu%5Fin%2FDocuments%2Fdeep%5Flearning%5Fvideo%2Emp4&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&ga=1&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E79e47e80%2D5efe%2D4347%2D8041%2D04e22e89f7a2&isDarkMode=true
