# Study_Monk_Task
# Task-1: Image Compression without affecting size or quality
## Image compression using k-means clustering
As discussed earlier in this post, image compression, in some techniques, involves reducing the color components of the image. With k-means clustering, this is what we’re doing.
We pre-define the value of k as the number of color components that we want to preserve in the image. The rest of the k-means algorithm is performed according to the above-mentioned steps.
With an increase in the value of k, as the number of clusters increases, the image will get closer and closer to the original image, but at the cost of more disk space for storage and a higher computational cost. We can experiment with the values of k to get desirable results.
We can also calculate the within-cluster sum of squared error to gain insight on whether the clusters are well fitted and correctly assigned or not, since it provides us with the variance of the cluster centroids.
Note: It’s advised to keep the value of k as a multiple (more preferably, a power) of 2, same as the conventional image formats, to get better results.
![Output](https://user-images.githubusercontent.com/80475016/163541668-8400e408-4052-4d92-b616-a175ce456ca8.png)
[Demo] Image Compression with kmeans 64 and 32 Clusters  
<img src='Image_Compressor/demo_working.mp4' title='Emotion' style='max-width:600px'></img>  
# Task-2: Prediction of valence and arousal levels from real time video using deep learning techniques
[Demo] Discrete Emotion + Continuous Valence and Arousal levels      
<img src='Emotion_detetction/demo_working.gif' title='Emotion' style='max-width:600px'></img>  
## Testing the pretrained models

The code requires the following Python packages : 

```
  Pytorch (tested on version 1.2.0)
  OpenCV (tested on version 4.1.0
  skimage (tested on version 0.15.0)
```

We provide two pretrained models : one on 5 emotional classes and one on 8 classes. In addition to categorical emotions, both models also predict valence and arousal values as well as facial landmarks.

To evaluate the pretrained models on the cleaned AffectNet test set, you need to first download the [AffectNet dataset](http://mohammadmahoor.com/affectnet/). Then simply run : 

```
  python test.py --nclass 8
```

where nclass defines which model you would like to test (5 or 8).

Please note that the provided pickle files contain the list of images (filenames) that we used for testing/validation but not the image files.

The program will output the following results :

#### Results on AffectNet cleaned test set for 5 classes


```
 Expression
  ACC=0.82

 Valence
  CCC=0.90, PCC=0.90, RMSE=0.24, SAGR=0.85
 Arousal
  CCC=0.80, PCC=0.80, RMSE=0.24, SAGR=0.79
```

#### Results on AffectNet cleaned test set for 8 classes

```
  Expression
    ACC=0.75

  Valence
    CCC=0.82, PCC=0.82, RMSE=0.29, SAGR=0.84
  Arousal
    CCC=0.75, PCC=0.75, RMSE=0.27, SAGR=0.80
```

#### Class number to expression name

The mapping from class number to expression is as follows.

```
For 8 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
5 - Disgust
6 - Anger
7 - Contempt
```

```
For 5 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
```
