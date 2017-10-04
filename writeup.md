# Behavioral Cloning


#### Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/hist1.png "Original angles"
[image2]: ./examples/hist2.png "Filtered angles"
[image3]: ./examples/hist3.png "Using left/right images"
[image4]: ./examples/sample_1.png "Original Image"
[image5]: ./examples/sample_2.png "Cropped Image"
[image6]: ./examples/sample_3.png "Resized Image"
[image7]: ./examples/demo.gif "Demo"

#### Submission

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Regarding compatibility issues note that Keras 2.0.8 was used to train and save the model.


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###  Model Architecture and Training Strategy

#### Model architecture

The model architecture is similar to the [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model. It was simplified iteratively, input image size is smaller, thus it trains faster. My model consists of a normalization layer, 3 convolutional layers and 3 fully connected layers. The data is normalized in the model using a Keras lambda layer (code line 91). Convolutional and fully connected layers have ELU activation to introduce nonlinearity. Convolutional layers have 5x5 filter sizes and depths of 24, 36 and 48 respectively. 

#### Reducing overfitting

The model contains dropout layers in order to reduce overfitting. After flattening there is an aggressive dropout layer. More moderate dropout layers also follow the first two fully connected layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (splitting samples at code line 56). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### Training data

The [Udacity samples](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) and own recordings were used as training data. Input images were cropped to remove car hood but contain the road ahead until just below the horizon. With the intention to create a kind of forward-looking model to keep the vehicle driving on the road, I only used center lane driving and did not include explicit recovery from the left and right sides of the road. For more details about see the next section. 

### Details

#### Solution Design Approach

The overall strategy for deriving a model architecture was to iterate based on a well-known model. So the first step was to use the convolution neural network model described in [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

I thought this model should be simplified because the simulator platform does not have the same complexity as real-world driving. I was experimenting with reducing convolutional layers and changing their subsampling. Meanwhile output dimension before flattening became too high, it resulted in too many parameters in the first fully connected layer. Then I reduced the input image dimension from `200x66` to `80x35`. Finally there are somewhat fewer model parameters than in the original model (`197179` compared to roughly`250000`) and the network has two less convolutional layers.

In parallel the steering angle dataset was also refined (see later for details) and models were tested on first track. To avoid overfitting and achieve good results I used aggressive dropout policy mainly after flattening, but also after first two fully connected layers.
At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

Brief visualization of autonomous driving on track one:

![alt text][image7]

#### Final Model Architecture

The final model architecture (model.py lines 89-103) consists of the following layers:

|   Layer (type)        | Output Shape            |     Parameters |  
| :--------------------:| :----------------------:| :------------: | 
|   Lambda normalization| (None, 35, 80, 3)       |     0          |  
|   Conv2D 5x5          | (None, 16, 38, 24)      |     1824       |   
|   Conv2D 5x5          | (None, 6, 17, 36)       |     21636      |   
|   Conv2D 5x5          | (None, 2, 13, 48)       |     43248      |   
|   Flatten             | (None, 1248)            |     0          |  
|   Dropout (0.2)       | (None, 1248)            |     0          |   
|   Dense               | (None, 100)             |     124900     |   
|   Dropout (0.6)       | (None, 100)             |     0          |   
|   Dense               | (None, 50)              |     5050       |  
|   Dropout (0.8)       | (None, 50)              |     0          |   
|   Dense               | (None, 10)              |     510        |   
|   Output              | (None, 1)               |     11         |   
|                       |                         |                | 
|   Total params:       | | 197,179 |

#### Creation of the Training Set & Training Process

The [Udacity samples](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) plus own recordings were used as training data. At first to capture good driving behavior, I started recording on track one using center lane driving for about two lapse. Looking at all the records center lane driving was obviously dominant in dataset.

![alt text][image1]

As a result the trained model has missed event the first turn. Getting rid of all near zero angles produced the opposite outcome, immediate slight steering to the side. The solution was to filter most of the zero angles leading to the angle distribution below.

![alt text][image2]

To make use of the left and right camera images I used them with `0.25` and `-0.25` angle corrections augmenting the dataset to 3x of its size. Flipping images would have been the next step to further augmentation (and helping better generalization) but at this point model performance was good enough for driving autonomously on track one.

![alt text][image3]

Proper cropping was another key point to achieve good autonomous driving behavior. Top part of images has no relevant information, sure. Cropping at `64` just below the horizon seemed like the right choice at first. But increasing the cut to `70` gave better results (not missing the second left turn particularly). Here is an example image of center lane driving at original size, cropped and resized versions.

![alt text][image4]

![alt text][image5]

![alt text][image6]

Udacity samples has already contained driving in reverse direction. 

At this stage iterating the model in combination with this dataset gave good results on track one which was the target. Of course this is still a very limited dataset that does not enable proper generalizing meaning that track two is out of the question right now. That would require using both tracks, possibly recovery recordings and further data augmentation using small distortions. 

So after the collection process I had`10383` steering angles, after filtering `5373` remained which was then augmented to dataset size of `16119`. Shuffling and setting aside 20% for validation the actual training set size resulted in `12895` data points. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 5. I used an Adam optimizer so that manually training the learning rate wasn't necessary. Preprocessing step was used as follows:
```python
def preprocess_image(img):
    # top is cropped just below horizon, bottom car hood is also unnecessary
    img = img[70:140,:,:]
    # resized image still contains enough information
    img = cv2.resize(img,(80, 35), interpolation = cv2.INTER_AREA)
    # YUV color space is advised to be used in Nvidia paper
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img
```