# MYSTnet

A Convolutional Neural Network for Identifying D'ni Numerals :1234:

## Overview

### Background

The society of D'ni exists in the Myst series Canon, and are a race gifted with the ability to create portals to any worlds that could possibly exist. These portals come in the form of books (called _Linking Books_), and with them comes an entire language similar in depth to Lord of the Rings.

Their number system is base-25, which means that the ones-place has characters for zero through twenty four. The next digit place represents quantities of 25, then 625, and so on.

![Dni Numbers](https://raw.githubusercontent.com/kylehovey/myst-net/master/dni-numbers.png)

### Generating Data

The hardest part about this project was generating the data necessary to train a network that could classify twenty-five separate symbols. My original plan was to hand-write two hundred examples of each digit for a total of 2500 data samples (which I would then have to split into training and validation sets). This would have taken hours, if not days, and would have made this not a short-term viable project.

I instead broke down the number system into atomic parts that I could compose together in different combinations. Most characters could be broken down into three parts, but some only could be broken into two. I ended up writing twenty examples of each component, meaning that I could have at least four hundred generated samples of each character making a total of five thousand procedurally generated data points.

**Example Atomic Segment:**

![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/20/dni_numeral_0000s_0000_Layer-180.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/bracket/dni_numeral_0009s_0009_Layer-11.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/18/dni_numeral_0000s_0011_Layer-209.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/2/dni_numeral_0006s_0008_Layer-52.png)

**Example Generated Forms:**

![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/20/3.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/20/8.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/18/10.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/3/3.png)

To combine the data, I added a linear combination of each component image to the other, then used OpenCV to threshold on the values so that everything was either black or white (in hopes of generating higher contrast data). Each image is `200x150` pixels.

### Network Architecture

I tested numerous architectures on this data and ended up using a network preceded by two convolutional layers (with pooling in-between), followed by two fully connected layers ending in 25 output softmax neurons. Each of the convolutional and fully-connected layers used a `ReLU` activation function.

## Accuracy

:ok_hand:

|         | Test Accuracy | Valid Accuracy |
|---------|---------------|----------------|
| MYSTnet | 99.93%        | 99.48%         |

## Running The Code

### Dependencies

This project uses only packages that we used in the USU CS5600 Intelligent Systems course, so if you have a Python virtual environment set up that works for the `tflearn` projects that we worked on in class, then you should be good-to-go. In case you run into any problems, here is a basic run-down of the packages that I import in this project (everything uses Python `2.7.10`, which I run on OS X `10.13.6`):

#### Packages (top-level deps):
* `OpenCV` (as `cv2`)
* `numpy`
* `tflearn`

##### Packages (all deps in virtual environment):
* absl-py (`0.5.0`)
* astor (`0.7.1`)
* backports.functools-lru-cache (`1.5`)
* backports.weakref (`1.0.post1`)
* cycler (`0.10.0`)
* enum34 (`1.1.6`)
* funcsigs (`1.0.2`)
* futures (`3.2.0`)
* gast (`0.2.0`)
* grpcio (`1.15.0`)
* h5py (`2.8.0`)
* Keras (`2.2.2`)
* Keras-Applications (`1.0.4`)
* Keras-Preprocessing (`1.0.2`)
* kiwisolver (`1.0.1`)
* Markdown (`3.0.1`)
* matplotlib (`2.2.3`)
* mock (`2.0.0`)
* numpy (`1.15.1`)
* opencv-python (`3.3.0.10`)
* pbr (`4.3.0`)
* Pillow (`5.3.0`)
* protobuf (`3.6.1`)
* pyfiglet (`0.7.6`)
* pyparsing (`2.2.2`)
* python-dateutil (`2.7.3`)
* pytz (`2018.5`)
* PyYAML (`3.13`)
* scipy (`1.1.0`)
* six (`1.11.0`)
* subprocess32 (`3.5.3`)
* tensorboard (`1.12.0`)
* tensorflow (`1.12.0`)
* tensorflow-hub (`0.1.1`)
* tensorflowjs (`0.6.7`)
* termcolor (`1.1.0`)
* tflearn (`0.3.2`)
* Werkzeug (`0.14.1`)

### Run Via Script

I have provided a basic shell script that you may use on an image. Note that the image must be `250x100` pixels in size, as the script does no re-sizing or re-scaling. I have included a template file `test.png` in the root of this repo that is the correct size. If you want, you can open it up with your photo editing software of choice and write a D'ni number in it. Once saved, you can detect what number is in the image by running:

```bash
./classify.py test.png
```

This will build the network and attempt to classify the digit written in `test.png`. Example output:

```
Building Network...
Loading Network Weights...
libpng warning: iCCP: known incorrect sRGB profile
Reading Image
Classifying

 .d8888b.
d88P  Y88b
888    888
888    888
888    888
888    888
Y88b  d88P
 "Y8888P"
```

### Run Unit Tests

I have included a file that runs unit tests for all of the code from reading images to running the network. They also serve as basic documentation so that you may see how the network data is assembled and consequently how the network is run. You can run these tests by running the `unit_tests.py` file. Here is an example output with mixed results (note that all tests are passing on my machine, I just included these "failed" outputs so you can see what it would look like if it did fail):

```
7 tests to run
<==================>
Running test 1 of 7: Image Loading
It loads images from the filesystem: 👍
-------------------
Running test 2 of 7: Image Processing
It processes a loaded image for the network: 👍
-------------------
Running test 3 of 7: Data Loading
It loads and formats all training and validation data: 👍
-------------------
Running test 4 of 7: Network Loading
It loads the network itself: 👍
-------------------
Running test 5 of 7: Network Running
It runs the network on input data: 👍
-------------------
Running test 6 of 7: Testing Accuracy
It is at least 95% accurate on testing data: 🌩
-------------------
Running test 7 of 7: validation Accuracy
It is at least 95% accurate on validation data: 🌩
-------------------
```

If all goes well though, you should get the same output I get on my machine, i.e. all tests run and pass:

```
7 tests to run
<==================>
Running test 1 of 7: Image Loading
It loads images from the filesystem: 👍
-------------------
Running test 2 of 7: Image Processing
It processes a loaded image for the network: 👍
-------------------
Running test 3 of 7: Data Loading
It loads and formats all training and validation data: 👍
-------------------
Running test 4 of 7: Network Loading
It loads the network itself: 👍
-------------------
Running test 5 of 7: Network Running
It runs the network on input data: 👍
-------------------
Running test 6 of 7: Testing Accuracy
It is at least 95% accurate on testing data: 👍
-------------------
Running test 7 of 7: validation Accuracy
It is at least 95% accurate on validation data: 👍
-------------------
```

If, for some reason, your machine does not like the emoji characters in these unit tests, you can try running the equivalent `unit_tests_ascii.py` file that I have included, which omits the emoji and instead just uses ASCII characters.

## Deliverables

### Proposed:

* Large dataset of handwritten D'ni numerals (did not exist before):
  * All in the `data` folder of this project (also can be generated using source code in `make_data.py`)
* Source code for a convolutional network that can classify D'ni numerals into base-10
  * See the `net.py` for the architecture and the previous instructions
* ~~(Stretch Goal) Source code that abstracts the second deliverable to take multi-numeral numbers and convert it to base-10~~

Note that in the original proposal, I had two entries for source code: one for a numeral identifier, and one for a convolutional network. This was either a typo (I forgot to delete one), or me expressing that I would provide source code for a non-convolutional network if the convolutional option did not pan out. Seeing as the convolutional option worked with flying colors, it is my main deliverable.

I was not able to achieve my stretch goal of a network that could recognize multi-digit lines. I am glad I made this a stretch goal, because I gave it a solid try and it became a nearly impossible task. My method was to use OpenCV to break apart each digit into each of their rectangular digits. The problem here is that the cells may be of varying aspect ratio, and the `1` component of the symbols can add another vertical separator that can make it confusing for an algorithm to effectively separate each digit out from the number:

![Multi Numeral](https://raw.githubusercontent.com/kylehovey/myst-net/master/multi-numeral.png)

### Conclusion:

Even though I did not achieve this stretch goal, I am incredibly happy with the result.Procedurally generating this data was a great challenge, and I am happy that the data I generated reflects digits that I actually write with Photoshop. It was also pleasing bringing down the amount of samples I had to provide from `5000` to `220` atomic symbols. What is even cooler is that I can extend this approach to any symbol set that can be broken down in this way and use this exact same code to build a net that can recognize samples from that symbol set. Perhaps this code will come in handy for future games.
