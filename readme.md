# MYSTnet

## A Convolutional Neural Network for Identifying D'ni Numerals

---

### Overview

The society of D'ni exists in the Myst series Canon, and are a race gifted with the ability to create portals to any worlds that could possibly exist. These portals come in the form of books (called _Linking Books_), and with them comes an entire language similar in depth to Lord of the Rings.

Their number system is base-25, which means that the ones-place has characters for zero through twenty four. The next digit place represents quantities of 25, then 625, and so on.

![Dni Numbers](https://raw.githubusercontent.com/kylehovey/myst-net/master/dni-numbers.png)

### Generating Data

The hardest part about this project was generating the data necessary to train a network that could classify twenty-five separate symbols. My original plan was to hand-write two hundred examples of each digit for a total of 2500 data samples (which I would then have to split into training and validation sets). This would have taken hours, if not days, and would have made this not a short-term viable project.

I instead broke down the number system into atomic parts that I could compose together in different combinations. Most characters could be broken down into three parts, but some only could be broken into two. I ended up writing twenty examples of each component, meaning that I could have at least four hundred generated samples of each character making a total of five thousand procedurally generated data points.

**Atomic Segments:**

![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/20/dni_numeral_0000s_0000_Layer-180.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/bracket/dni_numeral_0009s_0009_Layer-11.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/18/dni_numeral_0000s_0011_Layer-209.png)
![Segment](https://raw.githubusercontent.com/kylehovey/myst-net/master/components/2/dni_numeral_0006s_0008_Layer-52.png)

**Generated Forms:**

![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/20/3.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/20/8.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/18/10.png)
![Form](https://raw.githubusercontent.com/kylehovey/myst-net/master/data/3/3.png)
