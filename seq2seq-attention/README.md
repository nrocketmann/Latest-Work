# Attention Networks for Titling BBC Articles #
Here, I implemented a few approaches from this paper (https://arxiv.org/pdf/1508.04025v5.pdf)
along with some of my own to generate titles for BBC articles. I also adjusted the model architecture
several times in an attempt to execute the model as a TensorFlow graph rather than in eager execution.
Unfortunately, TensorFlow was only able to take gradients of the model in eager mode, which drastically
slowed training times.
