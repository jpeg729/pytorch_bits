# pytorch-bits

Experiments for fun and education. Mostly concerning time-series prediction.

I started my experiments with [@osm3000](https://github.com/osm3000)'s [sequence_generation_pytorch](https://github.com/osm3000/sequence_generation_pytorch/) repo and some of that code still subsists in these files.

## How to run these experiments

1. clone/download this repo
1. `pip install -r requirements.txt`
1. `python experiment.py [ARGS]`

Possible arguments include...
* `--data_fn FN` where FN is one of the data generation functions listed below
* `--add_noise` to add noise the generated waveform
* `--length TIMESERIES_LENGTH`
* `--batch_size BATCH_SIZE`
* `--seq_len SEQ_LEN` the subsequence length used in training
* `--epochs MAX_EPOCHS`
* `--lr LR`
* `--layers LAYERTYPE_SIZE [LAYERTYPE_SIZE ...]` see the section on Model generation
* `--sigmoid REPLACEMENT` to use an alternative to sigmoid, this can be of the activations mentioned below, e.g. `ISRU_sigmoid`, or any function from torch.nn.functional
* `--tanh REPLACEMENT` to use an alternative to tanh, this must be one of the activations mentioned below, e.g. `ISRU_tanh`, or any function from torch.nn.functional
* `--warmup WARMUP` do not use the loss from the first WARMUP elements of the series in order to let the hidden state warm up.
* `--verbose`

## Data generation

* `sine_1` generates a sine wave of wavelength 60 steps
* `sine_2` overlays `sine_1` with a sine wave of wavelength 120 steps
* `sine_3` overlays `sine_2` with a sine wave of wavelength 180 steps
* `mackey_glass` generates a Mackey-Glass chaotic timeseries using the [signalz](https://matousc89.github.io/signalz/) library
* `levy_flight` generates a Lévy flight process using the [signalz](https://matousc89.github.io/signalz/) library
* `brownian` generates a Brownian random walk using the [signalz](https://matousc89.github.io/signalz/) library

The generator produces a tensor of shape `(length, batches, 1)` containing `batches` independantly generated series of the required length.

## Model generation

The `--layers` argument takes a simplistic model specification. First you specify the layer type, add a "\_", then if the layer type needs a size, you add a number. Then you can follow up with "\_k=value" for any keyword arguments. If the keyword contains "\_" replace it with "-".

For example: `--layers LSTM_50 Dropout_p=.5 CausalConv1d_70_kernel-size=3` specifies a three layer network with 50 LSTM units in the first layer, Dropout with p=.5 as the second layer, and 70 CausalConv1d units with kernel_size=3 in the third layer.

If the output of the last requested layer doesn't match the number of target values (for these experiments the target size is 1) then the script adds a Linear layer to produce the required number of output values.

## Layers

All of these recurrent layers keep track of their own hidden state (if needed, the hidden state is accessible via the `hidden` attribute). They all have methods to `reset_hidden()` and to `detach_hidden()`. 

`reset_hidden()` should be used before feeding the model the start of a new sequence, and `detach_hidden()` can be called in-between batches of the same set of sequences in order to truncate backpropagation through time and thus avoid the slowdown of having to backpropagate through to the beginning of the entire sequence.

Moreover they all take input of shape `(seq_len, batch_size, features)`. This allows vectorising any calculations that don't depend on the hidden state.

* LSTM - the typical LSTM adapted from the PyTorch source code for LSTMCell.
* GRU - the typical GRU adapted from the PyTorch source code for GRUCell.
* MGU and variants - from [arxiv:Minimal Gated Unit for Recurrent Neural Networks](https://arxiv.org/abs/1603.09420) and simplified in [arxiv:Simplified Minimal Gated Unit Variations for Recurrent Neural Networks](http://arxiv.org/abs/1701.03452). I have only coded the original MGU and MGU2 variant because they say it is the best of the three.
* RAN - the [arxiv:Recurrent Additive Network](http://arxiv.org/abs/1705.07393)
* SRU - the Simple Recurrent Unit from [arxiv:Training RNNs as fast as CNNs](http://arxiv.org/abs/1709.02755v3). They provide a cuda optimised implementation. This is a simplistic implementation that optionally vectorises the calculation of the gates. The test_sru_optimisation.py script tests the speed of the optimisation when running on the cpu. The difference is negligible.
* CausalConv1d - a wrapper for Conv1d that permutes the input shape to that required by Conv1d, and adds the padding that ensures that each timestep sees no future inputs.
* QRNN - an unoptimised implementation of [arxiv:Quasi-recurrent neural networks](http://arxiv.org/abs/1611.01576v2) The paper makes the QRNN seem rather complex, but when you write out the step equations you see that it is not very different from most other RNNs.
* TRNN - a strongly typed simple RNN from [arxiv:Strongly-Typed Recurrent Neural Networks](https://arxiv.org/abs/1602.02218)
* Chaos Free Network (CFN) from [arxiv:A recurrent neural network without chaos](https://arxiv.org/abs/1612.06212) h_t = f * tanh(h_tm1) + i * tanh(Wx), where f, i = sigmoid(Wx + Vh_tm1 + b)

### Planned

* Strongly typed LSTM and GRU from [arxiv:Strongly-Typed Recurrent Neural Networks](https://arxiv.org/abs/1602.02218)
* Recurrent Identity Network (RIN) from [arxiv:Overcoming the vanishing gradient problem in plain recurrent networks](https://arxiv.org/abs/1801.06105) They converge faster and achieve better accuracy. The basic idea is to initialise the hidden-hidden weights to be ~= 1 rather than ~= 0.
* Phased LSTM from [arxiv:Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](https://arxiv.org/abs/1610.09513) The hidden-hidden update is gated by a periodic function which lets updates through only a small percentage of the time. This opens the possibility of irregular updates.

### Ideas/research

* I plan to study [arxiv:Unbiased Online Recurrent Optimization](http://arxiv.org/abs/1702.05043), but for the moment it is not clear to me how best to implement it.
* Optional noisy initial hidden states. Otherwise the model will learn to cope with the fact of having zero initial hidden state which may hinder learning the hidden state dynamics later in the sequences. This probably isn't very important if I have only a few sequences that are very long and that are normalised to zero mean.
* The LSTM class in PyTorch builds a configurable number of identically sized LSTM layers. This architecture allows us to calculate W x h_tm1 for all layers in one single operation. I may try adapting the above layers to take advantage of this.
* Sigmoid activation is typically used for gates, but it is not symetric. Lets try using tanh instead [Using tanh instead of sigmoid in RNNs](https://github.com/jpeg729/pytorch_bits/wiki/Tanh-instead-of-sigmoid).

## Optimisers

COCOB is great for quick experiments, it has a near optimal learning rate with no hyperparameter tuning, so you can quickly tell which experiments are going nowhere. However I suspect that it relies too heavily on assumptions of convexity of the loss. Other optimisers often get lower loss after many epochs.
Adam_HD tunes the learning rate of Adam by backpropagating through the update function. It learns pretty fast too.

* COCOB - COntinuous COin Betting from [arxiv:Training Deep Networks without Learning Rates Through Coin Betting](https://arxiv.org/abs/1705.07795)
* Adam_HD - Adam with Hypergradient descent from [arxiv:Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782) I have set the learning rate's learning rate to 0.1 which is much higher than they recommend, but it works well for the experiments I have run.

### Planned

* SGD_HD - SGD with Hypergradient descent from [arxiv:Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782).
* ADINE - ADaptive INErtia from [arxiv:ADINE: An Adaptive Momentum Method for Stochastic Gradient Descent](https://arxiv.org/abs/1712.07424)

### Ideas/research

* Use [arxiv:Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients](https://arxiv.org/abs/1712.06563) to produce random perturbations when training gets stuck. The basic idea involves scaling weight changes according to the gradient of the model's *output* w.r.t. its weights.

## Activations

Note that PyTorch Tanh, Sigmoid and ELU are already very well optimised when run on cpu. My tests show that my simplistic implementation provides little difference when running on cpu.

* ISRLU - [arxiv:Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)](https://arxiv.org/abs/1710.09967) An alternative to ELU that is supposed to be faster to calculate.
* ISRU_tanh - from the same paper. A proposed alternative to tanh.
* ISRU_sigmoid - from the same paper. A proposed alternative to sigmoid.
* Bipolar activation wrapper from [arxiv:Shifting Mean Activation Towards Zero with Bipolar Activation Functions](https://arxiv.org/abs/1709.04054) Bipolar(f, x_i) = f(x_i) if i is even else -f(-x_i). The resulting activation has mean zero. Note: initialise weights so that the layer has variance ~= 1.
* The soft exponential activation from [arxiv:A continuum among logarithmic, linear, and exponential functions, and its potential to improve generalization in neural networks](https://arxiv.org/abs/1602.01321) A learnable parameter allows the model to choose between f(x)=log(x), f(x)=x and f(x)=exp(x). Neural networks can easily add values, but can't easily multiply them. With this activation multiplication becomes easy e.g. exp(log(x_0) + log(x_1)).
* TernaryTanh from [the R2RT blog](https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html) Tanh is essentially a binary function, it has two stable output zones, 1 and -1. TernaryTanh also has a flat area around zero, enabling a model to choose between 1, -1 and 0 output with greater ease.

### Planned

* [arxiv:Noisy activation functions](https://arxiv.org/abs/1603.00391) are versions of saturating activation functions that add noise when the output is in the saturation zones.
  
## Regularisers

### Planned

* DARC1 regularizer from https://arxiv.org/abs/1710.05468
* Spectral norm regulariser from https://arxiv.org/abs/1705.10941