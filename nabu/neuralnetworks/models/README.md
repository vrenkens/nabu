# Models

A model takes possibly multiple inputs and produces possibly multiple outputs.
In Nabu a model is trained for a specific task by minimizing some loss function.
Every model in Nabu is an encoder-decoder model. An encoder-decoder model
consists of an encoder that encodes the input into some other hidden
representation and the decoder decodes this representation into the desired
output.

You can find more information about encoders [here](ed_encoders/README.md)
and decoders [here](ed_decoders/README.md).

An example of such an encoder-decoder model is the
[Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) model that encodes
the input to a high-level representation with a pyramidal DBLSTM (Listener) and
decodes this high level representation into text characters with an
attention-based rnn-decoder (Speller).

However, not every model can as easily be viewed as an encoder-decoder system.
For example a DBLSTM trained with CTC cannot be devided into to parts in a
straightforward way. In Nabu this system is simply implemented with a DBLSTM
encoder and a linear decoder that maps the encoder output to the desired
output dimension.

Similarily, an rnn language model is implemented with a dummy encoder that does
nothing and an rnn decoder that predicts the next character based on the
history.

You can build new models by combining new encoders and decoders or by creating
new encoders and decoders.
