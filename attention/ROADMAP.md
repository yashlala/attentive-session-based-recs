# Roadmap

## Architecture Perspective

We can't just "add attention on to GRU4REC". GRU4REC just uses a bunch of
GRU layers in a row, fed into a feedforward network.

All readings about attention involve their use in seq2seq style models.
Readings:

- All You Need is Attention (OG Transformer paper).
- Seq2Seq Paper.

Not much room for attention! We're not trying to map sequence to sequence here!
Unless:

## Hypothesis 1

The GRU4REC paper mentions passing the hidden states from one GRU layer to the
next ones.

> When multiple GRU layers are used, the hidden state of the previous layer is
> the input of the next one.

The Pytorch documentation says this:

> In a multilayer GRU, the input x of the l-th layer is the hidden state
> h of the previous layer multiplied by dropout δ.

Isn't attention basically this (but more complicated version of the dropout
function)? Is this all we have to do to make an attentive GRU4REC?

## Null Hypothesis

A few options:

1. Read out a sequence...that sequence will be the next things they click?
   Unlikely.


## Code perspective

`nn.MultiheadAttention` is great, as is `nn.GRU` -- IF we don't want to tweak
how the hidden layers interact.

If we clearly know what we want from every hidden layers, we can very
effectively use what's in `encoder_decoder.py` -- this implementation makes
every "step" of the encoder + decoder parts visible.

Basically, whatever we do, we've got sources available.
