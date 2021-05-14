# Roadmap

## Architecture Perspective

GRU4REC uses a bunch of GRU layers in a row, fed into a feedforward network.
This is great -- now, let's think about how to add attention to the network.

What attention-based readings have we looked through?

1. Seq2Seq for Sequential Recommendations (Sun + Qian). This paper isn't great,
   because it uses attention based off a specific feature, such as Genre. We
   don't want that here.
2. All You Need is Attention (the OG Transformer paper). This fulfills our
   attentive criteria, but isn't directly applicable to recommendations (it's
   still very rooted in translation).
3. GRU4REC. Unfortunately, I can't seem to find a way to add attentiveness here
   without adding in an encoding-decoding submodule (at which point it's
   basically the other papers).
4. NARM: Neural Attentive Session-based Recommendation (Li, Ren, et al). This
   is great! It uses GRU units, self-attention (no "specific feature selection"
   required for attention), and gives its output in a form directly usable for
   recommendations.

Code for each model is contained in their respective files.

## Code perspective

What existing assets do we have?

`nn.MultiheadAttention` is great, as is `nn.GRU` -- IF we don't want to tweak
how the hidden layers interact. If we *do* want to tweak this, we should
instead use what's in `encoder_decoder.py`, which makes every "step" of the
encoder + decoder parts visible.

Also, there's already a NARM implementation on GitHub. I've pulled it in here.
We may have to change the attention function from "addition-style" attention to
"dot-product-style" attention, so we can keep our results comparable to the
GRU4REC guys. TODO: double check which attention function they used.
