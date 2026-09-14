# Week-3 generation parity: BUGPress vs. baseline (pre-RoPE, values=on)

Model: `unsloth/Llama-3.2-1B-Instruct` -- greedy decode, 20 new tokens. '=' means the continuation is byte-identical to the no-press baseline.

SHORT prompts (T-n_sink <= rank) are a no-regression sanity check: the press has almost nothing to compress and should match the baseline exactly. LONG prompts (T >> rank) genuinely exercise the rank-r model; graceful divergence as rank drops is expected, garbage is not.

## SHORT prompts

| Prompt | baseline | r=64 | r=32 | r=16 | r=8 |
|---|---|---|---|---|---|
| [6tok] `The capital of France is` | 'Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in' | = (exact) | = (exact) | = (exact) | = (exact) |
| [7tok] `Water is composed of hydrogen and` | 'oxygen atoms. The chemical formula for water is H2O. This means that one molecule of water' | = (exact) | = (exact) | = (exact) | = (exact) |
| [9tok] `The first president of the United States was` | 'George Washington. He was inaugurated as the first president on April 30, 1789.' | = (exact) | = (exact) | = (exact) | = (exact) |
| [9tok] `In machine learning, a neural network is` | 'a type of machine learning model that is inspired by the structure and function of the human brain. Neural' | = (exact) | = (exact) | = (exact) | = (exact) |
| [11tok] `The three primary colors are red, blue, and` | 'yellow. These colors are combined in different ways to create various shades and hues of colors. The primary' | = (exact) | = (exact) | = (exact) | = (exact) |
| [9tok] `Photosynthesis is the process by which plants` | ', algae, and some bacteria convert light energy from the sun into chemical energy in the form of glucose' | = (exact) | = (exact) | = (exact) | = (exact) |
| [10tok] `The speed of light in a vacuum is approximately` | '299,792 kilometers per second (km/s). This is a fundamental constant of nature, and' | = (exact) | = (exact) | = (exact) | = (exact) |
| [10tok] `To make a cup of tea, first you` | 'need to boil water in a kettle. Then, you need to add tea leaves to the boiling water' | = (exact) | = (exact) | = (exact) | = (exact) |
| [8tok] `The theory of evolution was proposed by` | 'Charles Darwin in his book "On the Origin of Species" in 1859. The theory of' | = (exact) | = (exact) | = (exact) | = (exact) |
| [12tok] `A prime number is a natural number greater than one that` | 'has exactly two distinct positive divisors: 1 and itself. In other words, a prime number' | = (exact) | = (exact) | = (exact) | = (exact) |

Exact-match rate for **SHORT prompts**:
- rank 64: 10/10 exact, compression_ratio=0.875
- rank 32: 10/10 exact, compression_ratio=0.938
- rank 16: 10/10 exact, compression_ratio=0.969
- rank 8: 10/10 exact, compression_ratio=0.984

## LONG prompts

| Prompt | baseline | r=64 | r=32 | r=16 | r=8 |
|---|---|---|---|---|---|
| [167tok] `The theory of relativity was dev...f relativity is that` | 'the laws of physics are the same for all observers in uniform motion relative to one another, and that' | = (exact) | = (exact) | 'the laws of physics are the same everywhere in the universe, and that time and space are relative,' | 'the speed of light is constant and unchanging, and that time and space are relative. The theory' |
| [141tok] `Photosynthesis is the biological...r photosynthesis are` | 'carbon dioxide and water. The energy from sunlight is used to convert these raw materials into glucose and oxygen' | 'carbon dioxide and water. The energy from sunlight is used to convert these raw materials into glucose, which' | 'carbon dioxide and water, and the products are glucose and oxygen. The process is essential for life on' | 'carbon dioxide and water, and release oxygen as a byproduct. The process is essential for life on' | 'carbon dioxide and water and release oxygen as a byproduct. The process of photosynthesis is essential for' |
| [145tok] `The Industrial Revolution was a ...l Revolution was the` | "steam engine, which was invented by James Watt in 1769. Watt's improvements to the steam" | = (exact) | = (exact) | 'steam engine, which was invented by James Watt in 1769. The steam engine provided a reliable' | 'steam engine, which revolutionized industry and transportation, and the development of new technologies such as the tele' |

Exact-match rate for **LONG prompts**:
- rank 64: 2/3 exact, compression_ratio=0.875
- rank 32: 2/3 exact, compression_ratio=0.938
- rank 16: 0/3 exact, compression_ratio=0.969
- rank 8: 0/3 exact, compression_ratio=0.984
