# Demystifying LLM Reasoning: From Language to Mathematical Constructs

## The Central Question of LLM Reasoning

When we ask a Large Language Model (LLM) to solve a math word problem, it generates text step-by-step. But beneath the surface of this text generation lies a profound mystery: How does a neural network, which fundamentally only predicts the next word, transform a linguistic description into a rigorous mathematical construct?

Consider this prompt:
"A shop sells a product for $120 after applying a 20% discount. What was its original price?"

The model might generate: "Let the original price be P. A 20% discount means the customer pays 80% of the original price. Therefore, 120 = 0.8P."

The central question is not how it predicts the next word, but how it arrives at the mathematical idea that a 20% discount means paying 80%? How does a statistical language model bridge the gap between semantic language and symbolic mathematics?

This article provides a comprehensive, step-by-step breakdown of this process. We will distinguish between what we know about the mechanical computation of transformers and the conceptual interpretation of what the model has learned, culminating in a detailed mathematical walkthrough of a hypothetical transformer processing this exact phrase.

---

## 1. The Context Window: The Foundation of Reasoning

Before understanding how the model calculates, we must understand what it is calculating over. The model does not process the prompt and its generation in isolated chunks; it processes the entire context simultaneously.

### The Dual Nature of Context
At the moment the model is about to generate the mathematical translation, its context contains two distinct pieces of information:

1. The Original Problem: The user's prompt containing the entities ($120, 20%, discount, original price).
2. The Generated Scratchpad: The model's own preceding text ("Let the original price be P.").

Crucially, the model does not forget the original question after generating P. Every subsequent token prediction is conditioned on the entire available context (subject to the context window limits).

[Mermaid diagram in original: Original Prompt + Generated Text -> Context Window -> Next-Token Prediction -> Output]

Conceptual Takeaway: The model's "working memory" is the attention mechanism's ability to look back at both the user's constraints and its own intermediate logical steps simultaneously.

---

## 2. The Training Crucible: Learning Mathematical Relationships

The model does not have a hardcoded rulebook of math formulas. Instead, it has learned distributed statistical relationships during training.

Imagine the model has processed millions of text sequences containing mathematical reasoning. During training, it adjusts its billions of parameters to minimize the error in predicting the next token in these sequences.

### Examples of Learned Patterns
Through exposure to vast corpora, the model encounters patterns like:

| Linguistic Pattern | Mathematical Translation |
| "A 10% discount means paying 90%" | P - 0.10P = 0.90P |
| "A 20% discount means paying 80%" | P - 0.20P = 0.80P |
| "A 20% increase means multiplying by 1.20" | P x 1.20 |
| "Half price" | 0.50P |

The model learns that:
Remaining Percentage = 100 - Discount Percentage
Remaining Multiplier = 1 - d/100

Important Distinction: We should not assume the model has stored the formula 1 - d/100 as a single, explicit, localized rule. Its knowledge is distributed across the weights of the network. The formula is a human-readable abstraction of the high-dimensional mathematical relationship the network has learned.

---

## 3. The Transformer Engine: How Language Activates Math

How does the model translate the token "discount" into a mathematical operation? This happens inside the Transformer architecture through three main mechanisms: Embeddings, Self-Attention, and Feed-Forward Networks (FFN).

### Step 1: Tokenization and Embeddings
The input text is broken into tokens and mapped to high-dimensional numerical vectors (embeddings). These vectors capture semantic meaning.

| Language Element | Conceptual Vector Dimensions (Simplified) |
| "Original price" | [Unknown, Baseline, Value, Currency] |
| "20%" | [Quantity, Fraction, Percentage, Rate] |
| "Discount" | [Operation, Reduction, Subtraction, Price] |
| "$120" | [Known, Final, Value, Currency] |
| "P" | [Symbol, Variable, Unknown, Baseline] |

### Step 2: Self-Attention (Contextualization)
Self-attention allows tokens to "look at" each other. The representation of "20%" is updated based on its proximity to "discount". The model learns that "20%" is not just a number; in this context, it is a rate of reduction.

### Step 3: Feed-Forward Networks (Non-linear Transformation)
The attention output is passed through an FFN. This is where the heavy lifting of "translating" language to math occurs. The FFN applies non-linear transformations that map the contextualized language vector into a new region of the embedding space that corresponds to mathematical operations.

---

## 4. The Step-by-Step Translation: From Words to Equations

Let's break the conceptual transition from language to math into granular steps. The model doesn't necessarily execute these as discrete, conscious steps, but its hidden states progressively build this logical chain.

[Flow in original: Step A Recognize Structure -> Step B Identify Reference -> Step C Establish Remainder -> Step D Simplify Expression -> Step E Connect to Observed -> Final Equation: 120 = 0.80P]

* Step A (Recognize Structure): The model identifies "20% discount" as a percentage reduction applied to a baseline.
* Step B (Identify Reference): It links the discount to the "original price" (P). Mathematically: Discount = 0.20P.
* Step C (Establish Remainder): It calculates what is left. Remaining = P - 0.20P.
* Step D (Simplify): It factors out P. P(1 - 0.20) = 0.80P.
* Step E (Connect to Observed): It equates the remaining expression to the known final price ($120). 120 = 0.80P.

The language statement "A 20% discount means the customer pays 80%" and the equation "120 = 0.8P" are simply two different textual representations of the exact same underlying mathematical reality that the model has learned to map between.

---

## 5. Context is King: Disambiguating Mathematical Operations

Why does "20% discount" lead to subtraction (or multiplying by 0.8), while "20% increase" leads to addition (multiplying by 1.2)?

The number 20% is mathematically ambiguous on its own. The operation is entirely dictated by the surrounding linguistic context.

| Question Context | Surrounding Language | Mathematical Relationship | Operation |
| "...after a 20% discount." | discount, reduction, off | 120 = 0.8P | Multiply by 0.8 |
| "...after a 20% increase." | increase, markup, added | 120 = 1.2P | Multiply by 1.2 |
| "...which is 20% of..." | is, of, fraction, part | 120 = 0.2P | Multiply by 0.2 |

The model must use self-attention to weigh the token "20%" against the tokens "discount", "increase", or "of". The attention heads learn that the semantic direction of the operation is carried by the noun/verb, while the magnitude is carried by the percentage.

---

## 6. The Three Levels of LLM Understanding

When we ask, "Is there an actual mathematical representation inside the neural network?", we must be precise about what we mean. Researchers in Mechanistic Interpretability divide understanding into three distinct levels:

[Diagram in original: Level 1 Observable (Generated Text) -> Level 2 Representational (Hidden States, probing) -> Level 3 Causal Computation (Ablation/Intervention)]

1. Level 1 - Observable (Behavioral): We can read the generated text. This proves the model outputs the right answer, but tells us nothing about how it got there.
2. Level 2 - Representational (Information): We can train a "probe" (a simple linear classifier) on the hidden states to see if the number 0.8 or the concept of subtraction is encoded in the vectors. Finding information in a hidden state does not prove the model used that specific information to compute the answer.
3. Level 3 - Causal (Mechanistic): We intervene. If we artificially zero out a specific attention head or neuron, does the model's math break? This proves causal necessity.

---

## 7. The One-Word Experiment: Proving Contextual Math

To prove the model isn't just memorizing that "20%" maps to "0.8", we can run a controlled behavioral experiment by changing only one word.

### The Experiment
We feed the model three structurally identical prompts, altering only the relational word:

* Experiment 1: "A product costs $120 after a 20% discount." -> Expected: 120 = 0.8P
* Experiment 2: "A product costs $120 after a 20% increase." -> Expected: 120 = 1.2P
* Experiment 3: "A product costs $120, which is 20% of its original price." -> Expected: 120 = 0.2P

### The Conclusion
If the model consistently generates the correct, distinct equations for all three, it provides strong behavioral evidence that it has learned contextual mathematical relationships. It is dynamically composing the magnitude (20%) with the relational operator (discount vs increase vs of) to compute the correct multiplier.

---

## 8. Where is the Equation? The Deep Research Question

This brings us to the frontier of AI research: Where exactly is the equation 120 = 0.8P inside the model?

There are three possibilities researchers investigate:
1. Linguistic Representation: Does the model just understand the synonyms? (e.g., "discount" = "subtract").
2. Mathematical Representation: Does the hidden state literally encode the vector for 0.8 or the algebraic structure y = mx?
3. Computational Mechanism: Which specific attention heads and MLP neurons form the "math circuit"?

The Reality of Distributed Representations:
We cannot assume there is one "math neuron" that holds the equation. In deep learning, representations are highly distributed and superimposed (a phenomenon known as superposition). The equation isn't stored in one place; it is an emergent property of millions of parameters interacting across dozens of layers.

---

## 9. The Mathematical Microscope: A Hypothetical Transformer Walkthrough

To move beyond verbal explanations, let's construct a highly simplified, hypothetical transformer example to show exactly how the mathematics of the network processes the phrase "20%" and "discount" to support the relationship 120 = 0.8P.

Note: Real LLMs use dimensions of 4,096+ and billions of parameters. We will use 3-dimensional vectors and a tiny 4-neuron hidden layer for conceptual clarity.

### Step 9.1: Input Embeddings
The tokens "20%" (t1) and "discount" (t2) are converted into 3D embedding vectors. Let's assign conceptual meanings to the dimensions: [Magnitude, Operation_Type, Direction].

E(t1) = [0.8, 0.1, 0.0] (High magnitude, neutral operation, neutral direction)
E(t2) = [0.0, 0.9, -0.8] (Zero magnitude, strong operation, negative/reduction direction)

### Step 9.2: Self-Attention (Information Routing)
The model calculates Query (Q), Key (K), and Value (V) vectors. For simplicity, we look at the Attention Score Matrix to see how the tokens interact.

The token "discount" (t2) needs to know the rate, so it pays strong attention to "20%" (t1).

Attention Scores = [[0.2, 0.8], [0.9, 0.1]] (Row 2 shows t2 attending 90% to t1, and 10% to itself).

The Contextualized Vector for "discount" (C2) is the weighted sum of the original embeddings:
C2 = 0.9 x E(t1) + 0.1 x E(t2) = [0.72, 0.18, -0.08]

What just happened? The vector for "discount" now contains the magnitude 0.72 (derived from 20%) while retaining its negative directional bias (-0.08). The language and the math have been fused into a single vector.

### Step 9.3: The Feed-Forward Network (The "Math" Translation)
The contextualized vector C2 is passed through a 2-layer Feed-Forward Network (FFN) with a hidden dimension of 4. The FFN acts as a key-value memory.

FFN(x) = W2 . ReLU(W1 . x + b1) + b2

Let's look at the hidden layer activations (after ReLU). Suppose Neuron 3 in the hidden layer is a "feature detector" specifically tuned to fire when it sees a positive magnitude combined with a negative direction (i.e., a percentage reduction).

Hidden Activations = [0.0, 0.0, 1.5, 0.0]

Neuron 3 fires strongly! This neuron has learned during training that the pattern [magnitude + negative direction] corresponds to the mathematical concept of "Remaining Multiplier = 1 - d".

### Step 9.4: Output Projection (Generating the Math)
The output of the FFN is projected back into the vocabulary space. Because Neuron 3 fired, it pushes the final output vector into a region of the embedding space that is highly aligned with the tokens "0.8", "80%", and "multiply by".

When the model goes to predict the next token after "A 20% discount means the customer pays...", the logits (raw scores) for the vocabulary are calculated.

Because the internal state has been shifted by the FFN to represent "80% remaining", the probability distribution for the next token looks like this:

| Next Token Candidate | Logit Score | Probability |
| "80%" | 8.4 | 65% |
| "0.8" | 7.1 | 18% |
| "20%" | -2.5 | 0.1% |
| "100%" | -4.0 | 0.0% |

The model confidently predicts "80%". In the subsequent step, it will predict "of", then "the", then "original", then "price", and finally formulate the equation 120 = 0.8P.

### Summary of the Microscope Walkthrough
1. Embeddings separated the magnitude (20%) from the operation (discount).
2. Attention fused them together, giving the operation the magnitude.
3. The FFN recognized the fused pattern (magnitude + negative direction) and activated a specific "math feature" neuron.
4. The Output Layer translated that activated feature into the textual token "80%".

---

## Conclusion

Understanding how an LLM reasons requires us to look past the illusion of human-like thought and examine the mechanical reality of high-dimensional vector spaces.

The model does not "know" math in the way a human does. It does not hold a symbolic equation in a working memory buffer. Instead, through billions of parameters adjusted during training, it has learned to map linguistic contexts to mathematical regions in a vector space. When it encounters "20% discount", self-attention fuses the tokens, and the feed-forward network translates that fused state into a vector that statistically aligns with the concept of 0.8.

By bridging the gap between conceptual interpretation and mechanistic computation, we can demystify the "black box." The magic of LLM reasoning is not magic at all; it is the beautiful, complex, and highly structured mathematics of deep learning.
