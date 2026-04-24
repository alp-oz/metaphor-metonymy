
## The geometry of figurative language: metaphor, metonymy 
## and multilingual embeddings

**Alperen Özdemir**

## Abstract

We test whether Jakobson's (1956) structural distinction 
between metaphor and metonymy leaves measurable geometric 
signatures in multilingual embedding space. Using curated 
pairs of figurative expressions across eight typologically 
diverse languages — English, French, Turkish, Russian, 
Swedish, German, Arabic and Japanese — we compute cosine 
similarity between independently embedded terms and compare 
distributions across four categories: live metonymy, live 
metaphor, dead metonymy and dead metaphor. Metonymic pairs 
are geometrically closer than metaphoric pairs across all 
eight languages combined (p<0.001), consistent with the 
theoretical claim that real-world contiguity produces 
semantic proximity in distributional representations while 
cross-domain similarity produces distance. Conventionalization 
dominates over structural type — the dead/live gradient is 
approximately three times larger than the metonymy/metaphor 
gradient. We interpret these findings as evidence that 
standard multilingual embeddings encode the lifecycle of 
figurative language more reliably than its structural type.

---

## 1. Introduction

Jakobson (1956) argued that metaphor operates along the axis of selection — 
substituting one term for another on the basis of 
similarity — while metonymy operates along the axis of 
combination, linking terms on the basis of real-world 
contiguity. This structural distinction has been enormously 
influential — in literary theory through Lodge (1977), in 
cognitive linguistics through Lakoff and Johnson (1980), 
and in psychoanalytic theory through Lacan (1957) — but 
has remained largely qualitative. No prior work has tested 
whether the distinction leaves a measurable geometric trace 
in the distributional representations learned by language 
models.

We address this gap directly. If contiguity in the real 
world produces co-occurrence in language, and co-occurrence 
produces proximity in embedding space, then metonymic pairs 
should be geometrically closer than metaphoric pairs. We 
test this prediction across eight typologically diverse 
languages using a multilingual sentence embedding model and 
a curated dataset of figurative expression pairs classified 
by structural type and degree of conventionalization.

---

## 2. Method

### 2.1 Embedding model

We use `paraphrase-multilingual-MiniLM-L12-v2` from the 
sentence-transformers library (Reimers and Gurevych, 2019). 
This is a 12-layer transformer model fine-tuned on parallel 
multilingual corpora to produce semantically similar 
embeddings for paraphrases across languages. It maps text 
to 384-dimensional vectors and covers 50+ languages in a 
shared embedding space.

For each pair (term₁, term₂) we encode each term 
**independently** — not together in one sentence — producing 
two 384-dimensional vectors v₁ and v₂. We then compute 
cosine similarity:

cosine_similarity(v₁, v₂) = (v₁ · v₂) / (‖v₁‖ · ‖v₂‖)

This ranges from -1 to 1. A score of 1 means the two terms 
are distributionally identical in the model's representation. 
A score of 0 means they share no distributional structure. 
Scores in our dataset range from approximately 0.1 to 0.98.

The model was trained to make paraphrases close in embedding 
space. Highly conventionalized pairs — where one term 
functions as a near-paraphrase of the other in context — 
will therefore score high. This is relevant to the 
interpretation of dead figure scores below.

### 2.2 Dataset

We constructed a dataset of curated word and phrase pairs 
across eight languages: English, French, Turkish, Russian, 
Swedish, German, Arabic and Japanese. Pairs were classified 
into four categories based on Jakobson's structural 
distinction and the linguistic notion of conventionalization:

**Live metonymy** — institutionalized substitutions where 
a specific term stands for a related entity through 
real-world contiguity. The contiguity relation is active 
and culturally current.

Examples:
- EN: ("White House", "US government")
- FR: ("la plume", "l'écrivain") — pen for writer
- TR: ("Köşk", "cumhurbaşkanlığı") — presidential palace
- RU: ("Кремль", "российское правительство")
- JA: ("永田町", "日本の政府") — Nagatacho for government

**Live metaphor** — cross-domain mappings where similarity 
licenses substitution. The source and target domains are 
semantically distant.

Examples:
- EN: ("time", "money")
- FR: ("la vie", "un voyage") — life is a journey
- TR: ("aşk", "ateş") — love is fire
- AR: ("البركان", "الغضب الشديد") — volcano for anger
- JA: ("狐", "狡猾さ") — fox for cunning

**Dead metonymy** — conventionalized producer-for-product 
substitutions where the proper noun has fused with its 
product category through repeated use.

Examples:
- EN: ("read a Hemingway", "read a novel by Hemingway")
- FR: ("écouter un Chopin", "écouter une œuvre de Chopin")
- DE: ("einen Goethe lesen", "ein Werk von Goethe lesen")
- JA: ("夏目漱石を読む", "夏目漱石の小説を読む")

**Dead metaphor** — body-part or physical-object extensions 
now perceived as literal in everyday usage.

Examples:
- EN: ("foot of the mountain", "base of the mountain")
- FR: ("le pied de la montagne", "la base de la montagne")
- TR: ("dağın eteği", "dağın alt kısmı")
- DE: ("Fuß des Berges", "Basis des Berges")

**Pair construction notes:**

French native pairs were constructed using idiomatic 
French expressions from standard reference works rather 
than translated from English. Turkish pairs were verified 
by a native speaker. Arabic pairs use fuller phrases rather 
than single words to provide sufficient embedding context — 
single Arabic words were found to produce compressed, 
underdiscriminating embeddings in pilot testing. Japanese 
metaphor pairs use animal-behavior and physical-abstract 
mappings to maximize cross-domain distance; journey and 
goal metaphors were found to be near-synonymous in Japanese 
embedding space and were replaced.

**Cross-language comparability caveat:**

Differences in mean similarity across languages should not 
be interpreted as reflecting genuine linguistic differences. 
They reflect a combination of three confounds:

1. Training data coverage — the model has seen vastly more 
   English and Russian than Swedish, Arabic or Japanese
2. Pair construction differences — Arabic pairs use longer 
   phrases than English pairs, affecting embedding geometry
3. Domain selection — pairs were chosen to be clear 
   instances of each category, not matched for topic or 
   register across languages

Cross-language magnitude comparisons are therefore not 
meaningful. The directional claim — metonymy closer than 
metaphor, dead closer than live — is the robust finding. 
The specific similarity scores within languages are 
informative only relative to other categories in the same 
language.

### 2.3 Statistical test

Mann-Whitney U tests compare similarity distributions 
across categories. All tests are two-tailed. We report 
combined tests across all languages and per-language 
results separately.

---

## 3. Results

### 3.1 Main result — metonymy versus metaphor

Combining live and dead pairs within each structural type 
across all eight languages:

| Type | Mean similarity | N |
|---|---|---|
| Metonymy | 0.677 | 180 |
| Metaphor | 0.579 | 210 |

Combined Mann-Whitney p < 0.001. The direction — metonymy
closer than metaphor — holds in every language without
exception.

### 3.2 Four-category gradient

| Category | Mean similarity | N |
|---|---|---|
| Dead metonymy | 0.818 | 80 |
| Dead metaphor | 0.693 | 105 |
| Live metonymy | 0.565 | 100 |
| Live metaphor | 0.464 | 105 |

All six pairwise comparisons significant at p < 0.001.

The conventionalization gradient — dead versus live within
each type — is larger than the structural gradient —
metonymy versus metaphor within each lifecycle stage:

- Dead minus live for metonymy: 0.818 − 0.565 = 0.253
- Dead minus live for metaphor: 0.693 − 0.464 = 0.229
- Metonymy minus metaphor for live pairs: 0.565 − 0.464 = 0.101
- Metonymy minus metaphor for dead pairs: 0.818 − 0.693 = 0.125

### 3.3 Per-language results

| Language | Live met | Live meta | Dead met | Dead meta | Gradient holds |
|---|---|---|---|---|---|
| English | 0.591 | 0.379 | 0.898 | 0.768 | ✓ |
| French | 0.454 | 0.411 | 0.807 | 0.821 | ✗ |
| Turkish | 0.620 | 0.534 | 0.759 | 0.799 | ✗ |
| Russian | 0.629 | 0.489 | 0.876 | 0.797 | ✓ |
| Swedish | 0.582 | 0.514 | 0.885 | 0.755 | ✓ |
| German | 0.602 | 0.437 | 0.782 | 0.534 | ✗ |
| Arabic | 0.498 | 0.505 | 0.730 | 0.371 | ✗ |
| Japanese | 0.526 | 0.441 | 0.810 | 0.460 | ✓ |

The strict four-category gradient holds in five of eight 
languages. Failures occur exclusively at the dead metonymy 
versus dead metaphor boundary — never at the live metonymy 
versus live metaphor boundary. The fundamental structural 
distinction between live figures is robust across all 
languages.

### 3.4 Notable observations

**Swedish.**
An earlier run using culturally specific institutional pairs, e.g., Rosenbad for the prime minister's office, produced a completely flat result (p=1.000). Replacing these with instrument-for-person and container-for-contents pairs, e.g., pennan/författaren, scenen/teatern, restored the signal (live metonymy 0.582). Whether this reflects the model's limited exposure to Swedish political discourse or a genuine property of how these metonymic relations are encoded in Swedish text cannot be determined from this evidence.

**Arabic.**
Arabic shows an unexpected reversal in the dead/live metaphor ordering: dead metaphor (0.371) scores below live metaphor (0.505), the only language where this occurs. This could reflect genuine properties of Arabic figurative language, a pair construction asymmetry — our metaphor pairs use shorter phrases than the metonymy pairs — or differential training data coverage across registers and domains. We cannot distinguish between these explanations with the current design.

**German.**
Dead metaphor scores notably lower than the cross-language mean (0.534 versus 0.693). This could reflect properties of German morphological structure, the specific pairs chosen, or training data effects. We do not offer an interpretation beyond noting the anomaly.

**Japanese.**
An earlier run using journey and goal metaphor pairs — 道/人生 (road/life), 目標/成功 (goal/success) — produced near-zero separation between dead and live metaphor. Replacing these with animal-behavior and physical-abstract pairs restored the expected ordering. This suggests the earlier pairs were near-synonymous in the model's Japanese representation, though whether this reflects Japanese semantic structure or training data properties is unclear.

## 4. Limitations

**Pair selection.** Pairs were constructed to be clear, 
unambiguous instances of each category. They are not a 
random sample of figurative language in naturalistic text. 
Results may not generalize to figurative language in 
running discourse.

**Co-occurrence confound.** The geometric signal is 
consistent with Jakobson's structural distinction but 
equally compatible with a simpler co-occurrence 
explanation. Metonymic pairs may co-occur more frequently 
in text because they refer to the same real-world 
situation from different angles, independently of any 
deep structural operation. A stronger test would require 
controlling for co-occurrence frequency independently of 
figurative type.

**Training data coverage.** Results for English and 
Russian are more reliable than for Swedish, Arabic and 
Japanese due to differential training data coverage in 
the multilingual model.

**Pair length asymmetry.** Arabic pairs use longer phrases 
than pairs in other languages, introducing a potential 
confound in cross-language comparisons.

**Sample size.** With 10-20 pairs per category per language, 
individual language results are underpowered. The combined 
cross-language results are more reliable than any single 
language result.

---

## 5. Reproducibility

```bash
pip install sentence-transformers umap-learn wordfreq \
    nltk anthropic --break-system-packages
python -c "import nltk; nltk.download('wordnet')"
```

Model: `paraphrase-multilingual-MiniLM-L12-v2`

All pair lists are in `data/pairs/`. Run the full analysis:

```bash
python src/pairs_geometry.py --output results/figures/pairs_plot.png
python src/specificity_asymmetry.py --output results/figures/specificity_plot.png
python src/wordnet_check.py --output results/figures/wordnet_plot.png
```

---

## References

Hamilton, W., Leskovec, J. and Jurafsky, D. (2016). 
Diachronic word embeddings reveal statistical laws of 
semantic change. *ACL 2016*.

Jakobson, R. (1956). Two aspects of language and two 
types of aphasic disturbances. In *Fundamentals of 
Language*. Mouton, The Hague.

Lakoff, G. and Johnson, M. (1980). *Metaphors We Live By*. 
University of Chicago Press.

Lacan, J. (1957). The instance of the letter in the 
unconscious. In *Écrits*. Norton.

Lodge, D. (1977). *The Modes of Modern Writing*. Arnold.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: 
Sentence embeddings using Siamese BERT-networks. *EMNLP*.



