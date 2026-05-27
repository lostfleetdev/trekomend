# Trekomend: A Content-Based Movie Recommendation Engine

**May 2026**

A system that embeds 1.4 million TMDB movies into 1024-dimensional vectors
with Qwen3-Embedding-0.6B, indexes them with FAISS IVF-PQ, and serves
recommendations through a FastAPI server behind Nginx on a $6/month VPS.
Memory footprint is under 200 MB. Default pipeline latency is about 14 ms
warm, delivering 6-9 unique genres per query.

---

## 1. What this system does

Trekomend is a content-based retrieval engine for movies. It takes a movie
title, a natural language description, or a list of movies you like and
returns recommendations. Every search happens against 1.4 million stored
embeddings. There is no collaborative filtering, no user profiles, no
session state. The model that produces the embeddings runs once on Kaggle's
free T4 GPUs and the output ships as three files: an HDF5 embedding matrix
(4.8 GB), a FAISS IVF-PQ index (108 MB), and a SQLite metadata database
(912 MB). Those files are hosted on HuggingFace. The web server loads
the index and database at startup and responds to HTTP requests.

A content-based system has obvious limits. It cannot tell you that "people
who liked Inception also watched Shutter Island" unless those movies happen
to be near each other in embedding space. It cannot model how your taste
evolves over time. It has no notion of popularity beyond what the training
data baked into the embeddings. But it requires no user data, no tracking,
no GPU for inference, and no machine learning at serving time beyond what
the FAISS index already does. For a project built by one person on a $6
VPS, those constraints are the whole point.

## 2. Embedding pipeline

### The model

Qwen3-Embedding-0.6B converts movie text into 1024-dimensional vectors.
It runs on less than 1.2 GB of VRAM on a T4. It was pretrained with
Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022),
meaning the first 256 dimensions form a valid, coarser embedding on their
own, and each additional block of dimensions adds finer detail. The system
uses the full 1024 dimensions. For a mobile or edge deployment, truncating
to 256 or 512 would drop the HDF5 from 4.8 GB to about 1.2 GB at roughly
80 percent quality retention.

The model uses asymmetric prompts. On the query side, a short instruction
precedes the user's text: "Given a description of movie preferences
including liked films, desired genres, themes, mood, and watching context,
retrieve movies that best match these criteria." The document side gets raw
text with no instruction prefix. This is the documented Qwen3 convention and
recovers 1-5 percent on retrieval benchmarks compared to symmetric encoding.

We use four instruction templates depending on search context. The general
template balances all signals equally. The mood template prioritizes
atmospheric and emotional fit. The genre template emphasizes plot and
narrative similarity. The hybrid template combines taste history with
current mood for profile-based searches.

### The Kaggle notebook

The embedding notebook runs 14 cells on Kaggle's free dual T4 runtime and
finishes in about three hours. It builds text for 1.4 million movies, loads
two independent Qwen3 instances (one per GPU), embeds in parallel using
manual model replication rather than DataParallel (which adds per-forward
overhead and the T4s lack NVLink anyway), and writes HDF5 shards with SQLite
checkpointing for crash recovery.

Length-sorted batching minimizes padding. The tokenizer and embedding loops
run in Python threads with Rust-backed tokenization releasing the GIL, so
the dual-GPU pipeline saturates both T4s. An adaptive OOM handler halves
batch size on CUDA errors and grows it back gradually. Kaggle sessions time
out after 12 hours, but the SQLite checkpoint means the notebook picks up
from the last completed shard on rerun.

After embedding, the notebook merges all shards into a single HDF5 file
with gzip level-1 compression, builds a FAISS IVF-PQ index with 2048
centroids and 64-byte PQ codes, and exports the SQLite metadata database
with FTS5 full-text search on titles, overviews, and keywords.

### What goes into the embedding text

Twelve of the 24 TMDB CSV columns make it into the embedding. The fields
are ordered by semantic priority:

```
Layer 1 (identity):     Movie: Inception
                        Also known as: (only if original_title differs)

Layer 2 (strong signal): Genres: Action, Science Fiction, Thriller
                         Themes and elements: dream, subconscious, heist

Layer 3 (rich content):  Plot: A thief who steals corporate secrets
                         through dream-sharing technology...

Layer 4 (supplementary): Tagline: Your mind is the scene of the crime

Layer 5 (context):       Year: 2010 | 148 min | Language: en |
                         Country: United States of America |
                         Studio: Warner Bros. Pictures
```

Plot overview gets up to 2000 characters. Keywords get 800. Genres get 500.
Production context (country, studio, runtime) condenses into a single
pipe-delimited line because those signals are individually weak.

Keywords carry roughly twice the impact of plot text alone for similarity
matching, based on ablation tests with the genre classification probe.
Genres provide essential coarse clustering. Title and year help with
identity and era preference. Everything else is supplementary.

The 12 excluded columns are budget, revenue, vote average, vote count,
popularity, adult flag, poster path, backdrop path, homepage URL, IMDB ID,
spoken languages, and status. These are collaborative signals, not content
signals. Including them made the embeddings worse when tested. This
exclusion turned out to align with Meehan and Pauwels (RecSys 2025), who
showed that content-only cold-start models amplify popularity bias when
trained with popularity-adjacent features.

### Genre classification as a quality probe

A logistic regression classifier trained on the raw embeddings guesses the
primary genre with 89.7 percent accuracy against a 44.1 percent
most-common-genre baseline, a 2.0x improvement. This confirms the
embeddings encode meaningful genre structure without explicit training.
The probe on its own does not guarantee recommendation quality, but if the
embeddings could not separate genres, nothing downstream would work either.

## 3. Search architecture

### FAISS IVF-PQ

The system uses a FAISS IndexIVFPQ with 2048 clusters (nlist), 64
subquantizers (M), and 8 bits per quantization code (nbits). Each vector
compresses to 64 bytes instead of the raw 4096 bytes of float32. The index
file is 108 MB and maps into memory via mmap at startup. At nprobe=32, the
default, latency is about 15 ms for a 1024-dimensional query against 1.4
million vectors.

Increasing nprobe improves recall at the cost of speed. At nprobe=64,
latency roughly doubles but recall climbs past 98 percent for most queries.
At nprobe=128, the index scans about 12 percent of all vectors and recall
approaches 99.5 percent at around 60 ms. For interactive use, nprobe=32
hits a sweet spot: fast enough that users do not notice the wait, accurate
enough that the top results match brute-force search for most queries.

### SQLite metadata

The metadata database uses FTS5 for full-text search across titles,
overviews, keywords, taglines, genres, original language, and production
countries. Movie lookups join the FTS5 index with the main movies table.
Genre and year indices support filtered browsing. A batch lookup function
fetches up to 999 movie IDs in a single SQL query by chunking into groups
of 900 to stay under SQLite's variable binding limit. The database caches
frequently accessed embeddings in a 2048-entry LRU cache backed by a
persistent HDF5 file handle, reducing random reads from about 8 ms cold to
near-zero warm.

### Retrieval pipeline

Every recommendation endpoint follows the same pattern. The query is
resolved to an embedding vector by either fetching it from the HDF5 file
(title-based queries), calling Ollama for live text embedding (text
queries), or computing a centroid from multiple liked movies plus an
optional mood vector (profile queries). The vector is L2-normalized and
searched against the FAISS index. The pipeline fetches up to 200
candidates, more than the final result count, to leave room for downstream
re-ranking and diversification.

The profile endpoint blends liked movie embeddings into a taste centroid,
optionally projects away from disliked movies by subtracting a fraction of
the projection onto the dislike centroid, and optionally blends in a mood
text embedding via Ollama. The mood weight defaults to 0.3, meaning 70
percent taste, 30 percent current mood. The dislike weight defaults to 0.3
and controls how strongly the system pushes away from unwanted directions
in embedding space.

## 4. Post-retrieval pipeline

Three layers run on the top 200 FAISS candidates before returning results
to the user. All three are optional. The default path skips the third layer
and runs in about 14 ms warm.

### Layer 1: LightGBM re-ranker

A gradient boosted tree model with 100 trees and 31 leaves per tree learns
to predict relevance from 15 features per query-candidate pair. The feature
set combines embedding similarity, metadata overlap, and quality signals:

| Feature | Type | What it captures |
|---|---|---|
| cosine_sim | float | The FAISS dot-product score |
| genre_jaccard | float | Overlap in comma-separated genre tags |
| genre_match_count | int | Raw count of matching genre tags |
| year_diff_abs | int | Absolute year difference |
| year_diff_bucket | int | 0=same year, 1=within 5 years, 2=within 15, 3=far |
| popularity_percentile | float | Percentile rank of the candidate's vote count |
| embedding_norm_candidate | float | L2 norm of the candidate embedding |
| same_language | bool | Original language match |
| keyword_jaccard | float | Jaccard similarity on comma-split keywords |
| vote_average_candidate | float | Candidate's average rating |
| vote_average_query | float | Query movie's average rating |
| rating_diff | float | Absolute difference in ratings |
| runtime_ratio | float | min/max ratio of runtimes |
| same_primary_genre | bool | Whether primary genres match |
| embedding_norm_query | float | L2 norm of the query embedding |

The model trains on pseudo-labels: a composite score that weights cosine
similarity (0.50), normalized rating (0.15), log vote count (0.10), genre
match (0.10), keyword overlap (0.10), and year proximity (0.05). These are
not ground truth labels but directionally correct proxies that encode what
a reasonable recommendation looks like. The LightGBM model learns nonlinear
interactions the handcrafted formula misses.

The training pipeline selects 240 stratified query movies across major
genres, retrieves top FAISS candidates for each, adds random negatives, and
produces about 60,000 training pairs. Training takes under a second on CPU.
The model file is 585 KB. Validation RMSE is 0.0031 with R-squared 0.999 on
held-out queries. Top features by gain: cosine_sim (1335), vote_average
(824), genre_jaccard (812), year_diff_abs (733), popularity_percentile (500).

The re-ranker runs before the genre diversification step. Running it after
collapsed the genre spread because the cosine-dominated scores pulled
same-genre items back to the top regardless of the round-robin ordering.
With LightGBM applied first, the within-genre selections benefit from the
improved relevance scores while the round-robin interleaving preserves genre
diversity on top.

LightGBM was chosen over a small neural network because it trains in under
a second on CPU, infers on 200 candidates in about 1 ms, produces
interpretable feature importance, and resists overfitting on the small
training set. A neural net with 60K examples would require careful
regularization and hyperparameter tuning for comparable performance.

### Layer 2: Genre round-robin diversification

A zero-training layer that groups the 200 candidates by primary genre and
interleaves them. Within each genre group, items sort by their LightGBM
score. The round-robin picks the first item from each genre, then the
second, and continues. Soft constraints cap each genre at three appearances
in the top 12 and require at least three unique genres.

A serendipity slot at position 8 reserves space for a high-rated movie
from a genre not yet seen in the top results, provided it meets minimum
quality thresholds. This slot directly implements the "balanced
recommendations with limited harm to accuracy" recommendation from Meehan
and Pauwels (RecSys 2025).

This layer adds near-zero overhead. Genre labels come from a single batched
SQLite query against the indexed primary_genre column. For most queries, it
boosts unique genre count from 2-4 to 6-9 with no perceptible latency cost.
The Stitch Fix engineering team described a similar "top-N per category"
pattern in their 2021 blog post about image-based recommendation.

### Layer 3: DPP set selection

An opt-in layer available through the `/recommend/diverse` and
`/recommend/explore` endpoints. A low-rank Determinantal Point Process
selects a mathematically diverse subset from the top 50 candidates that
have passed through Layers 1 and 2.

DPPs model the probability of selecting a subset as proportional to the
determinant of a kernel matrix. Items that are similar have higher
off-diagonal kernel values, which reduces the determinant and lowers their
probability of co-occurring. The result is a globally diverse set rather
than a greedily diverse one. Maximum Marginal Relevance, by contrast,
penalizes pairwise similarity to already-selected items, which can create
chains where A is close to B, B is close to C, but A and C are far apart.
DPP's exponential penalty for similarity prevents this.

The kernel uses a quality-diversity decomposition from Gartrell et al.
(KDD 2016):

```
L_ij = q_i * S_ij * q_j
```

where q_i is the quality score of item i (from LightGBM, with a round-robin
position bonus) and S_ij is the cosine similarity between their embeddings.
The lambda_qd parameter controls the tradeoff. At lambda=1.0, the kernel
reduces to pure quality scoring. At lambda=0.0, it maximizes set diversity
regardless of relevance. The default is 0.7.

A low-rank factorization with r=20 reduces the computational cost from
O(K^3) to O(K * r^2) per greedy selection step. With K=12 and r=20, this
amounts to about 5,000 operations per step. The factorization uses the
matrix determinant lemma and Sherman-Morrison updates for the Cholesky
factor of the r-by-r Gram matrix. Each candidate scores in O(r^2) instead
of recomputing determinants from scratch.

The DPP layer adds about 3 ms of warm latency and roughly 30 ms cold (when
it needs to fetch 50 candidate embeddings from the HDF5 file). The LRU
cache absorbs most of this in repeated queries. Given the default pipeline
already delivers strong genre diversity through round-robin alone, DPP is
most useful when a user wants fine-grained set optimization or maximal
exploration.

## 5. Evaluation approach

### Metrics that work without user data

Traditional recommendation metrics like precision@K and recall@K require
ground-truth labels: ratings, clicks, watch history. A content-only system
has none of these. The evaluation suite instead measures geometric and
statistical properties of the result set.

**Intra-List Diversity (ILD)** is 1 minus the average pairwise cosine
similarity of all result pairs. An ILD of 0.30 means the results are
tightly clustered, probably the same genre and era. An ILD of 0.55 means
good cross-genre spread. ILD above 0.60 can indicate the query was too
broad or the results are effectively random. ILD only measures internal
diversity. It does not care whether the results are relevant to the query.
That is why it always pairs with average similarity to the query vector.

**Redundancy** is the mean of the three highest pairwise similarities in
the result set. It catches near-duplicate recommendations that ILD's
averaging might hide. A result set can have ILD 0.50 (decent) but
redundancy 0.85 (three pairs are nearly identical). The main cause in
this dataset is multiple TMDB entries with the same title: "Inception"
appears as the original film, a documentary about the film, and a
re-release, all with near-identical embeddings.

**Genre entropy** (Shannon) measures how evenly the primary genres
distribute across results. If all 12 results are Action movies, entropy is
zero. If 12 results span six genres equally, entropy is about 2.58. The
normalized version divides by log2 of the number of unique genres, making
it comparable across different result sizes. Genre entropy is noisy because
TMDB genre tags are inconsistently applied and the primary genre is simply
the first tag in a comma-separated list, which is an arbitrary choice.

**Novelty** is defined as -log10(average vote count). It compresses vote
counts on a log scale because they follow a power law: a few movies have
millions of votes, most have thousands, and a raw average would be
dominated by one blockbuster. A novelty score around -4.0 means mainstream
blockbusters (10K to 100K votes). Around -2.5 is niche. Around -1.5 is
extremely obscure and often indicates bad metadata rather than a hidden gem.

**Serendipity** (profile searches only) is the product of relevance and
unexpectedness, where unexpectedness means dissimilarity to movies the
user already likes. A result is serendipitous if it is both relevant to
the search and different from what the user already knows. This captures
the "pleasantly surprising" quality. A recommendation can be relevant
without being serendipitous (more of the same), and it can be unexpected
without being relevant (random). Serendipity requires both.

### What cannot be measured yet

Without user interaction data, the system cannot compute precision, recall,
NDCG, click-through rate, watch-through rate, or session continuation.
Adding collaborative signals from the MovieLens 25M dataset would enable
backtesting against held-out ratings. For now, the evaluation suite tells
you whether the results are internally diverse and novel. It cannot tell
you whether they are the right movies for a specific human.

### Quality grades

A simple heuristic assigns a grade based on threshold tests across ILD,
redundancy, genre entropy, and novelty. Results need at least 75 percent of
the maximum possible score across all metrics for an EXCELLENT grade, 50
percent for GOOD, and 25 percent for FAIR. This is a rough signal. It
rewards diversity and novelty more than raw relevance because relevance is
already guaranteed by cosine similarity retrieval. The whole point of the
post-retrieval layers is making results less homogeneous.

## 6. Popularity bias

Meehan and Pauwels (RecSys 2025) showed that cold-start recommendation
models inherit and amplify popularity bias from their training data. Even
when explicit popularity signals are stripped out, the embedding space
still encodes popularity: movies with content similar to popular movies
get over-recommended because the training distribution is skewed toward
those movies' textual patterns.

Their proposed mitigation uses embedding vector magnitude as a proxy for
predicted popularity. We implemented a variant that instead uses explicit
vote counts since the raw data is available: the system computes the
embedding centroid of the 10,000 most-voted movies as a "popularity
direction" and penalizes candidates whose vectors project onto it. The
penalty weight defaults to 0.15. Only positive projections get penalized.
Movies pointing away from the popular direction pass through unchanged.

With debiasing enabled, the average vote count of recommendations drops
from roughly 13,500 to about 2,400. The novelty score improves by about 0.7
on the log scale. Some of the surfaced movies have genuinely sparse metadata
and their embedding quality suffers as a result. A minimum metadata quality
threshold would help, but filtering too aggressively removes the obscure
movies the debiasing is supposed to surface.

Embedding norm appears as a LightGBM feature (number 7) so the re-ranker
can learn to discount popular items contextually rather than with a
uniform penalty. The feature is currently zero-importance with the
pseudo-label training data, suggesting the proxy is too noisy or the
training signal does not capture the debiasing objective. A future
training run with real user feedback would likely surface this signal.

## 7. Literature and production systems

### Foundational work

Carbonell and Goldstein (1998) introduced Maximum Marginal Relevance for
text summarization. The iterative selection algorithm balancing relevance
against novelty has been standard in information retrieval for 25 years and
was the system's first diversity mechanism. The MMR formulation directly
inspired the DPP selection in Layer 3, which generalizes pairwise diversity
to global set diversity.

Kulesza and Taskar (2012) developed Determinantal Point Processes as a
principled mathematical framework for subset selection with diversity
constraints. Their monograph is the standard reference.

Cormack, Clarke, and Buettcher (2009) introduced Reciprocal Rank Fusion
for merging ranked lists from heterogeneous sources. RRF is
score-distribution agnostic (only ranks matter), which makes it the right
choice for fusing content-based and collaborative retrieval results when
the latter is added.

### Core techniques

Kusupati et al. (NeurIPS 2022) introduced Matryoshka Representation
Learning, the technique that lets a single embedding model produce valid
vectors at multiple dimensions. Qwen3 was pretrained with MRL, so the
embeddings support truncation from 1024 down to 32 dimensions.

Gartrell, Paquet, and Koenigstein (KDD 2016) introduced the quality-
diversity kernel decomposition for DPPs and the low-rank factorization
that makes them computationally practical for recommendation. Their
GitHub code was the starting point for the DPP implementation in this
system. Chen, Zhang, and Zhou (NeurIPS 2018) later accelerated the
greedy MAP inference with local repulsion heuristics and showed better
relevance-diversity tradeoffs than MMR in online A/B tests.

Ke et al. (NeurIPS 2017) introduced LightGBM. The histogram-based
gradient boosting algorithm with leaf-wise tree growth trains on CPU
in seconds and produces compact, human-readable models. It is the
re-ranker used in Layer 1 of the pipeline.

Covington, Adams, and Sargin (RecSys 2016) described YouTube's two-stage
architecture: candidate generation from an approximate nearest neighbor
index followed by a deep ranking model with feature crosses. This
two-stage pattern (ANN retrieval then re-ranking) is the architecture
adopted here, with LightGBM and diversity layers replacing the deep
ranking model.

### Production systems

Huang et al. (KDD 2020) described Facebook's embedding-based retrieval
pipeline for search, with hard negative mining and quantization for
serving. Their parameter tuning methodology for ANN indices informed
the nprobe selection strategy for the FAISS index.

Li et al. (KDD 2024) demonstrated that diversity must exist upstream of
re-ranking. Their Contextual Distillation Model trains a diversity-aware
student from an MMR teacher by encoding positive and negative context
signals. The paper's core insight, that diversifying early in the
pipeline matters more than diversifying late, led to placing the
round-robin layer before the DPP selector.

Ibrahim et al. (RecSoGood 2025) provided practitioner guidance for
personalized DPP deployment with quality-diversity kernel decomposition
and full code on GitHub. Their work informed the lambda_qd tradeoff
parameter and the personalization bias for quality scores.

Stitch Fix's engineering blog (2021) described their multi-objective
scoring function and top-N per category diversity pattern, which
inspired the genre round-robin interleaving strategy.

### Sequential and collaborative models

Kang and McAuley (ICDM 2018) introduced SASRec, a self-attention
sequential recommendation model that captures the order of user
interactions, not just their set. Sun et al. (CIKM 2019) extended this
with bidirectional context in BERT4Rec. Both are relevant for future
work if user interaction sequences become available.

He et al. (SIGIR 2020) showed that graph convolution for recommendation
can be dramatically simplified by removing feature transformation and
nonlinear activation, leaving only neighborhood aggregation on the user-
item graph. LightGCN achieves 16 percent improvement over previous graph
methods while being simpler and faster. A LightGCN model trained on the
MovieLens 25M interaction graph and initialized with the existing Qwen3
embeddings as node features would add collaborative signal to the system
without the complexity of a full two-tower training pipeline.

Wei et al. (ACM Multimedia 2021) proposed contrastive learning for
cold-start recommendation by maximizing mutual information between
content and collaborative signals via InfoNCE loss. Their approach,
training embeddings so that movies watched together are close while
random pairs are far apart, would produce content embeddings that carry
collaborative signal without needing a separate model.

## 8. Implementation notes

### File layout

```
trekomend/
    main.py                     FastAPI server
    test_api.py                 48 integration tests
    src/
        config.py                Settings, paths, instruction templates
        io.py                    Ollama embedding functions
        faiss_search.py          FAISS IVF-PQ searcher, SQLite lookups, HDF5 cache
        diversity.py             GenreRoundRobin, DPPSelector
        features.py              FeatureBuilder for LightGBM
        train_ranker.py          LightGBM training pipeline
    models/
        ranker_v1.txt            Trained LightGBM model (585 KB)
        ranker_v1.importance.json
    kaggle-kernel-trekomend/     GPU embedding notebooks
    new-kaggle-output/           Large binary files (HuggingFace hosted)
```

### HDF5 read performance

The HDF5 file is 4.8 GB on disk. Sequential access is fast. Random access
to individual movie embeddings through h5py fancy indexing requires the
indices to be in sorted increasing order, a constraint of the HDF5
library. Without sorting, random reads cost about 8 ms each. Fifty reads
for the DPP candidate pool would therefore cost about 400 ms cold. A
2048-entry LRU cache absorbs repeated reads, reducing warm latency to near
zero but the cold start cost on a VPS with limited disk I/O is noticeable.
The default pipeline avoids this by skipping DPP entirely.

### SQLite batch operations

SQLite limits the number of bound variables per query to 999. Batch
lookups with more than 999 movie IDs get chunked into groups of 900.
Genre lookups and metadata fetches use single batched queries. The
metadata database weighs 912 MB and the FTS5 index adds roughly 200 MB.
Page cache on a 4 GB VPS covers about 50 MB, meaning repeated queries
hit memory but cold queries touch disk. A b-tree index on (year,
vote_average) would speed up the browse endpoint, which currently scans
the full table for year range queries at about 2 seconds cold.

### Training the re-ranker

The LightGBM model trains on pseudo-labels derived from a weighted blend
of cosine similarity, rating, genre match, keyword overlap, and year
proximity. These are not ground truth. They are directionally correct
proxies. The model learns which features matter most for this composite
score and surfaces nonlinear interactions the handcrafted formula misses.
The training set is small (60K pairs from 240 query movies), so the model
generalizes well to held-out queries but should not be expected to learn
nuanced user preferences that the pseudo-labels do not encode.

Training takes under a second on CPU. The model is 585 KB. Retraining with
more queries or real user feedback would be straightforward. The training
script accepts a --queries argument for larger runs and outputs the model
plus a JSON feature importance report.

### What worked well

The text template design paid off. The genre classification accuracy of
89.7 percent at 2x baseline confirms the embeddings encode strong genre
structure from the field ordering and priority choices.

The LightGBM re-ranker exceeded expectations. Training on 60K pseudo-labeled
pairs with 15 features produced a model that improves result ordering on
held-out queries, runs in 1 ms, and fits in 585 KB. The round-robin layer
provides near-zero-overhead genre diversification that lifts most queries
from 2-4 to 6-9 unique genres.

The FAISS IVF-PQ index hits the VPS form factor perfectly. 108 MB on disk,
under 200 MB total RAM with the SQLite page cache, 15 ms queries at
nprobe=32, no GPU required. The embedding cache with a persistent HDF5
handle avoids the cold-start penalty for title-based queries.

### What needs work

The browse endpoint is slow on year-ranged queries because the SQLite table
lacks a composite index on (year, vote_average). Adding one would bring the
2-second cold latency down to tens of milliseconds.

The DPP layer cold start is too slow for interactive use. While warm queries
hit the LRU cache and run in about 30 ms, a cold query that must read 50
embeddings from the HDF5 file takes roughly 500 ms. For a VPS with limited
disk I/O, pre-warming the cache on startup or switching to a memory-mapped
approach would help.

Movie deduplication is incomplete. The TMDB dataset contains multiple entries
for the same film under different regional releases and reissues. The
round-robin layer pushes them apart by genre, but some still cluster in the
results. A title-based deduplication filter applied after diversity re-ranking
would clean this up.

Popularity debiasing uses a uniform penalty. The LightGBM feature for
embedding norm currently has zero importance in the trained model, meaning
the re-ranker cannot learn contextual debiasing from the pseudo-label data.
If real user feedback becomes available, this feature would likely gain
significance as the model learns that low-norm, obscure movies are sometimes
good recommendations and sometimes bad metadata.

## 9. References

1. Carbonell, J. & Goldstein, J. (1998). "The Use of MMR, Diversity-Based
   Reranking for Reordering Documents and Producing Summaries." SIGIR.
2. Cormack, G.V., Clarke, C.L., & Buettcher, S. (2009). "Reciprocal Rank
   Fusion Outperforms Condorcet and Individual Rank Learning Methods." SIGIR.
3. Kulesza, A. & Taskar, B. (2012). "Determinantal Point Processes for
   Machine Learning." Foundations and Trends in Machine Learning.
4. Gartrell, M., Paquet, U., & Koenigstein, N. (2016). "Low-Rank
   Factorization of Determinantal Point Processes." KDD. arXiv:1602.05436.
5. Covington, P., Adams, J., & Sargin, E. (2016). "Deep Neural Networks
   for YouTube Recommendations." RecSys.
6. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting
   Decision Tree." NeurIPS.
7. Kang, W.C. & McAuley, J. (2018). "Self-Attentive Sequential
   Recommendation." ICDM.
8. Chen, L., Zhang, G., & Zhou, H. (2018). "Fast Greedy MAP Inference for
   DPP to Improve Recommendation Diversity." NeurIPS. arXiv:1709.05135.
9. Sun, F. et al. (2019). "BERT4Rec: Sequential Recommendation with
   Bidirectional Encoder Representations from Transformer." CIKM.
10. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
    "LightGCN: Simplifying and Powering Graph Convolution Network for
    Recommendation." SIGIR. arXiv:2002.02126.
11. Huang, J.-T. et al. (2020). "Embedding-based Retrieval in Facebook
    Search." KDD. arXiv:2006.11632.
12. Wei, Y. et al. (2021). "Contrastive Learning for Cold-Start
    Recommendation." ACM Multimedia. arXiv:2107.05315.
13. Stitch Fix Engineering (2021). "Stitching Together Spaces for
    Query-Based Recommendations." MultiThreaded Blog.
14. Kusupati, A. et al. (2022). "Matryoshka Representation Learning."
    NeurIPS. arXiv:2205.13147.
15. Li, F. et al. (2024). "Contextual Distillation Model for Diversified
    Recommendation." KDD. arXiv:2406.09021.
16. Meehan, G. & Pauwels, J. (2025). "On Inherited Popularity Bias in
    Cold-Start Item Recommendation." ACM RecSys. arXiv:2510.11402.
17. Ibrahim, C. et al. (2025). "Diversified Recommendations of Cultural
    Activities with Personalized Determinantal Point Processes."
    RecSoGood. arXiv:2509.10392.
