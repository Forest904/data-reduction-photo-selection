# Data Reduction — Lecture Summary and Full Practical Assignment

**Lecturer:** Yannis Velegrakis, Utrecht University  
**Lecture source:** `Data Reduction.pdf`  
**Assignment source:** `Data Reduction Assignment.pdf`

> The first part is a structured summary of the complete lecture. The second part reproduces the practical assignment in full, retaining the original wording, notation, links, capitalization, and apparent typographical errors wherever possible.

## Table of contents

1. [Lecture summary](#part-i--lecture-summary)
2. [Practical assignment — full transcription](#part-ii--practical-assignment-verbatim)

---

# Part I — Lecture summary

## 1. Why data reduction matters

The lecture begins by challenging the assumption that “more data is always better.” Global data creation and storage have reached extraordinary scales: estimates for 2025 range around 180–200 zettabytes. One zettabyte is $10^{21}$ bytes, or one billion terabytes. At this scale, storage is not merely a technical concern; it becomes an economic, environmental, legal, operational, and analytical problem.

Data originates from many sources, including internet users, employees and data engineers, and machine-generated systems. It is stored across three broad locations:

- **Core:** centralized data centers and cloud infrastructure.
- **Edge:** intermediate infrastructure close to where data is produced or consumed.
- **Endpoints:** phones, sensors, laptops, vehicles, and other user or IoT devices.

Organizations store data for legitimate reasons such as security monitoring, incident response, auditability, accountability, fraud detection, legal and regulatory compliance, disaster recovery, and root-cause analysis. However, they also store more than they need because storage costs have fallen, deleting data takes effort, legislation is sometimes misunderstood, and organizations fear losing something that might later become useful.

## 2. Consequences of over-storage

### 2.1 Laborious data use

A large data collection does not automatically yield value. Data must be acquired, stored, cleaned, annotated, integrated, aggregated, explored, queried, modeled, and analyzed. Every additional dataset may increase the effort required at each stage. Without effective organization and reduction, a data lake can become a **data swamp**.

### 2.2 Space consumption

Even though the price per unit of storage has declined, total demand is growing faster than infrastructure capacity. Data-center capacity, storage supply, and physical space are finite. The lecture argues that the world is already confronting practical limits in how much data can be stored and served efficiently.

### 2.3 Reduced performance and information overload

Data often needs to be available whenever and wherever it is required, sometimes in real time. Information and communication technologies have throughput, memory, latency, and processing limits. Humans also have cognitive limits. Excessive data can therefore slow systems, complicate decisions, and make relevant information harder to find.

### 2.4 Low-quality data and poor AI

Large collections frequently contain:

- missing values;
- noise;
- outdated values;
- outliers;
- inconsistent values;
- unreliable sources;
- duplicate data.

Poor-quality data damages analytics and machine learning. The lecture emphasizes that generative AI is particularly sensitive because it is probabilistic, can create rather than retrieve facts, may invent nonexistent links, and may misinterpret context and intent. More data is not useful when the additional data is unreliable or misleading.

### 2.5 Privacy and security risks

Every stored item must be protected. Retaining unnecessary data enlarges the attack surface, increases the impact of breaches, and creates legal and ethical exposure. Sensitive data amplifies potential harm. Regulations such as the GDPR and CCPA reinforce the principle that organizations should retain only what is necessary.

### 2.6 Dark data

**Dark data** is information collected and stored during normal activity but not used for analytics, business relationships, or monetization. It may be retained only for compliance. Its storage and security costs can exceed its value, while also introducing risk.

### 2.7 Environmental and energy footprint

Cloud and data-center infrastructure consumes substantial electricity and creates carbon emissions. Storage, networking, computing, and cooling all contribute. The lecture cites estimates such as 40 kg CO₂e per terabyte stored in a U.S. data center and notes that storage can account for a significant share of total data-center power use. Reducing unnecessary data is therefore also a sustainability measure.

## 3. Defining data reduction

Given:

- a dataset $D$;
- a set of constraints $C$ describing what, how much, and when to retain or dispose;

we seek a subset $D' \subseteq D$ that satisfies the constraints and maximizes a utility function. The topic is related to terms such as **data forgetting**, **data amnesia**, **data rotting**, **data degradation**, and **data sustainability**.

The key questions are:

- What makes a data item important?
- What is the cost of retaining or disposing of it?
- Which part of an item is important?
- When can it be disposed of?
- How does importance change over time?
- By what mechanism should disposal occur?
- How will disposal affect the retained data and later analyses?

## 4. Taxonomy of data-reduction methods

The lecture organizes methods into four major families:

1. **Selective storage**
2. **Compression**
3. **Removal of redundancy**
4. **Deletion**

Each family contains several more specific techniques.

---

## 5. Selective storage

Selective storage keeps a compact representation instead of all raw data.

### 5.1 Statistical summarization

Descriptive statistics such as the mean, median, mode, standard deviation, minimum, maximum, and quantiles capture important properties of a dataset without preserving every observation.

### 5.2 Histograms

A histogram approximates a numerical distribution by dividing the value range into intervals, or **bins**, and recording the number of observations in each bin. It trades exact values for a compact approximation of the distribution.

### 5.3 Sketches

Sketches are compact, often probabilistic data structures intended for limited-memory or high-speed streaming settings. They sacrifice some accuracy for substantial gains in efficiency. Examples include:

- **Count-Min Sketch:** approximate item frequencies;
- **HyperLogLog:** approximate the number of distinct elements;
- **Bloom filter:** probabilistic membership testing;
- **quantile sketch:** approximate ranks or percentiles.

### 5.4 Coresets

A **coreset** is a small, often weighted subset that approximates the behavior of the full dataset for a specific objective such as clustering or regression. It exploits redundancy, structure, and smoothness. For example, many points concentrated around three centers can be represented by those centers with corresponding weights.

### 5.5 Wavelets

Wavelets represent signals compactly and hierarchically. A wavelet transform decomposes a signal into scaled and shifted versions of a mother wavelet. Keeping only the most important coefficients creates a lossy but efficient approximation.

### 5.6 Incremental view maintenance

Instead of repeatedly storing or recomputing all derived results, systems can maintain selected materialized views and update them incrementally when source data changes. This preserves useful query results while reducing repeated computation and redundant storage.

---

## 6. Compression

Compression replaces repeated or predictable raw data with a more compact representation. It can be **lossless**, allowing perfect reconstruction, or **lossy**, allowing approximation.

### 6.1 Syntactic or pattern-based compression

Syntactic compression uses repeated symbol patterns without needing to understand their meaning.

#### Huffman encoding

Huffman encoding assigns shorter bit strings to frequent symbols and longer strings to rare symbols. A binary tree is built from symbol frequencies, reducing the total number of bits needed when frequencies are uneven.

#### Lempel–Ziv

Lempel–Ziv methods use a sliding window to find repeated sequences and replace them with references to previous occurrences. A typical representation records the distance to the earlier occurrence, the match length, and the next character.

### 6.2 Dimensionality reduction

High-dimensional data may actually lie in a lower-dimensional subspace. If rows or observations can be reconstructed from a small number of basis vectors, the system stores low-dimensional coordinates rather than every original feature. This principle underlies matrix factorization and related methods.

### 6.3 Semantic compression

Semantic compression uses the meaning and structure of the data. A model $M$ is created and data is divided into:

- values derivable from $M$;
- representative data required to construct $M$;
- outliers not explained by $M$.

Only representative rows, exception information, and a compact agreement bitmap may need to be retained.

### 6.4 Raising abstraction

Data can also be compressed by moving to a higher abstraction level:

- **hierarchical abstraction:** aggregate fine-grained categories into broader concepts, such as months into quarters or quarters into years;
- **uncertainty-based abstraction:** replace precise values with generalized or uncertain values, sacrificing exact query accuracy to save space.

---

## 7. Removing redundancy

Redundancy can occur in schemas and in data instances.

### 7.1 Schema duplication

Different schemas may encode the same concept repeatedly. Schema mappings can transform source schemas into a less redundant target schema while preserving required information and relationships.

### 7.2 Data overlap

Two tables or datasets may contain overlapping columns, entities, or rows. Detecting the largest overlap allows duplicated content to be factored out or merged rather than stored twice.

### 7.3 Entity linkage and deduplication

Entity linkage identifies different representations of the same real-world entity and merges them into a unified record. A typical workflow is:

1. compute pairwise similarity between records;
2. cluster similar records;
3. merge each cluster into a clean representative record.

This process reduces instance duplication while improving data quality.

---

## 8. Deletion strategies

Deletion is the most direct form of reduction. The lecture divides deletion into distribution-based, property-based, and usage-based approaches.

### 8.1 Distribution-based deletion: sampling

Sampling retains a subset according to a policy.

#### Probabilistic sampling

- **Random sampling:** every element has a random chance of selection.
- **Stratified sampling:** divide data into similar groups and sample from each group.
- **Systematic sampling:** choose a starting point and then take every fixed-step item.
- **Cluster sampling:** form clusters and select entire clusters; efficient but potentially less accurate.
- **Multistage sampling:** sample through nested levels or subgroups.

#### Non-probabilistic sampling

- **Convenience sampling:** keep what is readily available.
- **Purposive sampling:** user-guided selection, with a risk of bias.
- **Quota sampling:** enforce rule-based representation quotas.
- **Referral or snowball sampling:** expand through connections or acquaintances.

### 8.2 Property-based deletion

#### Removing low-quality data

Data can be removed because it is missing, noisy, outdated, inconsistent, duplicated, unreliable, or an outlier. This can improve both compactness and analytical quality.

#### Recognizing and mitigating bias

Bias may originate in data generation, collection, institutional processes, sensitive attributes, or unrepresentative samples. Bias can be addressed during preprocessing, model training, or post-processing. The lecture also highlights legal constraints on modifying data and the need to explain and account for bias.

#### Evaluation through facets

Data value can be assessed through facets such as quality, format, and sensitivity. Each sub-facet receives a score in $[0,1]$, and scores are aggregated using weighted utility. This creates a structured basis for deciding what to retain.

### 8.3 Temporal amnesia

A database with amnesia gradually forgets data over time according to a decay function. Temporal strategies include:

- **uniform reservoir sampling**;
- **retrograde or FIFO behavior**, favoring recent data;
- **anterograde behavior**, prioritizing historical data.

Spatial amnesia can remove “infected” or undesirable regions of the data space. The guiding idea is to transform raw data into useful information quickly, before the raw data becomes stale or “rotten.”

### 8.4 Query-based amnesia

Usage-based deletion removes data that is rarely or no longer accessed. Policies include:

- least or most frequently used (LFU/MFU);
- least or most recently used (LRU/MRU);
- other query-history-based rules.

This treats observed workload as evidence of data value.

---

## 9. Formalizing data reduction as optimization

Deletion-based reduction can be expressed as selecting a subset that maximizes utility under a budget:

$$
D^* = \arg\max_{D'\subseteq D,\ |D'|\le B} f(D').
$$

A more general form uses a cost function:

$$
D^* = \arg\max_{D'\subseteq D,\ C(D')\le B} f(D'),
$$

where

$$
C(D') := \sum_{d\in D'} C(d).
$$

A set function is a mapping $f:\mathcal{P}(D)\to\mathbb{R}$, where $\mathcal{P}(D)$ is the power set of $D$.

- It is **monotone** when $A\subseteq B$ implies $f(A)\le f(B)$.
- It is **non-negative** when $f(A)\ge 0$ for all $A\subseteq D$.
- It is **modular** when each element has the same marginal contribution regardless of the existing set. With a modular objective and a budget, the selection problem is closely related to knapsack.
- It is **submodular** when it has diminishing returns: adding an item to a smaller set gives at least as much marginal gain as adding it to a larger set.

For $A\subseteq B\subseteq D$ and $d\in D\setminus B$:

$$
f(A\cup\{d\})-f(A) \ge f(B\cup\{d\})-f(B).
$$

Submodularity is important because many coverage and representativeness objectives satisfy it and admit efficient approximation algorithms.

## 10. Representativeness, coverage, and diversity

A common data-value objective is representativeness: a data point is important when it represents a large portion of the dataset. A good retained subset should provide both **coverage** and **diversity**.

One simple coverage objective sums similarity between every original point and retained points. Thresholded variants prevent heavily represented regions from dominating and encourage diverse coverage. Such objectives are often submodular because the benefit of adding another similar item decreases as the selected set grows.

## 11. Query-aware and workload-aware value

The lecture further connects data value to the queries users actually ask. A retained dataset is useful when answers on the reduced dataset remain similar to answers on the complete dataset. Query logs can therefore be treated as a distribution over information needs.

This leads to objectives that compare $q(D')$ with $q(D)$ for queries $q$ drawn from a query distribution. Similarity may be measured through cosine similarity, Jaccard similarity, or another task-specific metric. Workload-aware reduction links directly to query-based amnesia and the practical assignment.

## 12. Shapley values for data importance

Shapley values originate in cooperative game theory and assign each data item an average marginal contribution across all possible coalitions or subsets. They provide a principled notion of individual importance, but exact computation is expensive because it considers exponentially many subsets. Approximation or sampling is usually required for large datasets.

## 13. Stochastic submodular data forgetting

The final lecture material applies the preceding concepts to a photo-deletion problem. Photos are represented by embedding vectors, and a query log records which photos were returned for previous searches. The goal is to retain a limited subset while preserving expected query-answer quality.

The utility of a retained set $D'$ can be defined as the expected similarity between answers from the original and reduced datasets. Under suitable similarity functions, the objective has submodular structure and can be optimized approximately. The lecture contrasts exhaustive subset evaluation, greedy or independent scoring methods, Shapley-value selection, and student-designed alternatives.

## 14. Main conclusions

The lecture’s central message is that data should not be stored merely because storage appears cheap. Every item has ongoing costs in space, energy, security, governance, processing, and human attention. Effective data reduction therefore requires:

- an explicit utility model;
- a cost or budget constraint;
- awareness of quality, bias, privacy, and time;
- a suitable technique, such as summarization, compression, deduplication, sampling, or deletion;
- evaluation of how reduction affects downstream queries and models.

The best retained dataset is not necessarily the largest one. It is the smallest dataset that preserves the information, coverage, diversity, and task performance that matter.

---

# Part II — Practical assignment (verbatim)

## Page 1


# Data Reduction Practical

Yannis Velegrakis (Utrecht University)

**DEADLINE: June 15th 2026**

Your phone is becoming full of photos and is running out of space

You need to (unfortunately) delete a number of them. The goal of this practical is to devise a method to select the right photos to delete. To do each photo has been represented as an embedding vector in $\mathbb{R}^n$, e.g.,

`[0.0013208601,0.004669265,…,0.12525679,0.0]`

(This representation is already done for you. No need to do it. After all, there are multiple libraries that can do such a task based on the photo content or the metadata). The photos are then represented in a csv file (called photos.csv) where every line is a vector that corresponds to a single photo. Each photo has a unique identifier which is the line its vector has in the photos.csv file. For instance, the embedding vector of the photo with id number 73 is the vector in line 73 of the photos.csv file. The photo file can be found in this directory:

https://drive.google.com/drive/u/0/folders/1gXlpzEduvb1RZAGlyTLsitZC2WPWw2I4

In addition to the photos, we also have a query log that is the set of searches that you have so far placed on your photos. Each search you do generates a set of images as a response. The query log is in a file called queries.csv (you can find it in the aforementioned directory). Each line of the file corresponds to a search and contains the id of each of the images that have been returned as an answer to the search. For instance, the line

`[3272,38538,19626,32654,30990,21396]`

means that the images in lines 3272, 38538, 19626, 32654, 30990, and 21396 of the file photos.csv were returned.

---

## Page 2


Let D represent the set of photos that you have (i.e., the photos.csv file) and a budget B to be the maximum number of pictures you can keep on your phone. You are asked to devise some strategies to decide which photos to delete. To do so, you need to come out with a way to evaluate how useful a subset D’ of D is.

### Method A:

This method is based and described in the SIGMOD 2026 paper “Stochastic Submodular Data Forgetting” ( https://velgias.github.io/docs/RicoSV26.pdf ) . The function f is the utility function that indicates how useful the subset of photos D’ is by looking at different queries (q) drawn from the query distribution Q (the query set) and computing how similar is the answer one gets when asking the query on the reduced dataset (i.e., q(D’) ), compared to the full dataset D ( q(D) )

$$
f(D') = \mathbb{E}_{q\sim Q}\left[S(q(D),q(D'))\right]
$$

$$
S(q(D),q(D')) := \frac{1}{|q(D)|}\cdot \sum_{d\in q(D)}\max_{d'\in q(D')} \operatorname{sim}(d,d'),
$$

$$
f(D') = \mathbb{E}_{q\sim Q}\left[\frac{1}{|q(D)|}\cdot\sum_{d\in q(D)}\max_{d'\in q(D')}\operatorname{sim}(d,d')\right]
$$

$$
=\sum_{q\sim Q}\sum_{d\in q(D)}\frac{\mathbb{P}(q)}{|q(D)|}\cdot\max_{d'\in q(D')}\operatorname{sim}(d,d')
$$

As similarity between the two images d and d’, i.e., sim(d,d’), one could use the cosine similarity. The solution means to consider for a dataset D all the possible subsetsD’ and compute the utility f(D’). There are of course way too many subsets, thus, you are asked to consider only subsets that have half as many items as D. That basically means deleting 3 images only.

### Method B:

Instead of the above approach that has a high cost, we can consider as similarity (sim(d,d’)) the Jaccard (and not the cosine). In that case, an approximate algorithm which is greedy exists and is called IndepDF (described in detail in the paper mentioned above).

**Algorithm 3: IndepDF**

- **Data:** Dataset D, queryset Q with distribution Q, and budget B ∈ N.
- **Result:** Subset D′ ⊆ D with |D′| ≤ B (exactly) optimizing objective (1) for S(·,·) = Jaccard(·,·).

```text
1 for d ∈ D do
2     score(d) ← E_{q~Q} [ I_q(d)/|q(D)| ];
3 end
4 return Top B points d ∈ D with highest score(d);
```

In that case one can set the budget B on the number of pictures to keep much less. (say for instance half the number of images. )

---

## Page 3


### Method C:

Assume that you have decided to keep 3 photos only. Select the best 3-photo set by considering as the value f of the set, to be the sum of the Shapley values of the photos it contains. If you have enough computational power, you can also try to calculate the same for a 4-photos set.

### Method D:

Describe an idea of yours on how one could decide the best subset of images to keep.

What is expected from you is

1. A full description of the idea you propose (Method D). Provide how it works and analyze how it compares to the methods A-C.
2. An implementation of the methods A-D. Generate a github repository and place there the code of your implementation. Share it with the user velgias@gmail.com
3. The execution of a set of experiments and a study of the performance and quality of the methods A-D. Are they consistent with the theoretical complexity of those methods ? How do they company in terms of performance and quality?

Delivery methods:

- Create a github repository with your code and share it with the user velgias@gmail.com The repository should contain the #2 above.
- Create an overleaf repository and share it with the users velgias (velgias@gmail.com) and riccardo.torlone (riccardo.torlone@uniroma3.it). In that repository place your report. Your report should follow the following template: https://www.overleaf.com/latex/templates/association-for-computing-machinery-acm-sig-proceedings-template/bmvfhcdnxfty The report should contain the #1 and #3 points that are expected from you (see above).
- Send an email to the professors (i.velegrakis@uu.nl and riccardo.torlone@uniroma3.it) with the Subject title: Roma3 Data Reduction. The email should contain your full name, matricola, a link to your code in github, and a link to your overleaf repository.

For any questions, points that are not clear or any other issue, do not hesitate to contact the instructor of this course Prof. Yannis Velegrakis at i.velegrakis@uu.nl
