# CONTEXT.md — SD-FSProto

## Project identity

**SD-FSProto** (Static-Dynamic Few-Shot Prototypical Learning for PE Malware Family Classification) is a research project that classifies malware into families when only 1–5 labeled samples are available per novel family. It learns a joint embedding space from static PE features and dynamic sandbox behavior, then uses prototypical networks to generalize to unseen families.

All documentation and domain vocabulary are in Vietnamese (see glossary below). The project is in the research design phase — no implementation yet.

## Entities

### Portable Executable (PE / tệp PE)
A Windows executable file format. The system's static branch analyzes PE files by reading their raw bytes (for grayscale image conversion), metadata headers (for structural features), and optionally import/export tables and embedded strings.

### Sandbox Trace (dấu vết hộp cát / dấu vết động)
The execution log produced by running a PE file in a sandbox (Cuckoo/CAPE). Contains a sequence of API calls made by the process, plus optionally a behavior graph showing relationships between calls (process trees, registry/network/file dependencies).

### Static View (luồng tĩnh / góc nhìn tĩnh)
The representation produced by the Static Encoder: a 512-d embedding vector `z_s` encoding the PE file's structure and content. This view can be unreliable for packed, obfuscated, or armored malware.

### Dynamic View (luồng động / góc nhìn động)
The representation produced by the Dynamic Encoder: a 512-d embedding vector `z_d` encoding the sandbox execution trace. This view can be unreliable when malware detects the sandbox and alters its behavior (evasion).

## Core concepts

### Malware Family (họ mã độc / family)
A group of malware samples sharing a common codebase, author, or behavioral signature. Examples: Allaple.A, Swizzor.gen!E, Agent.FYI. The system must classify samples into families it has never seen during training, using only 1–5 labeled support examples per family at test time.

### Support Set (tập hỗ trợ)
The small set of labeled samples (1–5 per family, K-shot) provided at inference time for novel families. The system computes a prototype from these samples — no gradient update or retraining is performed.

### Query Set (tập truy vấn)
Unlabeled samples to classify by comparing their embeddings to the prototypes computed from the support set.

### Prototype (nguyên mẫu / prototype `p_c`)
The mean embedding of all support samples belonging to family `c`. The query sample is assigned to the family whose prototype is nearest in Euclidean distance. A multi-prototype variant clusters support samples into sub-types when a family has distinct variants.

### Episode (tập / episode)
A single training iteration that samples N families (N-way), K support samples per family (K-shot), and a batch of query samples. Each episode mirrors the test scenario, forcing the model to learn to compare rather than memorize.

### Reliability Score (độ tin cậy)
Scalar weights (`α_s`, `α_d`, `α_sd`) that modulate the contribution of static view, dynamic view, and their interaction in the fused embedding. Learned from metadata: PE header properties (entropy, section characteristics) predict static reliability; sandbox trace metadata (trace length, API call diversity, anomalies) predict dynamic reliability.

### Fused Embedding (phép nhúng kết hợp `z`)
The weighted combination `z = α_s·z_s + α_d·z_d + α_sd·z_sd` where `z_sd` captures cross-view interactions. This is the final representation fed into the prototypical network.

### Family-Disjoint Split (phân chia tách biệt họ)
The mandatory train/test/data split where no malware family appears in more than one partition. This prevents family leakage which would invalidate few-shot evaluation.

### Unknown Sample (mẫu không xác định)
A sample whose embedding distance to all known prototypes exceeds a threshold. Flagged as "unknown" rather than forced into a known family. Essential for operational use where novel malware families are encountered.

## Invariants

1. **Family-disjoint split is mandatory** — a family appearing in both train and test invalidates the few-shot evaluation.
2. **No fixed classifier head** — the system must not use a Softmax over known families; it must classify novel families purely via distance to support-set prototypes.
3. **Reliability gating must be trainable** — the fusion weights must be learned from metadata, not hand-tuned, so the model can adapt to packed/evasive inputs automatically.
4. **Unknown detection is required** — the model must reject samples far from all prototypes rather than hallucinating a family assignment.
5. **Both views can be missing** — a sample may have only static features (no sandbox trace) or only dynamic features (no PE file); the fusion module must handle partial input gracefully.

## Domain language reference

| Vietnamese | English | Used for |
|---|---|---|
| họ mã độc | malware family | The output classes |
| luồng tĩnh | static view / static branch | PE-based encoder pathway |
| luồng động | dynamic view / dynamic branch | Sandbox-based encoder pathway |
| phép nhúng kết hợp | fused embedding | Combined representation `z` |
| độ tin cậy | reliability | Learned attention weights for fusion |
| nguyên mẫu | prototype | Mean embedding per family |
| tập hỗ trợ | support set | Few-shot labeled examples |
| tập truy vấn | query set | Samples to classify |
| tập episode | episode | N-way K-shot training iteration |
| phân chia tách biệt họ | family-disjoint split | Data partitioning constraint |
| mẫu không xác định | unknown sample | Out-of-distribution rejection |
| mạng prototypical | prototypical network | Core classification mechanism |
| học meta theo tập | episodic meta-learning | Training paradigm |
