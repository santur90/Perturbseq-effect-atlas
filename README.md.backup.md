# Genome-scale Perturb-seq Effect Atlas

A reproducible computational framework for modeling CRISPR perturbation effects, identifying functional gene modules, and benchmarking phenotype-based functional prioritization from genome-scale Perturb-seq data.

**Project focus:** Perturb-seq · CRISPR screens · single-cell genomics · perturbation modeling · functional genomics · network analysis · dimensionality reduction · protein complexes · benchmarking · reproducible bioinformatics

This project analyzes the K562 genome-scale CRISPRi Perturb-seq dataset from Replogle et al. to quantify transcriptional perturbation effects and ask a practical functional-genomics question:

> **Can transcriptome-wide perturbation phenotypes recover known gene-function relationships and prioritize held-out CORUM complex annotations?**

The workflow combines perturbation-aware quality control, technical-confounder correction, dimensionality reduction, cosine-similarity networks, Leiden clustering, and independent validation against curated CORUM protein complexes.

---

## Key results

- Processed **11,258 perturbation profiles**; initial QC catalogued approximately **9,867 unique target identifiers**. This is not a count of successfully modeled genes: some identifiers could not be parsed, and downstream analyses apply additional eligibility filters.
- Identified a strong dependency between naïve transcriptome-wide effect magnitude and perturbation cell count
- Corrected this cell-count-dependent background using non-targeting controls
- Recovered coherent functional modules including ribosome, mitochondrial ribosome, proteasome, RNA polymerase II, respiratory complex I, and RNA exosome perturbations
- Constructed gene-level perturbation embeddings for **2,670 responsive genes**
- Evaluated functional recovery across **491 retained CORUM complexes**
- Benchmarked **928 responsive genes with curated CORUM annotations**
- Recovered a same-complex gene as the:
  - **Top-1 neighbor for 42.2%** of queries
  - **Top-5 neighbor for 60.8%**
  - **Top-10 neighbor for 65.7%**
  - **Top-20 neighbor for 71.4%**
- Top-1 recovery was **42.2% versus 0.58%** under 100 gene-label permutations
- Held-out complex prioritization achieved:
  - **Top-1 accuracy: 41.4%**
  - **Top-5 accuracy: 62.0%**
  - **Top-10 accuracy: 67.2%**
  - **MRR: 0.502**
- Pairwise perturbation similarity predicted shared CORUM complex membership with:
  - **AUROC: 0.820**
  - **Average Precision: 0.378**
  - **Positive prevalence: 1.73%**
  - **AP / prevalence: 21.8×**

The held-out complex benchmark ranks CORUM annotations by similarity-weighted votes from phenotype neighbors. The query's CORUM annotation is used to score the ranking, not to generate it. This is a held-out complex-prioritization benchmark, not a claim of de novo or unknown gene-function discovery.

---

## Benchmark summary

![CORUM benchmark](results/figures/07_portfolio_summary.png)

Perturbation-effect similarity substantially outperformed permuted gene labels for recovery of curated protein-complex relationships.

The pairwise benchmark is highly imbalanced, with only 1.73% positive gene pairs. For this reason, Average Precision is emphasized alongside AUROC. The observed AP of 0.378 corresponds to approximately **21.8-fold enrichment over positive prevalence**.

The held-out complex-prioritization benchmark withholds each query's CORUM annotation from the neighbor vote and evaluates the ranked complexes against that annotation. Since the candidate labels come from curated, annotated neighbors, this evaluates recovery and prioritization of known complex membership; it does not establish function for genes with no known annotation.

---

## Biological effect landscape

![Perturbation modules](results/figures/04_perturbation_modules.png)

Genome-scale perturbation phenotypes form structured neighborhoods in transcriptional-effect space.

Known functional gene groups occupy distinct regions of the embedding, including:

- cytosolic ribosome
- mitochondrial ribosome
- proteasome
- respiratory complex I
- RNA polymerase II
- RNA exosome

These annotations are used here as biological sanity checks. Formal benchmarking is performed independently using curated CORUM protein-complex annotations.

---

## Why perturbation-effect QC matters

A naïve transcriptome-wide effect score showed a strong dependency on the number of cells contributing to each perturbation profile:

**Spearman ρ = -0.860**

This indicated that raw effect magnitude could not be interpreted directly as biological perturbation strength.

I therefore modeled the non-targeting-control background as a function of cell count and calculated a cell-count-adjusted perturbation effect score.

After adjustment:

**Spearman ρ = 0.038**

![Adjusted effect score](results/figures/03_adjusted_effect_score.png)

This prevents low-cell-count perturbations from being systematically prioritized simply because their estimated transcriptional profiles are noisier.

The adjusted score is used as a descriptive perturbation-strength metric rather than an inferential statistical z-score.

---

## Strong transcriptional effects improve functional recovery

Known-complex perturbations that were recovered within coherent transcriptional neighborhoods had substantially larger adjusted transcriptional effects than members that remained scattered.

![Recovered versus scattered](results/figures/05_in_module_vs_scattered.png)

Median adjusted effect score:

- recovered/in-module members: **20.74**
- scattered members: **5.11**
- Mann–Whitney P = **3.4 × 10⁻14**

In contrast, the corresponding groups showed no comparable difference in:

- target knockdown efficiency
- number of cells contributing to the perturbation profile

Within this benchmark, recovery of known functional relationships is therefore associated primarily with the strength of the measured transcriptional phenotype rather than greater cell coverage or detectably stronger target knockdown.

---

## Analysis workflow

```text
Replogle et al. genome-scale Perturb-seq
                    │
                    ▼
          Perturbation-level QC
                    │
                    ▼
      Detect technical confounding
                    │
                    ▼
 Non-targeting-control background model
                    │
                    ▼
  Cell-count-adjusted effect strength
                    │
                    ▼
         PCA effect representation
                    │
                    ▼
        Cosine similarity / kNN
                    │
             ┌──────┴──────┐
             ▼             ▼
      Leiden modules   Gene neighbors
             │             │
             ▼             ▼
      Known-complex    Held-out complex
        recovery         prioritization
             │             │
             └──────┬──────┘
                    ▼
             CORUM benchmark
```

---

## Reproducibility and data availability

The analysis is organized as numbered scripts in [`scripts/`](scripts/), with benchmark tables and figures in [`results/`](results/). The source and processed AnnData matrices are excluded from version control because of their size; the raw Perturb-seq matrices must be obtained separately before rerunning the full workflow. CORUM annotations used by the benchmark are provided under [`data/reference/`](data/reference/).

Run the numbered analysis scripts from the repository root in order to regenerate the analysis outputs. The existing outputs in `results/` are the source of the summary figures and metrics reported above.

---

## Limitations

- The approximately 9,867 initial QC target identifiers should not be interpreted as the number of successfully parsed or modeled genes. Downstream response and quality filters reduce the analysis set; the reported gene-level embeddings include **2,670 responsive genes**.
- The CORUM benchmark evaluates recovery of curated protein-complex relationships represented in this dataset. It does not test discovery of completely unknown gene functions.
- The adjusted effect score is descriptive, not an inferential statistical z-score.
- Results reflect one K562 CRISPRi Perturb-seq dataset and the selected QC, dimensionality-reduction, similarity, and CORUM filtering choices; external datasets are needed to assess generalization.

---

## References

If you use the underlying Perturb-seq dataset, please cite:

> Replogle JM, Saunders RA, Pogson AN, et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*. 2022;185(14):2559–2575.e28. [doi:10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013).

For CORUM annotations, cite the CORUM resource publication:

> Tsitsiridis G, Steinkamp R, Giurgiu M, et al. CORUM: the comprehensive resource of mammalian protein complexes—2022. *Nucleic Acids Research*. 2023;51(D1):D539–D545. [doi:10.1093/nar/gkac1015](https://doi.org/10.1093/nar/gkac1015).
