# UNet-GAT Segmentation of Water Channels

This project presents a two-stage deep learning pipeline for detecting **ditches** and **natural streams** in Sweden using LiDAR-derived Digital Elevation Models (DEMs). The method combines a UNet segmentation model with a Graph Attention Network (GAT) designed to refine and correct UNet predictions at a graph-based level. The goal is to support more accurate mapping of small water features, which remain underrepresented in current national datasets.

---

## Background & Motivation

Small water channels,especially narrow ditches and streams, are critically important for forestry, agriculture, hydrology, and environmental protection. However, only a fraction of these features appear in Swedish national maps. High-resolution LiDAR data provides an opportunity to automatically detect them, but the task is challenging due to:

- extreme class imbalance  
- thin or fragmented water features  
- differences between man-made and natural channels  
- ambiguous topographical patterns  

This project builds on earlier UNet-based approaches by adding a graph-based refinement stage, aiming to reduce misclassification between streams and ditches. :contentReference[oaicite:2]{index=2}

---

## Method Summary

### **1. UNet for Semantic Segmentation**
- Input: slope-derived DEM tiles (500×500 m areas)  
- 10-fold stratified cross-validation  
- Trained using **Tversky Loss** to address the heavy class imbalance  
- Outputs pixel-level predictions for background, ditch, and stream pixels  

**Performance:**  
- **F1 (ditches):** 0.75  
- **F1 (streams):** 0.39  

### **2. GAT for Graph-Based Refinement**
- Super-pixels (SLIC segments) form nodes  
- Node features include slope statistics, flow accumulation, wetness index, area, and UNet probability summaries  
- Five-layer GAT with multi-head attention  
- Produces refined node-level classifications and reconstructs a corrected segmentation map  

**Performance:**  
- **F1 (ditches):** 0.76  
- **F1 (streams):** 0.41  

---

## Key Findings

- The GAT provides **slight improvements**, but Bayesian analysis shows these fall within a **Region of Practical Equivalence** (±0.05), meaning the gains are **not practically meaningful**.  
- Both approaches **outperform older baselines**, but do not surpass uncertainty-aware methods from previous work.  
- Ditches benefit from relatively consistent patterns and more frequent representation.  
- Streams remain challenging due to rarity, complexity, and extreme class imbalance.  

These insights align with the overall conclusion: the GAT can refine UNet outputs but cannot substantially correct missing or weak predictions.

---

## Dataset Summary

- 4,593 slope-derived DEM tiles  
- Pixel-level labels for background, ditch, and stream classes  
- Strong imbalance:  
  - ~99% background  
  - ~1% ditch  
  - ~0.5% stream  
- 10-fold cross-validation stratified by region and class frequency

---

## Evaluation

Metrics used:

- Precision  
- Recall  
- F1-score  
- Intersection-over-Union (IoU)  
- Matthews Correlation Coefficient (MCC)  

A Bayesian paired t-test was applied due to correlated fold-wise performance.  
With a ROPE of ±0.05, the performance difference between UNet and GAT was **not practically significant** across all classes.

---

## Conclusions

- A two-stage UNet → GAT pipeline is effective but **does not meaningfully outperform** UNet alone.  
- The refinement step is limited by the quality of UNet predictions, since missing features cannot be recovered graphically.  
- Severe imbalance and single-pixel water features remain the main obstacles.  
- Despite this, the method performs comparably to several earlier deep learning approaches.

---

## Future Improvements

Potential directions include:

- Constructing graphs directly from DEMs instead of UNet predictions  
- Incorporating **uncertainty quantification** in the UNet stage  
- Using additional topographical indices beyond slope  
- Improving label generation to reduce single-pixel artifacts  
- Exploring end-to-end hybrid CNN–GNN architectures

---

## Authors
1. Mahmut Osmanovic
2. Isac Paulsson 

