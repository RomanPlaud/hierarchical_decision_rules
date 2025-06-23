# To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers 🔍

**Official implementation of [To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers](https://openreview.net/forum?id=5zsBvPOIUQ), ICML 2025.**

---

## Installation

# Install the package in editable mode

```bash
git clone https://github.com/RomanPlaud/hierarchical_decision_rules.git
pip install -e .
```

# Interface demo 
To run our demo and get visual understanding of results of our decoding strategie follow the following instructions :
1. 
```bash 
unzip data/datasets/tieredimagenet_tiny.zip -d data/datasets/
```
2. Verify the directory structure:
```kotlin
data/
  └─ datasets/
      └─ tiered_imagenet_tiny/
          └── test/
            ├──n01440764/
                └──ILSVRC2012_val_00021740.JPEG
            ├──n01443537/
                └──ILSVRC2012_val_00002848.JPEG
            └── ...
```
3. Run :
```
python3 scripts/interface.py
```
4. Once the interface shows : 
You can 