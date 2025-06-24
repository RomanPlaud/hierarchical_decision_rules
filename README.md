# To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers 🔍

**Official implementation of [To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers](https://openreview.net/forum?id=5zsBvPOIUQ), ICML 2025.**

---

## Installation

Install the package in editable mode:

```bash
git clone https://github.com/RomanPlaud/hierarchical_decision_rules.git
pip install -e .
```

## Interface Demo

To explore our demo and visualize the results of different decoding strategies, follow these steps:

1. Unzip the dataset:
    ```bash
    unzip data/datasets/tieredimagenet_tiny.zip -d data/datasets/
    ```
2. Ensure the directory structure matches:
    ```
    data/
      └─ datasets/
          └─ tiered_imagenet_tiny/
              └── test/
                  ├── n01440764/
                  │     └── ILSVRC2012_val_00021740.JPEG
                  ├── n01443537/
                  │     └── ILSVRC2012_val_00002848.JPEG
                  └── ...
    ```
3. Launch the interface:
    ```bash
    python3 scripts/interface.py
    ```
4. Using the Interface

Once the interface is running, follow these steps:

| Step | Action                                                                                       |
|------|----------------------------------------------------------------------------------------------|
| 1    | **Select the dataset:** Choose `tieredimagenet_tiny` from the dropdown.                      |
| 2    | **Load an image:** Click **Load Random Image** to display a sample.                          |
| 3    | **Adjust blur:** Use the slider to set the blur level, then click **Apply Blur**.            |
| 4    | **Choose model:** Select a model from the available list.                                    |
| 5    | **Select metric:** Pick a metric to optimize. If you choose `hFß`, specify the ß value.      |
| 6    | **Pick decoding method:** Choose your preferred decoding strategy.                           |
| 7    | **Decode:** Click **Decode Proba**. Predictions will be shown: <span style="color:green">green</span> for correct, <span style="color:red">red</span> for incorrect. |

This interactive workflow helps you compare decoding strategies and metrics visually.