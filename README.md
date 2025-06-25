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

## Using Your Own Dataset

To use your own dataset with this project, follow these steps:

1. **Download Datasets**  
    - For `tieredimagenet` and `inat19`, refer to the instructions from [fiveai/makingbettermistakes](https://github.com/fiveai/makingbettermistakes).
    - A tiny version of `tieredimagenet` is provided (`data/datasets/tieredimagenet_tiny.zip`). Unzip it as shown above.

2. **Prepare Your Dataset**  
    - Place your dataset in `data/datasets/` following the [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) structure:
      ```
      data/datasets/
         └─ your_dataset/
              └─ test/
                    ├── class1/
                    ├── class2/
                    └── ...
      ```

3. **Add a Dataset Configuration**  
    - Create a config file at `configs/datasets/config_your_dataset.json` with the following structure:
      ```json
      {
         "name": "your_dataset",
         "class_to_idx": "data/hierarchies/your_dataset/your_dataset_class_to_idx.pkl",
         "idx_to_name": "data/hierarchies/your_dataset/your_dataset_idx_to_name.pkl",
         "hierarchy_idx": "data/hierarchies/your_dataset/your_dataset_hierarchy_idx.pkl",
         "path_dataset": "data/datasets/your_dataset",
         "path_dataset_test": "data/datasets/your_dataset/test"
      }
      ```
      - **name**: Name of your dataset.
      - **class_to_idx**: Path to a pickle file mapping leaf class names to indices, e.g., `{"class1": 0, "class2": 1, ...}` (indices should be consecutive from 0 to *num_leaf_classes*-1).
      - **idx_to_name**: Path to a pickle file mapping indices to class names, e.g., `{0: "persian_cat", 1: "siamese_cat", ...}` (indices should cover all nodes in the hierarchy).
      - **hierarchy_idx**: Path to a pickle file defining the hierarchy as a dictionary, e.g., `{4: [3, 2], 3: [0, 1]}` (keys are parent indices, values are lists of child indices).
      - **path_dataset**: Path to your dataset root.
      - **path_dataset_test**: Path to your test set.

4. **Run the Interface**  
    - Your dataset will now be available in the interface for selection and evaluation.

## Using Your Own Model



## Using Your Own Decoding Strategies

Several decoding strategies are already implemented in `hierulz/heuristics`, including:

- [Confidence Threshold](hierulz/heuristics/confidence_threshold.py) ([Valmadre, 2022](https://arxiv.org/pdf/2210.10929))
- [CRM-BM](hierulz/heuristics/crm_bm.py) ([Karthik et al., 2021](https://arxiv.org/abs/2104.00795))
- [Expected Information](hierulz/heuristics/expected_information.py) ([Deng et al., 2012](https://ieeexplore.ieee.org/document/6248086))
- [HIE](hierulz/heuristics/hie.py) ([Jain et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/c81690e2cfe63aede8519ad448f56d71-Paper-Conference.pdf))
- [Information Threshold](hierulz/heuristics/information_threshold.py) ([Valmadre, 2022](https://arxiv.org/pdf/2210.10929))
- [Plurality](hierulz/heuristics/plurality.py) ([Valmadre, 2022](https://arxiv.org/pdf/2210.10929))
- [Top-Down](hierulz/heuristics/top_down.py)
- [Argmax Leaf](hierulz/metrics/accuracy.py) ([Valmadre, 2022](https://arxiv.org/pdf/2210.10929))

You can use these strategies as provided, or add your own custom decoding strategy by following these steps:

1. **Implement Your Heuristic**  
    - Create a new file, e.g., `your_heuristic.py`, in the `hierulz/heuristics/` directory.
    - Define your heuristic as a class that inherits from the base [Heuristic](hierulz/heuristics/base_heuristic.py) class.

2. **Add a Heuristic Configuration**  
    - Create a JSON configuration file, e.g., `configs/heuristics/your_heuristic.json`, with the following structure:
      ```json
      {
         "heuristic": "your_heuristic_name",
         "kwargs": {
            "your_argument_to_init_your_heuristic": "value"
         }
      }
      ```
    - Replace `"your_heuristic_name"` with the class name of your heuristic, and specify any required initialization arguments in `kwargs`.

3. **Run the Interface**  
    - Your custom heuristic will now appear in the interface for selection and evaluation.

This modular approach makes it easy to experiment with and compare different decoding strategies within the provided interface.                          

## Using Your Own Metric

Several evaluation metrics are already implemented in `hierulz/metrics`, including:

- [Accuracy](hierulz/metrics/accuracy.py) (on leaf nodes)
- [Hierarchical F<sub>β</sub> score](hierulz/metrics/hf_beta_score.py) ([Kosmopoulos et al., 2013](https://arxiv.org/pdf/1306.6802))
- [Mistake Severity](hierulz/metrics/mistake_severity.py) (also called the shortest path metric, see [Bertinetto et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bertinetto_Making_Better_Mistakes_Leveraging_Class_Hierarchies_With_Deep_Networks_CVPR_2020_paper.pdf))
- Any metric comparing a node prediction to a leaf ground truth (see [Node2Leaf](hierulz/metrics/node2leaf_metric.py)), such as:
    - Wu-Palmer metric ([Wu & Palmer, 1994](https://aclanthology.org/P94-1019/))
    - Zhao similarity ([Zhao et al., 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhao_Open_Vocabulary_Scene_ICCV_2017_paper.pdf))
- Any metric comparing a leaf prediction to a leaf ground truth (see [Leaf2Leaf](hierulz/metrics/leaf2leaf_metric.py)).

You can use these metrics as provided, or add your own custom metric by following one of the options below:

### Option A: Precomputed Metrics (Recommended for Most Use Cases)

1. **Implement Your Metric**  
     - If your metric can be expressed as a function comparing either:
         - a node prediction to a leaf ground truth (Node2Leaf), or
         - a leaf prediction to a leaf ground truth (Leaf2Leaf),
     - Precompute the metric over the hierarchy and save it as a pickle file in `data/metrics/your_dataset_your_metric.pkl`.
     - The file should contain a NumPy array of shape `(num_nodes, num_leaf_classes)` for Node2Leaf (or `(num_leaf_classes, num_leaf_classes)` for Leaf2Leaf), normalized between 0 (perfect match) and 1 (no match).

2. **Add a Metric Configuration**  
     - Create a JSON config file, e.g., `configs/metrics/your_metric.json`, with the following structure:
         ```json
         {
             "tieredimagenet": {
                 "metric_name": "your_metric_name",
                 "kwargs": {
                     "cost_matrix": "data/metrics/your_metric_tieredimagenet.pkl"
                 }
             },
             "your_dataset": {
                 "metric_name": "your_metric_name",
                 "kwargs": {
                     "cost_matrix": "data/metrics/your_metric_your_dataset.pkl"
                 }
             }
         }
         ```
     - Each dataset should have its own entry, as the metric may differ across datasets.

### Option B: Custom Metrics (For Metrics That Cannot Be Precomputed)

1. **Implement Your Metric**  
     - Create a new file, e.g., `your_metric.py`, in the `hierulz/metrics/` directory.
     - Define your metric as a class inheriting from the base [Metric](hierulz/metrics/base_metric.py) class.

2. **Add a Metric Configuration**  
     - Create a JSON config file, e.g., `configs/metrics/your_metric.json`, with the following structure:
         ```json
         {
             "tieredimagenet": {
                 "metric_name": "your_metric_name",
                 "kwargs": {
                     "your_argument_to_init_your_metric": "value"
                 }
             },
             "your_dataset": {
                 "metric_name": "your_metric_name",
                 "kwargs": {
                     "your_argument_to_init_your_metric": "value"
                 }
             }
         }
         ```
     - Replace `"your_metric_name"` with your metric class name and specify any required initialization arguments in `kwargs`.

3. **Run the Interface**  
     - Your custom metric will now be available in the interface for selection and evaluation.

This modular design makes it easy to experiment with and compare different evaluation metrics within the provided interface.