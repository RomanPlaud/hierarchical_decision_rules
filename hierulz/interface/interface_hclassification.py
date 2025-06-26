import sys, os, random, pickle, time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QSizePolicy

)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms

import numpy as np
import pickle as pkl
from PIL import Image
import torch

from hierulz.hierarchy import load_hierarchy
from hierulz.datasets import get_dataset_config, get_default_transform
from hierulz.models import get_model_config, load_model
from hierulz.metrics import load_metric
from hierulz.heuristics import load_heuristic

dico_names_heuristics = {'confidence_threshold': 'Thresholding 0.5',
                         'crm_bm' : '(Karthik et al., 2021)',
                         'expected_information': 'Exp. Information',
                         'hie': 'Hie-Self (Jain et al., 2023)',
                         'information_threshold': 'Information threshold',
                         'plurality': 'Plurality',
                         'top_down': 'Top-down argmax',
                         'argmax_leaves': 'Argmax leaves'} 

class InterfaceHClassification(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hierarchical Image Decoder")
        self._initUI()

    def _initUI(self):
        layout = QVBoxLayout()
        layout.addLayout(self._build_image_hierarchy_layout())
        layout.addLayout(self._build_dataset_controls())
        layout.addWidget(self._build_load_image_button())  
        layout.addWidget(self._build_blur_controls())
        layout.addWidget(self._build_model_controls())
        metric_container, beta_container = self._build_metric_controls()
        layout.addWidget(metric_container)
        layout.addWidget(beta_container)
        layout.addWidget(self._build_decoding_controls())
        layout.addWidget(self._build_prediction_output())
        self.setLayout(layout)

    
    def _build_image_hierarchy_layout(self):
        layout = QHBoxLayout()

        layout.addWidget(self._build_image_side())
        layout.addWidget(self._build_hierarchy_widget("hierarchy_layout", "hierarchy_widget"))
        layout.addWidget(self._build_hierarchy_widget("prediction_layout", "prediction_widget"))

        return layout

    def _build_image_side(self):
        self.image_side_container = QWidget()
        layout = QVBoxLayout(self.image_side_container)

        self.image_label = QLabel()
        self.image_label.setFixedSize(224, 224)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.class_label = QLabel("Class: ")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("font-size: 22pt; font-weight: bold; margin-top: 10px;")

        layout.addWidget(self.image_label)
        layout.addWidget(self.class_label)

        self.image_side_container.setVisible(False)  # Initially hidden if needed
        self.image_side_container.setMinimumHeight(40)
        return self.image_side_container


    def _build_hierarchy_widget(self, layout_attr_name, widget_attr_name):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        widget = QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(250)

        container_layout = QVBoxLayout()
        container_layout.addWidget(widget, alignment=Qt.AlignTop)

        container = QWidget()
        container.setLayout(container_layout)

        setattr(self, layout_attr_name, layout)
        setattr(self, widget_attr_name, widget)

        return container
    
    def _build_load_image_button(self):
        self.btn_load = QPushButton("Load Random Image")
        self.btn_load.clicked.connect(self._load_random_image)
        self.btn_load.setVisible(False)  # Hidden initially
        self.setMinimumHeight(40)  # Set minimum height for the button
        return self.btn_load

    def _build_dataset_controls(self):
        layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        datasets_dir = os.path.join(os.path.dirname(__file__), "../../data/datasets")
        if os.path.isdir(datasets_dir):
            dataset_names = [""] + sorted([
            name for name in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, name))
            ])
        else:
            dataset_names = [""]

        self.dataset_combo.addItems(dataset_names)
        self.dataset_combo.currentTextChanged.connect(self._select_dataset)  # Add a handler if you want

        layout.addWidget(QLabel("Dataset"))
        layout.addWidget(self.dataset_combo)

        return layout


    def _build_blur_controls(self):
        container = QWidget()
        layout = QHBoxLayout(container)

        layout.addWidget(QLabel("Blur level"))

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 200)
        self.blur_slider.setValue(0)
        self.blur_slider.valueChanged.connect(self._update_blur_label)
        layout.addWidget(self.blur_slider)

        self.blur_value_label = QLabel("0.0")
        layout.addWidget(self.blur_value_label)

        self.blur_button = QPushButton("Apply Blur")
        self.blur_button.clicked.connect(self._apply_blur)
        layout.addWidget(self.blur_button)

        # Store the container so you can show/hide everything together
        self.blur_controls_container = container
        self.blur_controls_container.setVisible(False)  # Initially hidden
        self.blur_controls_container.setMinimumHeight(40)  # Set minimum height for the blur controls

        return self.blur_controls_container

    
    def _build_model_controls(self):
        container = QWidget()
        layout = QHBoxLayout(container)

        self.model_combo = QComboBox()
        # self.model_combo.currentTextChanged.connect(self._load_model)

        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.model_combo)

        self.model_controls_container = container
        self.model_controls_container.setVisible(False)  # Initially hidden
        self.model_controls_container.setMinimumHeight(40)

        return self.model_controls_container

    
    def _build_metric_controls(self):
        # --- Metric selection container ---
        self.metric_controls_container = QWidget()
        metric_layout = QHBoxLayout(self.metric_controls_container)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            " ", "hF_ß", "Mistake Severity", "Wu-Palmer", "Zhao",
            "Accuracy", "Hamming Loss", "Wu-Palmer", "Zhao"
        ])
        self.metric_combo.currentTextChanged.connect(self._select_metric)

        metric_layout.addWidget(QLabel("Metric"))
        metric_layout.addWidget(self.metric_combo)
        self.metric_controls_container.setVisible(False)  # Initially hidden
        self.metric_controls_container.setMinimumHeight(40)

        # --- Beta spinbox container ---
        self.beta_controls_container = QWidget()
        beta_layout = QHBoxLayout(self.beta_controls_container)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 9999.0)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(1.0)

        beta_layout.addWidget(QLabel("ß"))
        beta_layout.addWidget(self.beta_spin)
        self.beta_controls_container.setVisible(False)  # Initially hidden
        self.beta_controls_container.setMinimumHeight(40)


        return self.metric_controls_container, self.beta_controls_container


    
    def _build_decoding_controls(self):
        self.decoding_controls_container = QWidget()
        layout = QHBoxLayout(self.decoding_controls_container)

        self.decode_combo = QComboBox()
        heuristics_dir = os.path.join(os.path.dirname(__file__), "../../configs/heuristics")
        if os.path.isdir(heuristics_dir):
            heuristic_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(heuristics_dir)
            if f.endswith('.json')
            ]
            items = ['Optimal'] + sorted(heuristic_files)
        else:
            items = ['Optimal']
        # map heuristic names to user-friendly names
        items = [dico_names_heuristics.get(name, name) for name in items]
        self.decode_combo.addItems(items)

        layout.addWidget(QLabel("Decoding"))
        layout.addWidget(self.decode_combo)

        self.decode_button = QPushButton("Decode Proba")
        self.decode_button.clicked.connect(self._decode_proba)
        layout.addWidget(self.decode_button)

        self.decoding_controls_container.setVisible(False)  # Initially hidden
        return self.decoding_controls_container

    
    def _build_prediction_output(self):
        self.output_label = QLabel("Prediction: ")
        self.output_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.output_label.setVisible(False)  # Initially hidden
        return self.output_label

    def _update_blur_label(self, value):
        self.blur_value_label.setText(f"{value / 10:.1f}")
        # set model container visible when blur is applied


    def _select_dataset(self, dataset_name):
        # You can add logic here to load dataset info, reset UI, etc.
        print(f"Dataset selected: {dataset_name}")
        # For example:
        self.config_dataset = get_dataset_config(dataset_name)
        self.hierarchy = load_hierarchy(dataset_name)
        self.transform = get_default_transform(dataset_name)

        self.class_to_idx = pkl.load(open(self.config_dataset['class_to_idx'], 'rb'))
        self.idx_to_name = pkl.load(open(self.config_dataset['idx_to_name'], 'rb'))

        self.btn_load.setVisible(True)
        self.image_side_container.setVisible(True)  # Show image side when dataset is selected
        self.update_model_combo()  # Update model combo based on selected dataset
        



    def _load_random_image(self):
        path_folder_image = self.config_dataset.get('path_dataset_test', '')
        categories = os.listdir(path_folder_image)
        folder_category = random.choice(categories)
        folder_category_path = os.path.join(path_folder_image, folder_category)
        images = os.listdir(folder_category_path)
        image_name = random.choice(images)
        image_path = os.path.join(folder_category_path, image_name)

        self.current_image_path = image_path
        img = Image.open(image_path).convert("RGB")
        self.original_img = img

        self.display_image(img)
        self.display_label(folder_category)
        self.blur_controls_container.setVisible(True)  # Show blur controls when an image is loaded

    def display_image(self, img):
        img = img.resize((224, 224))
        img_np = np.array(img)
        qimg = QImage(img_np.data, img_np.shape[1], img_np.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)   

    def render_label_path(self, layout, label_ids, idx_to_name, highlight_ids=None, highlight_color="#d4fdd4"):
        """
        Render a hierarchy path given a list of label IDs into the provided layout.
        
        Args:
            layout (QVBoxLayout): where to display
            label_ids (List[int]): list of node IDs in order
            idx_to_name (dict): maps ID to name
            highlight_ids (set): optional set of IDs to highlight (e.g. correct prediction)
            highlight_color (str): background color for highlights
        """
        highlight_ids = set(highlight_ids or [])

        # Clear previous content
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Display hierarchy
        for i, node_id in enumerate(label_ids):
            name = idx_to_name[node_id]
            is_highlighted = node_id in highlight_ids

            node_label = QLabel(name)
            node_label.setMinimumHeight(30)
            node_label.setMaximumHeight(30)
            node_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            node_label.setStyleSheet(f"""
                QLabel {{
                    border: 1px solid #333;
                    border-radius: 6px;
                    padding: 2px;
                    background-color: {'#E48782' if not is_highlighted else highlight_color};
                    font-size: 10pt;
                    qproperty-alignment: AlignCenter;
                }}
            """)
            layout.addWidget(node_label)

            if i < len(label_ids) - 1:
                arrow = QLabel("↓")
                arrow.setMinimumHeight(20)
                arrow.setMaximumHeight(20)
                arrow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                arrow.setStyleSheet("font-size: 12pt; color: gray; qproperty-alignment: AlignCenter;")
                layout.addWidget(arrow)

    def display_label(self, label):
        label_idx = self.class_to_idx[label]
        ancestor_ids = np.where(self.hierarchy.leaf_events[label_idx])[0]
        ancestor_ids_sorted = sorted(ancestor_ids, key=lambda x: self.hierarchy.depths[x])
        self.last_groundtruth_path = ancestor_ids_sorted

        # Reuse the general display logic
        self.render_label_path(self.hierarchy_layout, ancestor_ids_sorted, self.idx_to_name, ancestor_ids_sorted)

        # Set class label on top bar
        self.class_label.setText(f"Class: {self.idx_to_name[ancestor_ids_sorted[-1]]}")



    def _apply_blur(self):
        level = self.blur_slider.value() / 10.0
        if level == 0:
            img = self.original_img
        else:
            blur = transforms.GaussianBlur(kernel_size=61, sigma=level)
            img = blur(self.original_img)
        self.display_image(img)
        self.model_controls_container.setVisible(True)


    def _load_model(self, model_name):
        config_model = get_model_config(model_name)
        self.model = load_model(config_model)
        self.model.eval()
        self._infer_proba()
        self.metric_controls_container.setVisible(True)

    def _infer_proba(self):
        img_transformed = self.transform(self.original_img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(img_transformed)
            self.proba = torch.nn.functional.softmax(output, dim=1).cpu().numpy()


    def _select_metric(self, metric_name):
        # Show beta only if "hF_ß" is selected
        if metric_name == "hF_ß":
            self.beta_controls_container.setVisible(True)
        else:
            self.beta_controls_container.setVisible(False)
        self.metric = load_metric(metric_name, self.config_dataset['name'])
    
        self.decoding_controls_container.setVisible(True)

    def _decode_proba(self):
        decoding_method = self.decode_combo.currentText()
        if decoding_method == "Optimal":
            self.decoding = self.metric
        else: 
            self.decoding = load_heuristic(decoding_method, self.config_dataset['name'])

        # Decode predictions
        # get nodes probas
        probas_nodes = self.hierarchy.get_probas(self.proba)
        # Decode predictions
        y_pred = np.where((self.decoding.decode(probas_nodes))[0])[0]
        # sort by ascending depth
        ancestor_ids_sorted = sorted(y_pred, key=lambda x: self.hierarchy.depths[x])
        print(f"Decoded path: {ancestor_ids_sorted}")
        

        self.render_label_path(
            self.prediction_layout,
            ancestor_ids_sorted,
            idx_to_name=self.idx_to_name,
            highlight_ids=self.last_groundtruth_path,  # optional
            highlight_color="#d4fdd4"  # greenish
        )
        
    
    def update_model_combo(self):
        selected_dataset = self.dataset_combo.currentText()
        models_dir = os.path.join(os.path.dirname(__file__), f"../../configs/models/{selected_dataset}")
        if os.path.isdir(models_dir):
            model_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(models_dir)
                if f.endswith('.json')
            ]
            self.model_combo.clear()
            self.model_combo.addItems([''] + sorted(model_files))
        self.model_combo.currentTextChanged.connect(self._load_model)


        
    
