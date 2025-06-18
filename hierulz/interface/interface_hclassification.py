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

from hierulz.hierarchy import load_hierarchy
from hierulz.datasets import get_dataset_config


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
        layout.addLayout(self._build_blur_controls())
        layout.addLayout(self._build_model_controls())
        layout.addLayout(self._build_metric_controls())
        layout.addLayout(self._build_decoding_controls())
        layout.addWidget(self._build_prediction_output())
        self.setLayout(layout)

    
    def _build_image_hierarchy_layout(self):
        layout = QHBoxLayout()

        layout.addLayout(self._build_image_side())
        layout.addWidget(self._build_hierarchy_widget("hierarchy_layout", "hierarchy_widget"))
        layout.addWidget(self._build_hierarchy_widget("prediction_layout", "prediction_widget"))

        return layout

    def _build_image_side(self):
        layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(224, 224)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.class_label = QLabel("Class: ")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("font-size: 22pt; font-weight: bold; margin-top: 10px;")

        layout.addWidget(self.image_label)
        layout.addWidget(self.class_label)

        return layout

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
        return self.btn_load

    def _build_dataset_controls(self):
        layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["", "tieredimagenet", "inat19"])
        self.dataset_combo.currentTextChanged.connect(self._select_dataset)  # Add a handler if you want

        layout.addWidget(QLabel("Dataset"))
        layout.addWidget(self.dataset_combo)

        return layout


    def _build_blur_controls(self):
        layout = QHBoxLayout()

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

        return layout
    
    def _build_model_controls(self):
        layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'alexnet', 'convnext_tiny', 'densenet121', 'efficientnet_v2_s',
            'inception_v3', 'resnet18', 'swin_v2_t', 'vgg11', 'vit_b_16'
        ])
        self.model_combo.currentTextChanged.connect(self._load_model)

        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.model_combo)

        return layout
    
    def _build_metric_controls(self):
        layout = QHBoxLayout()

        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "hF_ß", "Mistake Severity", "Wu-Palmer", "Zhao",
            "Accuracy", "Hamming Loss", "Node2Leaf", "Leaf2Leaf"
        ])
        self.metric_combo.currentTextChanged.connect(self._select_metric)

        layout.addWidget(QLabel("Metric"))
        layout.addWidget(self.metric_combo)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 9999.0)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(1.0)

        layout.addWidget(QLabel("ß"))
        layout.addWidget(self.beta_spin)

        return layout
    
    def _build_decoding_controls(self):
        layout = QHBoxLayout()

        self.decode_combo = QComboBox()
        self.decode_combo.addItems([
            'Optimal', 'Argmax leaves', 'Top-down argmax', 'Thresholding 0.5',
            'Plurality', 'Exp. Information', 'Hie-Self (Jain et al., 2023)',
            '(Karthik et al., 2021)'
        ])
        
        layout.addWidget(QLabel("Decoding"))
        layout.addWidget(self.decode_combo)

        self.decode_button = QPushButton("Decode Proba")
        self.decode_button.clicked.connect(self._decode_proba)
        layout.addWidget(self.decode_button)

        return layout
    
    def _build_prediction_output(self):
        self.output_label = QLabel("Prediction: ")
        self.output_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        return self.output_label

    def _update_blur_label(self, value):
        self.blur_value_label.setText(f"{value / 10:.1f}")



    def _select_dataset(self, dataset_name):
        # You can add logic here to load dataset info, reset UI, etc.
        print(f"Dataset selected: {dataset_name}")
        # For example:
        self.config_dataset = get_dataset_config(dataset_name)
        self.hierarchy = load_hierarchy(self.config_dataset['hierarchy_idx'])
        self.btn_load.setVisible(True)


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
        self.display_label(folder_category_path)

    def display_image(self, img):
        img = img.resize((224, 224))
        img_np = np.array(img)
        qimg = QImage(img_np.data, img_np.shape[1], img_np.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)   

    def display_label(self, label):  
        class_to_idx = pkl.load(open(self.config_dataset['class_to_idx'], 'rb'))
        idx_to_name = pkl.load(open(self.config_dataset['idx_to_name'], 'rb'))

        label_idx = class_to_idx[label]
        ancestors = np.where(self.hierarchy.leaf_events[label_idx])[0]
        ancestors_sorted_depths = sorted(ancestors, key=lambda x: self.hierarchy.depths[x])
        ancestors_names = [idx_to_name[idx] for idx in ancestors_sorted_depths]

        # clear previous labels
        while self.hierarchy_layout.count():
            child = self.hierarchy_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for i, label in enumerate(ancestors_names):
            node_label = QLabel(label)
            node_label.setMinimumHeight(30)
            node_label.setMaximumHeight(30)
            node_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            node_label.setStyleSheet("""
                QLabel {
                    border: 1px solid #333;
                    border-radius: 6px;
                    padding: 2px;
                    background-color: #f0f0f0;
                    font-size: 10pt;
                    qproperty-alignment: AlignCenter;
                }
            """)
            self.hierarchy_layout.addWidget(node_label)

            if i < len(ancestors_names) - 1:
                arrow = QLabel("↓")
                arrow.setMinimumHeight(20)
                arrow.setMaximumHeight(20)
                arrow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                arrow.setStyleSheet("font-size: 12pt; color: gray; qproperty-alignment: AlignCenter;")
                self.hierarchy_layout.addWidget(arrow)
        self.last_groundtruth_path = ancestors_sorted_depths


        # Set class label on top bar (optional)
        self.class_label.setText(f"Class: {ancestors_names[-1]}")

    def _apply_blur(self):
        pass

    def _load_model(self):
        pass

    def _select_metric(self):
        pass

    def _decode_proba(self):
        pass
    
