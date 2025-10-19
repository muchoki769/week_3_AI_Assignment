# Ethical Analysis Report: Bias Identification and Mitigation


# Results
![Portfolio Screenshot](Public/Screenshot%20for%20TensorFlow.PNG)
![Portfolio Screenshot](Public/screenshot%20tensorFlow2.PNG)

[📄 View Portfolio PDF](Public/AI%20Tools%20Assignment.pdf)
## MNIST Handwritten Digits Model Biases

### **Data Representation Biases:**
1. **Cultural Bias in Digit Writing:**
   - Digits may be written differently across cultures (e.g., 1 with/without base, 7 with/without crossbar)
   - Training data primarily from Western contributors creates representation gaps

2. **Demographic Bias:**
   - Original MNIST contributors were likely predominantly from specific geographic regions
   - Age bias: handwriting styles vary significantly by age group

3. **Quality and Style Bias:**
   - Clean, centered digits favor neat handwriting over messy real-world samples
   - Limited variation in writing instruments and surfaces

### **Mitigation Strategies with TensorFlow Fairness Indicators:**

```python
# Example implementation of fairness analysis
import tensorflow as tf
from tensorflow_model_analysis import fairness_indicators

# 1. Data slicing for bias detection
slice_specs = [
    tfma.slicer.SingleSliceSpec(columns=['digit']),
    tfma.slicer.SingleSliceSpec(columns=['writing_style']),
    tfma.slicer.SingleSliceSpec(columns=['region'])
]

# 2. Fairness metrics configuration
fairness_metrics = fairness_indicators.FairnessIndicators(
    thresholds=[0.5, 0.75, 0.9],
    target_prediction_key='prediction',
    labels_key='label'
)

# 3. Bias mitigation techniques
def mitigate_mnist_biases():
    strategies = {
        "data_augmentation": [
            "Rotate digits ±15 degrees",
            "Add noise to simulate different writing tools",
            "Vary digit sizes and positions",
            "Include international digit variations"
        ],
        "model_training": [
            "Use adversarial debiasing",
            "Implement reweighting for underrepresented styles",
            "Apply gradient reversal for sensitive attributes"
        ],
        "evaluation": [
            "Disaggregated metrics by writing style",
            "Fairness constraints during optimization",
            "Regular bias audits"
        ]
    }
    return strategies

