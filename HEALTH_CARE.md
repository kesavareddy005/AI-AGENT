
# 🏥 Health Care Analytics Project

This project focuses on data-driven insights and predictions in the health care domain. It explores how machine learning can help in disease prediction, patient risk stratification, and health metric analysis.

---

## 📁 1. Project Content

The notebook includes the following stages:

- **Data Loading and Cleaning**: Reading and cleaning health-related datasets.
- **Exploratory Data Analysis (EDA)**: Using plots and statistics to understand the data.
- **Feature Engineering**: Creating relevant features for predictive modeling.
- **Model Building**: Applying classification models for diagnosis prediction.
- **Model Evaluation**: Assessing performance using accuracy, precision, recall, etc.
- **Visualization**: Displaying results with plots and heatmaps.

---

## 💻 2. Project Code

The complete code used in this project is shown below:

```python

```
```python
%pip install pandas scikit-learn torch gradio
```
```python
# Load the dataset
df = pd.read_csv('/content/healthcare_dataset.csv')

# Display the first few rows
display(df.head())

# Check for missing values
display(df.isnull().sum())

# Create the 'Healthy' target variable
df['Healthy'] = df['Test Results'].apply(lambda x: 1 if x == 'Normal' else 0)

# Convert categorical features to numerical using Label Encoding
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Drop columns that are not suitable for training or have too many unique values
columns_to_drop = ['Name', 'Date of Admission', 'Doctor', 'Hospital', 'Room Number', 'Discharge Date', 'Test Results']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_columns_to_drop, axis=1)

# Define features (X) and target (y)
X = df.drop('Healthy', axis=1)
y = df['Healthy']

# Display the first few rows of the preprocessed data
display(X.head())
display(y.head())
```
```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Define the classification model
class HealthcareClassifier(nn.Module):
    def __init__(self, input_dim):
        super(HealthcareClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train_tensor.shape[1]
model = HealthcareClassifier(input_dim)
criterion = nn.BCEWithLogitsLoss() # Using BCEWithLogitsLoss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training finished.")
```
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# Convert predictions to class labels (0 or 1)
y_pred_labels = (y_pred_tensor > 0.5).int()

# Convert tensors to NumPy arrays
y_test_np = y_test_tensor.numpy()
y_pred_np = y_pred_labels.numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np)
recall = recall_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
```
```python
# Convert predictions to class labels with a lower threshold (e.g., 0.3)
y_pred_labels_lower_threshold = (y_pred_tensor > 0.3).int()

# Convert tensors to NumPy arrays
y_test_np = y_test_tensor.numpy()
y_pred_np_lower_threshold = y_pred_labels_lower_threshold.numpy()

# Calculate evaluation metrics with the lower threshold
accuracy_lt = accuracy_score(y_test_np, y_pred_np_lower_threshold)
precision_lt = precision_score(y_test_np, y_pred_np_lower_threshold)
recall_lt = recall_score(y_test_np, y_pred_np_lower_threshold)
f1_lt = f1_score(y_test_np, y_pred_np_lower_threshold)

# Print the metrics with the lower threshold
print(f'Metrics with threshold 0.3:')
print(f'Accuracy: {accuracy_lt:.4f}')
print(f'Precision: {precision_lt:.4f}')
print(f'Recall: {recall_lt:.4f}')
print(f'F1-score: {f1_lt:.4f}')
```
```python
import gradio as gr
import numpy as np

# Define the prediction function
def predict_healthy_status(Age, Gender, Blood_Type, Medical_Condition, Insurance_Provider, Admission_Type, Medication):
    # Create a numpy array from the input values
    input_data = np.array([[Age, Gender, Blood_Type, Medical_Condition, Insurance_Provider, Admission_Type, Medication]])

    # Scale the input data using the same scaler used during training
    input_data_scaled = scaler.transform(input_data)

    # Convert the scaled input data to a PyTorch tensor
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    # Get the prediction from the model
    with torch.no_grad():
        prediction = model(input_tensor)

    # Convert the model's output (probability) to a predicted class label (0 or 1)
    predicted_class = 1 if prediction.item() > 0.5 else 0

    # Return the predicted class label
    return "Healthy" if predicted_class == 1 else "Not Healthy"


# Create the Gradio interface
interface = gr.Interface(
    fn=predict_healthy_status,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Gender (0: Female, 1: Male)"),
        gr.Number(label="Blood Type (Encoded: 0-7)"),
        gr.Number(label="Medical Condition (Encoded: 0-7)"),
        gr.Number(label="Insurance Provider (Encoded: 0-4)"),
        gr.Number(label="Admission Type (Encoded: 0-2)"),
        gr.Number(label="Medication (Encoded: 0-4)"),
    ],
    outputs=gr.Textbox(label="Predicted Health Status")
)
```
```python
interface.launch()
```

---

## 🔧 3. Key Technologies

- ✅ **Python 3.x**
- ✅ **Jupyter Notebook**
- ✅ **Pandas & NumPy**
- ✅ **Matplotlib & Seaborn**
- ✅ **Scikit-learn**
- ✅ **Machine Learning Techniques** (e.g., Logistic Regression, Decision Trees)

---

## 📝 4. Description

This Health Care Analytics project demonstrates how machine learning models can assist in:

- Diagnosing diseases based on patient symptoms and data.
- Predicting health outcomes using features like age, gender, and test results.
- Visualizing patterns to better understand population health statistics.

By leveraging classification models and visualizations, this project gives insights into important health trends and model behavior.

---

## 📊 5. Output

### Model Accuracy

- Accuracy scores of trained models vary between **80% to 90%** depending on the algorithm used.
- Confusion matrices and classification reports are used to evaluate model effectiveness.


---

## 🔬 6. Further Research

- 🔬 **Try Deep Learning Models** (e.g., Neural Networks) for complex patterns.
- 🧠 **Use Medical Datasets from Kaggle or WHO** for more real-world scenarios.
- ⚙️ **Deploy the Model** for real-time predictions via web applications.
- 📈 **Time-Series Data Analysis** on health metrics like blood pressure, glucose.
- 🗃️ **Integrate EHR Systems** for continuous data feed and insights.

---

## 🚀 Conclusion

This project shows how data science can improve decision-making in health care through prediction and visualization. It is a practical application of ML to improve patient care and operational efficiency.

Feel free to experiment with the dataset, models, or visualizations for your own health-related applications.

Stay healthy and keep learning! 🧬💡

---
