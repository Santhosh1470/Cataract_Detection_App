# Cataract Detection Web App

## 📌 Project Overview
This is a **Cataract Detection System** that uses **Deep Learning** to classify eye images as **Cataract** or **No Cataract**. The model is trained using **TensorFlow/Keras** and deployed as a **Flask Web Application**.

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow/Keras** (for Model Training)
- **Flask** (for Web App Backend)
- **HTML, CSS, JavaScript** (for Frontend UI)
- **Matplotlib** (for Visualizing Training Accuracy)
- **Render** (for Deployment)

---

## 📂 Project Structure
```
Cataract_Detection_App/
│── static/
│   ├── uploads/            # Stores uploaded images
│   ├── accuracy_graph.png  # Model accuracy graph
│── templates/
│   ├── index.html          # Home page
│   ├── result.html         # Prediction result page
│── model/
│   ├── cataract_model.h5   # Trained Model
│── app.py                  # Flask Backend
│── train_model.py          # Model Training Script
│── requirements.txt        # Dependencies
│── README.md               # Project Documentation
```

---

## 🔧 1️⃣ Model Training (train_model.py)
1. **Import Dependencies**:
   ```python
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   import matplotlib.pyplot as plt
   ```

2. **Load Dataset** (Ensure dataset is structured properly):
   ```python
   train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
   train_data = train_datagen.flow_from_directory('dataset/', target_size=(224,224), batch_size=32, class_mode='binary', subset='training')
   val_data = train_datagen.flow_from_directory('dataset/', target_size=(224,224), batch_size=32, class_mode='binary', subset='validation')
   ```

3. **Define CNN Model**:
   ```python
   model = Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
       MaxPooling2D(2,2),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(1, activation='sigmoid')
   ])
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   ```

4. **Train the Model**:
   ```python
   history = model.fit(train_data, validation_data=val_data, epochs=10)
   ```

5. **Save the Model**:
   ```python
   model.save('model/vgg19_model_final.h5')
   ```

6. **Plot Accuracy Graph**:
   ```python
   plt.plot(history.history['accuracy'], label='Train Accuracy')
   plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.title('Model Accuracy')
   plt.savefig('static/accuracy_graph.png')
   ```

---

## 🌐 2️⃣ Flask Web App (app.py)
### **Load the Trained Model**
```python
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/vgg19_model_final.h5")

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
```

### **Home Page Route**
```python
@app.route('/')
def home():
    return render_template('index.html')
```

### **Prediction Route**
```python
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    file_path = "static/uploads/" + file.filename
    file.save(file_path)

    # Preprocess & Predict
    image = preprocess_image(file_path)
    prediction = model.predict(image)
    result = "Cataract Detected" if prediction[0][0] > 0.5 else "No Cataract"

    return render_template('result.html', result=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🎨 3️⃣ Frontend (HTML Pages)
### **index.html** (Upload Image Page)
```html
<h1>Cataract Detection System</h1>
<p>Upload an eye image to detect cataract.</p>
<form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <button type="submit">Predict</button>
</form>
```

### **result.html** (Show Prediction & Accuracy Graph)
```html
<h2>Prediction Result: {{ result }}</h2>
<img src="{{ image_path }}" alt="Uploaded Image" width="300">
<h3>Model Training Accuracy</h3>
<img src="{{ url_for('static', filename='accuracy_graph.png') }}" alt="Accuracy Graph" width="500">
```

---

## 🚀 4️⃣ Deployment on Render
1. **Install Render CLI & Login**:
   ```bash
   pip install render
   render login
   ```
2. **Create `requirements.txt`**:
   ```bash
   flask
   tensorflow
   pillow
   numpy
   matplotlib
   ```
3. **Initialize Git Repository & Push Code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```
4. **Deploy on Render**:
   - Go to [Render](https://render.com/)
   - Select **Flask App**
   - Connect your GitHub repo
   - Set `app.py` as the Start Command
   - Click **Deploy**

---

## 🎯 Final Outcome
✅ **Trained Model**
✅ **Flask App to Upload & Predict Cataract**
✅ **Accuracy Graph Displayed**
✅ **Deployed Online on Render**

🚀 **Now your Cataract Detection System is LIVE!** 🎉

