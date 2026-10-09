# 🚧 Road Damage Detection Using YOLO11

An AI-powered Computer Vision project that detects road damage in images using a trained **YOLO11 object detection model**. The project includes an interactive Streamlit web application where users can upload road images, view predictions, and download results.

## 🌐 Live Demo

**Try the application online:**

### [🚀 Open Road Damage Detection App](https://riyyanzaffar-road-damage-detection-using-yolo11.streamlit.app/)

Upload a road image and explore the model's road damage detection results directly in your browser.

## 📌 Project Overview

Road damage, including cracks and potholes, can create safety risks and affect vehicle performance. This project uses deep learning and computer vision to identify road damage in images.

The trained YOLO11 model predicts damage locations using bounding boxes and displays the results through a user-friendly Streamlit interface.

## ✨ Features

* 🖼️ Upload road images for detection.
* 🤖 AI-powered road damage detection using YOLO11.
* 📦 Display bounding boxes around detected damage.
* 📊 View prediction results and damage class summaries.
* 📥 Download annotated prediction images.
* 📄 Download PDF reports, if available in the application.
* 🌐 Interactive Streamlit web interface.
* ☁️ Online deployment using Streamlit Community Cloud.

## 🛠️ Technologies Used

* Python
* YOLO11
* Ultralytics
* PyTorch
* Computer Vision
* Deep Learning
* Streamlit
* Pillow

## 📈 Model Performance

The trained model was evaluated on the test dataset.

| Evaluation Metric | Score |
| ----------------- | ----: |
| Precision         | 62.5% |
| Recall            | 56.9% |
| mAP@50            | 58.8% |
| mAP@50-95         | 38.8% |

*Note: Results on new images may vary depending on lighting, image quality, camera angle, and road conditions.*

## 📂 Project Structure

```text
road-damage-detection-using-yolo11/
│
├── app.py              # Streamlit web application
├── best (3).pt         # Trained YOLO11 model weights
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```

## ⚙️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/road-damage-detection-using-yolo11.git
```

### 2. Open the Project Folder

```bash
cd road-damage-detection-using-yolo11
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🚀 How to Use

1. Open the [Live App](https://riyyanzaffar-road-damage-detection-using-yolo11.streamlit.app/).
2. Upload a road image.
3. Run the detection.
4. View the predicted road damage and bounding boxes.
5. Review the detection summary.
6. Download the annotated image or PDF report if those options are available.

## 🧠 Model Information

* **Model:** YOLO11
* **Task:** Object Detection
* **Input:** Road images
* **Output:** Predicted classes, confidence scores, and bounding boxes
* **Model Weights:** `best (3).pt`

The application uses a trained model to detect road damage in uploaded images. Predictions depend on the quality and variety of the training data.

## 🎯 Project Objective

The main objective is to apply computer vision and deep learning to road damage detection and provide an accessible web application for testing the trained model on road images.

## 👨‍💻 Author

**Riyyan Zaffar**

* **Live Application:** [Road Damage Detection](https://riyyanzaffar-road-damage-detection-using-yolo11.streamlit.app/)
* **GitHub Repository:** [Road Damage Detection using YOLO11](https://github.com/YOUR-USERNAME/road-damage-detection-using-yolo11)

---

⭐ If you find this project interesting, consider giving the repository a star!
