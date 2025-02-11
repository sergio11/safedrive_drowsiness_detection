# 🚗 **Deep Learning for Safer Roads** 🚗  Exploring CNN-Based and YOLOv8 Driver Drowsiness Detection 💤


This notebook delves into the exciting world of **deep learning** and its potential to **save lives** on the road. 🚦 We tackle the critical issue of **driver drowsiness detection**, using cutting-edge models to help **prevent fatigue-related accidents**. We explore two powerful approaches: **CNN-based models with Transfer Learning** and **YOLOv8 integrated with Roboflow**. 📊 Let’s see which one emerges as the most effective for **real-time driver monitoring**! ⏱️

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

## 🚙 **Why is This Important?** 🚙  
Driver fatigue is a major cause of road accidents worldwide 😴. Detecting **drowsy driving** in real-time is crucial for **preventing accidents** and ensuring road safety. This project leverages the **Driver Drowsiness Dataset (DDD)**, a collection of images of drivers in **Drowsy** and **Non-Drowsy** states, to build two different models aimed at **real-time detection** of drowsiness. 🌍

## 💻 **Approaches Explored** 💻  
### **1. CNN-based Models**: The Power of Convolutional Neural Networks 🔍  
CNNs are at the heart of modern image recognition and excel at learning **spatial hierarchies** of image data. For this task, we use **MobileNetV2**, a **lightweight** and **efficient** CNN architecture that’s ideal for **real-time applications** in vehicles. 🚗

- **Why CNNs?**  
   ✅ Efficient at detecting drowsiness through **facial state recognition**.  
   ✅ Works great on **smaller datasets**.  
   ✅ Uses **pre-trained models** to speed up the process. 🏎️

### **2. YOLOv8 with Roboflow**: Real-Time Object Detection at Its Best 🕵️‍♂️  
YOLOv8 is a state-of-the-art **object detection model** designed for **speed** and **accuracy**. Paired with **Roboflow**, YOLOv8 becomes even more powerful by streamlining the **dataset preparation** and **deployment** process. 🚀

- **Why YOLOv8?**  
   ✅ Super fast, making it perfect for **real-time applications**. ⏱️  
   ✅ Capable of **localizing and detecting** multiple objects in a single frame.  
   ✅ **Roboflow integration** simplifies dataset annotation and augmentation. 📝

## 🧠 **How Does the Model Work?** 🧠  
The **Driver Drowsiness Dataset (DDD)** is the foundation for training both models. Here’s what you need to know about the data:

- **Image Format:** RGB images with facial features of drivers.  
- **Classes:** `Drowsy` vs. `Non-Drowsy`.  
- **Resolution:** 227 x 227 pixels, optimized for deep learning tasks.  
- **Size:** 41,790+ labeled images.  
- **File Size:** ~2.32 GB of drowsiness-related data!

## 🔥 **Project Goals** 🔥

1. **Evaluate the two models**—CNN-based transfer learning and YOLOv8 with Roboflow—for detecting **drowsy drivers**.
2. **Compare their performance** using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
3. **Optimize the models** for real-time deployment in vehicles.

## 🛠️ **Methodology** 🛠️  
We break the project into several stages:

### 1. **Data Preprocessing** 🧹  
- **Normalize** pixel values.  
- **Augment** data to improve generalization.  
- **Split** the dataset into **training**, **validation**, and **testing** sets.

### 2. **Model Development** 💡  
- **Train CNN** models with **MobileNetV2** for transfer learning.  
- **Train YOLOv8** using **Roboflow** for real-time object detection.

### 3. **Model Evaluation** 📊  
- Compare performance using key metrics.  
- **Assess real-time feasibility** for vehicle deployment.

### 4. **Real-Time Testing** ⏱️  
- Simulate **real-world conditions** and evaluate **model responsiveness**.

## 🛣️ **Why Does This Matter?** 🛣️  
This project aims to **reduce road accidents** by **detecting driver fatigue** using AI. Through **deep learning**, we can proactively detect when a driver is drowsy, preventing potential accidents before they happen. **Safer roads** lead to **lives saved**. 🌟

## 🤖 **Comparing the Two Approaches**: CNN + Transfer Learning vs YOLOv8 with Roboflow ⚖️

### **1. CNN + Transfer Learning**: Fast and Effective 🌟
This approach uses **pre-trained CNNs** (e.g., MobileNetV2) to **classify driver states** as drowsy or non-drowsy based on facial images.

- **Pros:**  
  - **Quick setup**.  
  - **Works well with smaller datasets**.  
  - **Pre-trained models** speed up training.

- **Cons:**  
  - **No object localization**.  
  - **Limited real-time suitability** due to preprocessing.  
  - May struggle with **high-resolution data**.

### **2. YOLOv8 + Roboflow**: Real-Time Detection 🚀  
YOLOv8 excels at **detecting and localizing objects** (e.g., faces, eyes) in real-time, especially when paired with **Roboflow**.

- **Pros:**  
  - **Perfect for real-time detection**.  
  - **Detects and localizes** multiple objects at once.  
  - Simplified data preparation via **Roboflow**.

- **Cons:**  
  - **Resource-heavy**, requiring powerful hardware.  
  - **Higher complexity** in setup and fine-tuning.  
  - **Time-intensive annotations** for accurate detection.

### **Side-by-Side Comparison**

| **Aspect** | **CNN + Transfer Learning** | **YOLOv8 + Roboflow** |
|------------|-----------------------------|-----------------------|
| **Primary Goal** | Classification (Drowsy vs. Non-Drowsy) | Real-Time Detection and Localization |
| **Real-Time Suitability** | Limited (Preprocessing adds latency) | Optimized for real-time applications |
| **Ease of Use** | Beginner-friendly, quick setup | More complex, streamlined with Roboflow |
| **Dataset Needs** | Small to moderate datasets | Detailed annotations, larger datasets |
| **Performance Focus** | Classification Accuracy | Speed + Detection Accuracy |
| **Hardware Requirements** | Moderate (CPUs or mid-tier GPUs) | High (Powerful GPUs recommended) |

## 🤔 **Which Approach Should You Choose?** 🤔

- **Choose CNN + Transfer Learning** if:  
  - You need a **quick setup** for **image classification**.  
  - Your **dataset is small** or lacks detailed annotations.  
  - You don't need real-time detection and have **limited hardware**.

- **Choose YOLOv8 + Roboflow** if:  
  - **Real-time detection and localization** are essential for your project.  
  - You have access to **powerful hardware** and a **well-annotated dataset**.  
  - You need a **seamless, end-to-end detection system**.

## 🚀 **Let’s Dive In!** 🚀  
Both models are powerful tools for **driver drowsiness detection**, and each offers unique benefits. The choice of approach depends on your specific goals, dataset, and available resources.



https://blog.roboflow.com/how-to-train-a-yolov8-classification-model/
https://www.kaggle.com/code/mernaabdallah/driver-fatigue-monitoring-system-with-yolo-v8#Model-2
