# Cat vs Dog Classifier

This is a Flask-based web application that allows users to upload an image and classify it as either a **cat** or a **dog** using a pre-trained deep learning model.

## 🚀 Features
- Classifies the image as **Cat**, **Dog**
- Displays the classification result along with confidence score

## 🛠️ Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/amngiri/cat-dog-classifier.git
cd cat-dog-classifier
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Pre-trained Model
Make sure to place your **catdogclassifier.h5** model file in the project directory.

### 5. Run the Flask App
```bash
python app.py
```

### 6. Open in Browser
Visit: [http://127.0.0.1:5600](http://127.0.0.1:5600)

## 📂 Project Structure
```
cat-dog-classifier/
│── templates/
│   ├── index.html  # Upload Page
│   ├── result.html  # Result Page
│── app.py  # Main Flask Application
│── catdogclassifier.h5  # Pre-trained Model (Not included in repo)
│── requirements.txt  # Required Python Packages
│── README.md  # Project Documentation
```

## 📌 Dependencies
- Flask
- TensorFlow
- OpenCV
- NumPy

## 🌍 Hosted on Hugging Face
You can also try out the application on Hugging Face Spaces:
[Cat-Dog Classifier on Hugging Face](https://huggingface.co/spaces/GiriAman/cat-dog-classifier)

