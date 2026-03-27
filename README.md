# 🍏 Nutrition Analysis using ML (NutriScore Predictor)

## 📖 Overview
The **NutriScore Predictor** is a Machine Learning-powered REST API designed to evaluate the nutritional value of food products. By analyzing macronutrients—such as energy (calories), sugars, saturated fats, proteins, and fiber—the model predicts the product's official NutriScore (ranging from A to E). 

This project bridges the gap between raw nutritional data and consumer-friendly health metrics, allowing users or client applications to instantly gauge the healthiness of a product. It features a built-in integration with the OpenFoodFacts API to fetch real-time product data via barcode, passing the nutrients through a trained Random Forest classifier for instant scoring.

## ✨ Key Features
* **ML-Powered Predictions:** Utilizes a custom-trained Random Forest model to accurately classify food products based on complex nutritional profiles.
* **Barcode Integration:** Includes endpoints to fetch live product data directly from the OpenFoodFacts database using standard barcodes.
* **RESTful API:** Built with Flask, providing clean and accessible JSON endpoints for easy frontend or mobile app integration.
* **Automated Cloud Deployment:** Configured for seamless deployment, utilizing a custom build script to fetch heavy ML models dynamically from cloud storage, bypassing Git repository size limits.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
* **Backend Framework:** Flask, Gunicorn
* **Data Source:** OpenFoodFacts (12GB+ Global Product Dataset)
* **Deployment & CI/CD:** Render, Bash Scripting, Git

---
> **Note on Data and Models:** > To maintain repository performance and adhere to GitHub's file size limits, the original 12GB dataset and the compiled `.pkl` model files are not hosted directly in this repository. The deployment environment is configured to automatically pull the necessary pre-trained models from a secure cloud storage link during the build phase.
## 🚀 How to Run Locally

Because this project relies on large Machine Learning models that exceed standard version control limits, the pre-trained `.pkl` files are not included in this repository. 

Follow these steps to set up the environment and fetch the necessary model files automatically.

### Prerequisites
* **Python 3.8+** installed on your machine.
* **Git** installed.

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone [https://github.com/VishwasShirurmath/Nutrition-Analysis-using-ML.git](https://github.com/VishwasShirurmath/Nutrition-Analysis-using-ML.git)
cd Nutrition-Analysis-using-ML 

On Linux/macOS (Bash/Zsh):
python3 -m venv .venv
source .venv/bin/activate

On Linux/macOS (Fish Shell):
python3 -m venv .venv
source .venv/bin/activate.fish

On Windows:
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Fetch the Pre-Trained Models
Run the included bash script to automatically download the Random Forest model files from secure cloud storage into the correct directories.
bash build.sh

Start the Server
Once the models are successfully downloaded, start the local development server:
gunicorn app:app
```
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e82d90d8-5fc0-468e-a42b-ebbf6228c4c9" />



