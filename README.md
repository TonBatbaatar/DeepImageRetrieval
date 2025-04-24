# 🔍 Image Similarity Search with Flask & Deep Learning

A web application that retrieves visually similar images using **deep metric learning** (contrastive, triplet, and proxyNCA losses). Upload an image or provide a URL, and the system returns the closest matches from a dataset.

## 🛠️ Features
- **Multiple Models**: Choose between 7+ metric learning approaches.
- **Web Interface**: Simple UI for uploads and results visualization.
- **Dataset Support**: Pre-configured for birds/cars (extensible to custom datasets).
- **Evaluation Mode**: Compare model performances side-by-side.

## 📦 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/TonBatbaatar/DeepImageRetrieval.git
   cd DeepImageRetrieval

2. **Set up a virtual environment**:
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows

3. **Install dependencies:**:
   pip install -r requirements.txt

## 🚀 Usage
1. Run the Flask app
   python app.py
2. Access the web interface at http://localhost:5000:
   - Upload an image or paste a URL.
   - View similarity results ranked by model confidence.
