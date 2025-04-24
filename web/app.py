# import modules

# main flask framwork
# Flask: The main web framework.
    # render_template: Renders HTML templates.
    # request: Handles HTTP requests (GET/POST data, file uploads).
    # session: Stores user-specific data (like selected model/dataset).
    # url_for: Generates URLs for Flask routes.
    # redirect: Redirects to another route.
    # send_from_directory: Sends files (e.g., uploaded images).
from flask import Flask, render_template, request, session, url_for, redirect, send_from_directory

# secure_filename: Sanitizes filenames for safe storage.
from werkzeug.utils import secure_filename

# transforms: PyTorch image preprocessing (resize, normalize, etc.).
from torchvision import transforms

# os/sys: File/path operations.
# urllib: Downloads images from URLs.
# PIL (Pillow): Image processing.
# matplotlib.pyplot: Plotting (unused in this code).
import os
import sys
import urllib
import PIL
import matplotlib.pyplot as plt

# Custom module for image retrieval (retrieve() and get_model_result()).
from retrieve import retrieve
from retrieve import get_model_result

# import metrics
import metrics.contrastive as contrastive
import metrics.triplet as triplet
import metrics.softtriple as softtriple
import metrics.multisimilarity as multisimilarity
import metrics.proxyNCA as proxyNCA


# implementation
# Defines where uploaded files are stored (/static/upload).
UPLOAD_PATH = sys.path[0] + '/static/upload'

# Allowed image file extensions.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Max upload file size (16MB).
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Number of supported models (e.g., contrastive, triplet, etc.).
MODEL_NUM = 7

# Initializes the Flask app.
app = Flask(__name__)

# Initializes the Flask app.
html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Picture upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

# Configures upload folder and max file size.
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Default image paths for the UI.
uploaded_path = 'img/default.png'
result_path = ['img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png',
               'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', ] # 10 default images
ret_path_eval = [['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png']] # 7 models x 3 images

# Secret key for session encryption.
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# route for root
# Redirects root to the /search page.
@app.route("/", methods=['GET', 'POST'])
def origin():
    return redirect(url_for("search"))

# route for choose
# Stores user-selected dataset/model in the session.
# Redirects back to /search.
@app.route('/choose', methods=['GET'])
def choose_model():
    if request.values.get("choosedataset") is not None:
        session["dataset"] = request.values.get("choosedataset")
    if request.values.get("choosemodel") is not None:
        session['model'] = request.values.get("choosemodel")
    if request.values.get("choosecarmodel") is not None:
        session['carmodel'] = request.values.get("choosecarmodel")
    return redirect(url_for("search"))

# route for search
@app.route('/search', methods=['GET', 'POST'])
def search():

    # Creates upload folder if it doesn’t exist.
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    # Loads model/dataset from session or uses defaults.
    model = '0' # Default model
    dataset = 'bird' # Default dataset
    if 'model' in session:
        model = session["model"]
    if 'dataset' in session:
        dataset = session['dataset']
        if dataset == 'car':
            model = '7'
            if 'carmodel' in session:
                model = session['carmodel']
    
    error = None
    filename = None
    img_path = None

    # Handles file uploads or URL downloads:
    if request.method == 'GET':
        if request.values.get("search") is not None:
            url = request.values.get("search")
            filename = secure_filename(url.split('/')[-1])
            if allowed_file(filename):
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    urllib.request.urlretrieve(url, img_path) # Downloads an image from a URL.
                except urllib.error.HTTPError:
                    error = "Image can not be downloaded successfully!"
            else:
                error = "Suffix of url must be jpg or png!"
    
    # Handles file uploads.
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
        else:
            error = "Only jpg or png images are accepted!"

    # Processes the image and retrieves similar images using get_model_result().
    if error is None and img_path is not None:
        file_url = url_for('uploaded_file', filename=filename)
        try:
            image = PIL.Image.open(img_path)
            results = get_model_result(image, 10, model, dataset)
            print(results)
            return render_template('search.html', uploaded_path=file_url, result_path=results, model = model, dataset = dataset)
        except PIL.UnidentifiedImageError:
            error = "Can not open the file uploaded!"
    
    # Renders the search page with results or errors.
    return render_template('search.html', uploaded_path=uploaded_path, result_path=result_path, error = error, model = model, dataset = dataset)

# route for evaluation
@app.route('/evaluation',methods=['GET', 'POST'])
# Compares all 7 models by retrieving top 3 similar images for each.
def evaluation():
    if request.method == 'GET':
        if request.values.get("askforevainput") is not None:
            file_url = request.values.get("askforevainput")
            img_path = sys.path[0]+ "/static" + file_url
            results = []
            image = PIL.Image.open(img_path)
            for i in range(0,MODEL_NUM):
                ret = get_model_result(image, 3, str(i))
                results.append(ret)
            print(results)
            return render_template('evaluation.html', uploaded_path=file_url, result_path=results)
    return render_template('evaluation.html', uploaded_path=uploaded_path, result_path = ret_path_eval)


# route for save Evaluation result
@app.route('/saveEval', methods=['GET', 'POST'])
# Saves user evaluations to a CSV file.
def saveEval():
    if request.method == 'GET':
        eval_ret = ""
        for i in range(0,MODEL_NUM):
            if request.values.get(str(i)) is not None:
                eval_ret += request.values.get(str(i))
                if i == (MODEL_NUM-1):
                    eval_ret += "\n"
                else:
                    eval_ret += ","
        print(eval_ret)

        eval_path = sys.path[0] + "/static"
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(eval_path + '/eval_result.csv', 'a+') as f:

            f.write(eval_ret)

    return render_template('submitted.html')

# route for download
@app.route('/download', methods=['GET', 'POST'])
def download():
    return render_template('download.html')

# route for algorithm
@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    return render_template('algorithm.html')

# helper function: Checks if a filename has an allowed extension.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# route for upload
# Simple file upload interface.
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            return html + '<br><img src=' + file_url + '>'
    return render_template("upload.html")

# Serves uploaded images.
@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# Runs the app in debug mode.
if __name__ == '__main__':
    app.run(debug=True)
