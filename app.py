import os
import numpy as np
from PIL import Image
from flask import Flask,flash, render_template, request, redirect, session, send_from_directory
from flask import redirect, send_file, url_for
import torch
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, configure_uploads, IMAGES
from generator import Generator

model = Generator(3,64)
model_path = 'saved_model/gen_epoch20.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 4096 * 4096

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])

def model_output(img_arr):
    transform = A.Compose([
        A.Resize(256,256),
        A.Normalize(
             [0.5 for _ in range(3)], [0.5 for _ in range(3)]
        ),
        ToTensorV2()
    ])
    img = transform(image=img_arr)['image']
    img = img.unsqueeze(0)
    pred = model(img)
    
    pred = pred.squeeze(0)
    pred = pred.permute(1,2,0)
    pred = pred.detach().numpy()
    pred = (pred * 255).astype(np.uint8)
    return pred



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/temp.jpeg'),  code=301)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_img():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_arr = np.array(img)
        print(img_arr.shape)
        out_img = model_output(img_arr)
        img = Image.fromarray(img_arr)
        img.save("static/uploads/temp.png")
        out_img = Image.fromarray(out_img)
        out_img.save("static/uploads/result.png")
        return render_template('index.html', filename='temp.png')
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)