# encoding:utf-8
# !/usr/bin/env python
import json

from gevent import pywsgi
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory
import os
import random
from flask_cors import CORS
from model.predict_service import service
from unique_id import create_uuid

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 跨域限制
UPLOAD_FOLDER = "files"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'}
ip = 'http://120.24.230.237'
port = ':5000'
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
# 如果路径不存在则创建一个
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

my_service = service(model_name="resnet34",
                     weight_path=os.path.join(basedir,"model_weight/restnet_resnet34_0.8741_.pth"),
                     label2name_path=os.path.join(basedir,"model_weight/label2name_resnet34.pkl")
                     )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 小测试
@app.route('/test')
def index():
    return '<h1>Hello World!<h1>'

@app.route('/upload')
def upload_test():
    return render_template('up.html')


# 上传文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['file']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        ext = fname.rsplit('.', 1)[1]
        new_filename = create_uuid() + '.' + ext
        f.save(os.path.join(file_dir, new_filename))
        path_dl = ip + port + "/download/" + new_filename
        path_show = ip + port + "/show/" + new_filename
        return jsonify({"success": 0, "path_dl": path_dl, "path_show": path_show})
    else:
        return jsonify({"error": 1001, "msg": "error"})

@app.route('/random', methods=['GET'], strict_slashes=False)
def get_random_photos():
    uploaded_files = os.listdir(file_dir)
    selected_file = random.choice(uploaded_files)
    path_show = ip + port + "/show/" + selected_file
    return jsonify({"success": 0, "path_show": path_show})

# http://127.0.0.1:5000/download/2021120715463268.jpg
# http://127.0.0.1:5000/download/2021120715340447.jpg

@app.route('/getres', methods=['POST'])
def get_res():
    file_name_b = request.data
    print(file_name_b)
    my_json = file_name_b.decode('utf8').replace("'", '"')
    url = json.loads(my_json)['url']
    file_name = url.split("/")[-1]
    file_path = os.path.join(file_dir, file_name)
    print(file_path)
    res = my_service.predict(file_path)
    # 把结果整理成符合格式的json对象
    Res = [{"p": str(round(list(r.values())[0] * 100, 2)) + '%', "cata": list(r.keys())[0]} for r in res]
    json_str = json.dumps(Res)
    print(json_str)
    json2python = json.loads(json_str)
    return jsonify({"success": 0,
                    "res": json2python
                    })


@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
        if os.path.isfile(os.path.join(file_dir, filename)):
            return send_from_directory(file_dir, filename, as_attachment=True)
        pass


# http://127.0.0.1:5000/show/2021120715340447.jpg

# show photo  会直接显示
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host='0.0.0.0',port=5000)
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
