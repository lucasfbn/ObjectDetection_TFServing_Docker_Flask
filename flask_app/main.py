import json

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from config import ORIGINALS_DIR, PROCESSED_DIR, VALID_FILE_EXTENSIONS
from src import pre_process, post_process

app = Flask(__name__)


def _is_file_input():
    return request.method == "POST" and request.files['file']


def _is_valid_file(filename):
    return any(filename.endswith(ext) for ext in VALID_FILE_EXTENSIONS)


def _handle_file_input():
    file = request.files['file']

    src_fn = file.filename

    src_path = f"./{ORIGINALS_DIR}/{src_fn}"
    out_path = f"./{PROCESSED_DIR}/{src_fn}"

    file.save(src_path)

    img_np, response = pre_process.pipeline(src_path)
    objects = post_process.pipeline(out_path, img_np, response)

    return src_fn, objects


@app.route('/', methods=["GET", "POST"])
def index():
    """
    Entry point to the webapp. Handles the POST request from the form.
    If an invalid file is selected and submitted a warning is shown. Otherwise, the results page is shown.
    """

    if _is_file_input():

        if not _is_valid_file(request.files['file'].filename):
            return render_template('index.html', show_warning=True)

        filename, objects = _handle_file_input()
        return redirect(url_for('results', filename=filename, objects=json.dumps(objects)))

    return render_template('index.html', show_warning=False)


@app.route('/results', methods=["GET", "POST"])
def results():
    """
    Renders the results page. If a new file is submitted, a new resuls page is shown.
    If the submitters file is not valid, a warning is shown.
    """

    filename = request.args.get('filename')
    objects = json.loads(request.args.get('objects'))
    n_objects = len(objects)

    if _is_file_input():

        if not _is_valid_file(request.files['file'].filename):
            return render_template('results.html',
                                   show_warning=True,
                                   filename=filename,
                                   objects=objects,
                                   n_objects=n_objects)

        filename, objects = _handle_file_input()
        return redirect(url_for('results', filename=filename, objects=json.dumps(objects)))

    return render_template('results.html', show_warning=False, filename=filename, objects=objects, n_objects=n_objects)


@app.route(f"/{ORIGINALS_DIR}/<path:fn>")
def originals(fn):
    # Register the directory
    return send_from_directory(f"./{ORIGINALS_DIR}", fn)


@app.route(f"/{PROCESSED_DIR}/<path:fn>")
def processed(fn):
    # Register the directory
    return send_from_directory(f"./{PROCESSED_DIR}", fn)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
