from flask import Flask, render_template, request
from models import load_model, preprocess_image, predict, decode_results
from utils import save_file

app = Flask(__name__)
model = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = save_file(file)
            try:
                pre_image = preprocess_image(filename)
                prediction = predict(model, pre_image)
                results = decode_results(prediction)
                return render_template("results.html", results=results)
            except Exception as e:
                # Handle error
                return render_template("error.html", error_message=str(e)), 500  # Return a 500 error code

    return "Error"


if __name__ == "__main__":
    app.run(debug=True)
