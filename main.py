from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/eda")
def eda():
    return render_template("eda.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/comparison")
def comparison():
    return render_template("comparison.html")


if __name__ == "__main__":
    app.run(debug=True)