from flask import Flask, render_template, request, url_for

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("C:/Users/Swetha/PycharmProjects/pythonProject2/Templates/front pg.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('photo', '')
        best, others, img_name = ["image"]
        return render_template("C:/Users/Swetha/PycharmProjects/pythonProject2/Templates/predict.html", best=best, others=others, img_name=img_name)

    @app.route('/final', methods=['POST'])
    def final():
        if request.method == 'POST':
            image = request.files.get('photo', '')
            best, others, img_name = ["image"]
            return render_template("C:/Users/Swetha/PycharmProjects/pythonProject2/Templates/final.html", best=best, others=others, img_name=img_name)


if __name__ == "__main__":
    app.run()
