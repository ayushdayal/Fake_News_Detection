# definitions
from flask import Flask, render_template, request
from models import predict as pred

app = Flask('__name__')


@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')


def getOutput(headline, body):
    return pred(headline, body)


@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model()  # you can use pickle to load the trained model
        headline = request.form['headline']
        body = request.form['body']
        output = getOutput(headline, body)
        output1 = output[0][0]
        output2 = output[0][1]
        output3 = output[0][2]
        output4 = output[0][3]

        return render_template('resultsform.html', headline=headline, body=body, output1=output1, output2=output2,
                               output3=output3, output4=output4)


app.run("localhost", "9999", debug=True)
