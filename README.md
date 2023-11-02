### Instructions to inference the model

Inference by Web Application
Go to the URL https://receiptcountprediction-a1044ca4a2c7.herokuapp.com. Upload the file of any year but having the same format as 2021 data and hit Predict to get the predictions and visualizations for the next year.

Inference by API locally
Run "python app.py" in your terminal. Create a Postman POST request. In the Body section form-data, add the key as "file" and in the hidden drop down menu change text to file. Now select the CSV file which you wish to upload and hit send to the URL http://127.0.0.1:5000/predict_api.

### Software And Tools Requirements

1. [Github Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com/)
4. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

### Create a new environment
```
conda create -p venv python==3.9 -y
```

### Install the requirements
```
pip install -r requirements.txt
```
