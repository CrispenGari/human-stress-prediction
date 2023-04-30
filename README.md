### Human Stress Prediction(HSP)

`HSP` is an API that is served using flask to perform binary classification to determine wether the person is stressed or not based on their text. If the person is stressed or even not stressed the model also go ahead and detect what context is the text about. So this AI API is doing both binary and multi-class classification on textual data.

<p align="center"><img src="cover.jpg" alt="cover" width="100%"/></p>

### Starting the server

To run this server and make predictions locally you need follow the following steps

0. clone this repository by running the following command:

```shell
https://github.com/CrispenGari/human-stress-prediction.git
```

1. Navigate to the folder `human-stress-prediction` by running the following command:

```shell
cd human-stress-prediction
```

2. create a virtual environment and activate it, you can create a virtual environment in may ways for example on windows you can create a virtual environment by running the following command:

```shell
virtualenv venv && .\venv\Scripts\activate
```

3. run the following command to install packages

```shell
pip install -r requirements.txt
```

4. To start the server you need to run the following command

```shell
python server.py
```

If everything works you will be able to see the following logs on the console:

```shell
✅ LOADING PYTORCH S2DC MODEL!

✅ LOADING PYTORCH S2DC MODEL DONE!

 * Serving Flask app 'api.app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3001
Press CTRL+C to quit
```

### HSPModel

This model was created and modeled as illustrated bellow:

```shell
        |-------------> [out1] --- [1]
        |                 |
[embedding]--------->[bi-lstm]
                          |
                          |--------------> [out2] --- [10]
```

This is how the model was implemented in code:

```py

class HSPModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,         output_size1, output_size2 , num_layers
                    , bidirectional, dropout, pad_idx):
            super(HSPModel, self).__init__()
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=pad_idx),
                nn.Dropout(dropout)
            )
            self.lstm = nn.Sequential(
                nn.LSTM(
                embedding_size,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=dropout
                )
            )
            self.out1 = nn.Sequential(
                nn.Linear(hidden_size * 2, out_features=128),
                nn.Dropout(dropout),
                nn.Linear(128, out_features=output_size1),
                nn.Dropout(dropout)
            )
            self.out2 = nn.Sequential(
                nn.Linear(hidden_size * 2, out_features=128),
                nn.Dropout(dropout),
                nn.Linear(128, out_features=output_size2),
                nn.Dropout(dropout)
            )
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        # set batch_first=true since input shape has batch_size first and text_lengths to the device.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)
        packed_output, (h_0, c_0) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = torch.cat((h_0[-2,:,:], h_0[-1,:,:]), dim = 1)
        return self.out1(output), self.out2(output)

```

The model outputs something that looks like this after initialization:

```shell
HSPModel(
  (embedding): Sequential(
    (0): Embedding(6394, 100, padding_idx=1)
    (1): Dropout(p=0.5, inplace=False)
  )
  (lstm): Sequential(
    (0): LSTM(100, 256, num_layers=2, dropout=0.5, bidirectional=True)
  )
  (out1): Sequential(
    (0): Linear(in_features=512, out_features=128, bias=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): Linear(in_features=128, out_features=1, bias=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (out2): Sequential(
    (0): Linear(in_features=512, out_features=128, bias=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): Linear(in_features=128, out_features=10, bias=True)
    (3): Dropout(p=0.5, inplace=False)
  )
)
```

### Data WordCloud

A word cloud was then created so to visualize the most common `100` words in the corpus that we used to train the model.

<p align="center"><img src="wc.png" alt="cover" width="100%"/></p>

> The word "feel" appears the most in the `corpus`.

### model parameters

In the following table we are going to show the model parameters, both trainable and total parameters.

<table>
    <thead>
        <tr><th>TOTAL MODEL PARAMETERS</th><th>TOTAL TRAINABLE PARAMETERS</th></tr>
    </thead>
    <tbody>
      <tr><td>3,082,291</td><td>3,082,291</td></tr>
    </tbody>
</table>

### model metrics

First let's have a look on how many examples our dataset was having in each set. We had `3` sets which are `train`, `validation` and `test`. In the following table we will see how many examples for each set was used to train this model.

<table border="1">
    <thead>
      <tr>
        <th>TRAIN EXAMPLES</th>
        <th>VALIDATION EXAMPLES</th>
        <th>TEST EXAMPLES</th>
        <th>TOTAL EXAMPLES</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2,838</td>
        <td>1,1361</td>
        <td>1,702</td>
        <td>5,676</td>
      </tr>
    </tbody>
  </table>

The model was trained for `200` epochs and the training the following table shows the training summary for the model.

<table border="1">
    <thead>
      <tr>
        <th>TOTAL EPOCHS</th>
        <th>LAST SAVED EPOCH </th>
        <th>TOTAL TRAINING TIME</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>200</td>
        <td>199</td>
        <td>0:09:39.39</td>
      </tr>
    </tbody>
</table>

The model only trained for `~9.5` minutes and was able to yield the following training history.

<p align="center"><img src="history.png" alt="cover" width="400"/></p>

In the following table we are going to show the best model's `train`, `evaluation` and `test` `accuracy`.

<table border="1">
    <thead>
      <tr>
        <th>MODEL NAME</th>
        <th>TEST ACCURACY(1)</th>
        <th>TEST ACCURACY(2)</th>
        <th>VALIDATION ACCURACY(1)</th>
        <th>VALIDATION ACCURACY(2)</th>
        <th>TRAIN ACCURACY(1)</th>
        <th>TRAIN ACCURACY(2)</th>
         <th>TEST LOSS(1)</th>
         <th>TEST LOSS(2)</th>
        <th>VALIDATION LOSS(1)</th>
        <th>VALIDATION LOSS(2)</th>
        <th>TRAIN LOSS(1)</th>
        <th>TRAIN LOSS(2)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>hsp_model.pt</td>
        <td>99.83%</td>
        <td>99.50%</td>
        <td>99.566%</td>
        <td>99.913%</td>
        <td>76.600%</td>
        <td>65.381%</td>
        <td>0.010</td>
        <td>0.011</td>
        <td>0.010</td>
        <td>0.007</td>
        <td>0.348</td>
        <td>0.863</td>
      </tr>
    </tbody>
</table>

### REST API

This project exposes a `REST` api that is running on port `3001` which can be configured by changing the `AppConfig` in the `server.py` file that looks as follows:

```py
class AppConfig:
    PORT = 3001
    DEBUG = False
```

Here are the example of url's that can be used to make request to the server using these model versions.

```shell
http://localhost:3001/api/v1/predict-stress
```

> Note that all the request should be sent to the server using the `POST` method and the expect a request body with "text" key.

### Expected Response

The expected response at POST `http://localhost:3001/api/v1/predict-stress` json body with key `text` will yield the following `json` response to the client.

```json
{
  "prediction": {
    "_type": {
      "confidence": 1.0,
      "label": "survivors of abuse",
      "labelId": 6
    },
    "label": {
      "confidence": 1.0,
      "label": "stressed",
      "labelId": 0
    },
    "text": "trauma changed the trajectory of my life. but i don't know if i would feel this way about my options if i wasn't anxious and wounded. my ex and i broke up because he never liked to leave the house, even for daytime activities. i wonder sometimes how i am going to feel when i hit middle age. am i going to feel like i do now?"
  },
  "success": true,
  "time": 1.777550220489502
}
```

### Using `cURL`

> To make a `curl` `POST` request at `http://localhost:3001/api/v1/predict-stress` with the the response body with key `text` as follows.

```shell
cURL --request POST --header "Content-Type: application/json" --data '{"text": "trauma this"}' http://127.0.0.1:3001/api/v1/predict-stress
```

### using javascript `fetch` api.

To test this api using javascript fetch api you can do it as follows:

```js
fetch("http://127.0.0.1:3001/api/v1/predict-stress", {
  method: "POST",
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text: "trauma changed the trajectory of my life. but i don't know if i would feel this way about my options if i wasn't anxious and wounded. my ex and i broke up because he never liked to leave the house, even for daytime activities. i wonder sometimes how i am going to feel when i hit middle age. am i going to feel like i do now?",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

### Dataset

The dataset that was used in this project was obtained on [kaggle](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction)

### Notebooks

The `ipynb` notebook that i used for training the model and saving an `.pt` file was can be found at:

1. [Model Training And Saving - `01_HUMAN_STRESS_PREDICTION`](https://github.com/CrispenGari/nlp-pytorch/blob/main/12_HUMAN_STRESS_PREDICTION/01_HUMAN_STRESS_PREDICTION.ipynb)

### LICENSE

In this notebook I'm using the `MIT` license which reads as follows:

```
MIT License

Copyright (c) 2023 crispengari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
