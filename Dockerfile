FROM tensorflow/tensorflow

WORKDIR /var/app

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "server.py" ]