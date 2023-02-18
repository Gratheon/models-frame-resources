from flask import Flask
from detection_and_classification import run
from platform import python_version

print("Starting app.py")
print(python_version())

app = Flask(__name__)

@app.route("/")
def hello():
    run()
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8540)
