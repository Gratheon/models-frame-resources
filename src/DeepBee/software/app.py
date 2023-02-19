# from flask import Flask
from detection_and_classification import run
# from platform import python_version

# app = Flask(__name__)

# print("Python version: ", python_version())

# @app.route("/")
# def hello():
#     # try:
#     #     print("running")
#     run()
#     #     print("Done")
#     # except Exception as e:
#     #     print(e)
#     return "Hello, World!"

if __name__ == "__main__":
    run()
    # app.run(host="0.0.0.0", port=8540, debug=True, threaded=False)
