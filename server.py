from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
# import tempfile # Removed
# import os # Removed
import cgi
import time
# import subprocess # Removed
import logging
import numpy as np # Added for NumpyEncoder if needed
from src.DeepBee.software.detection_and_classification import run
# import config
#
# import sentry_sdk
# from sentry_sdk.integrations.logging import LoggingIntegration

# sentry_logging = LoggingIntegration(
#     level=logging.INFO,        # Capture info and above as breadcrumbs
#     event_level=logging.ERROR  # Send errors as events
# )
# sentry_sdk.init(
#     dsn=config.sentry_dsn,
#     integrations=[
#         sentry_logging,
#     ],
#     # Set traces_sample_rate to 1.0 to capture 100%
#     # of transactions for performance monitoring.
#     # We recommend adjusting this value in production.
#     traces_sample_rate=1.0
# )

# Define the request handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Handle GET requests
    def do_GET(self):
        self.send_response(200)  # Send 200 OK status code
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Send the HTML form as the response body
        form_html = '''
        <html>
        <body>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit" value="Upload" />
        </form>
        </body>
        </html>
        '''
        self.wfile.write(form_html.encode('utf-8'))

    # Handle POST requests
    def do_POST(self):
        content_type = self.headers['Content-Type']

        # Removed temporary directory creation
        # reqdir = "/app/tmp/" + str(time.time())+"/"
        # os.makedirs(reqdir, exist_ok=True)

        # Check if the content type is multipart/form-data
        if content_type.startswith('multipart/form-data'):
            # Parse the form data
            form_data = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )

            # Check if 'file' field exists
            if "file" not in form_data:
                 self.send_response(400)
                 self.send_header("Content-type", "application/json")
                 self.end_headers()
                 response = {"message": "Missing 'file' field in form data"}
                 self.wfile.write(json.dumps(response).encode("utf-8"))
                 return

            file_field = form_data['file']

            # Check if it's a valid file upload FieldStorage instance with a filename
            if not isinstance(file_field, cgi.FieldStorage) or not file_field.filename:
                 self.send_response(400)
                 self.send_header("Content-type", "application/json")
                 self.end_headers()
                 response = {"message": "'file' field is not a valid file upload"}
                 self.wfile.write(json.dumps(response).encode("utf-8"))
                 return

            # Read file content into memory
            image_data = file_field.file.read()

            # Removed temporary file saving logic
            # with tempfile.NamedTemporaryFile(dir="/app/tmp", delete=False) as tmp_file:
            #     tmp_file.write(file_field.file.read())
            #     tmp_file_path = tmp_file.name
            #     filename = os.path.basename(file_field.filename)
            #     new_filename = reqdir + filename
            #     os.rename(tmp_file_path, new_filename)

            try:
                # Call run with image_buffer
                result = run(
                    logging=logging,
                    image_buffer=image_data, # Pass image data directly
                )

                # Removed file existence check and reading from file
                # if not os.path.exists(reqdir + "/result.json"): ...
                # with open(reqdir + "/result.json", 'r') as file: ...

                # Process the returned result
                if result is not None and len(result) > 0:
                    # Define NumpyEncoder locally if needed for serialization
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return json.JSONEncoder.default(self, obj)
                    response_data = {'message': 'File processed successfully', 'result': result}
                    response_body = json.dumps(response_data, cls=NumpyEncoder).encode('utf-8')
                else:
                    response_data = {'message': 'Nothing found', 'result': []}
                    response_body = json.dumps(response_data).encode('utf-8')

                # Removed subprocess.call(["rm", "-rf", reqdir])

                self.send_response(200)  # Send 200 OK status code
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(response_body) # Write the constructed response body

            except Exception as e:
                logging.exception(e)
                # Return error response without file cleanup
                self.send_response(500) # Internal Server Error
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'message': 'Error processing image', 'error': str(e)}).encode('utf-8'))
                # Removed subprocess.call(["rm", "-rf", reqdir])
        else:
            # Handle cases where content type is not multipart/form-data
            self.send_response(415) # Unsupported Media Type
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'message': 'Unsupported content type. Please use multipart/form-data.'}
            self.wfile.write(json.dumps(response).encode('utf-8'))


# Create an HTTP server with the request handler
server_address = ('', 8540)  # Listen on all available interfaces, port 8540
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Start the server
print('Server running on port 8540...')
httpd.serve_forever()
