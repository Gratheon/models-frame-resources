from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import tempfile
import os
import cgi
import time
import subprocess
import logging
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

        reqdir = "/app/tmp/" + str(time.time())+"/"
        os.makedirs(reqdir, exist_ok=True)

        # Check if the content type is multipart/form-data
        if content_type.startswith('multipart/form-data'):
            # Parse the form data
            form_data = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            # Get the uploaded file field
            file_field = form_data['file']

            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(dir="/app/tmp", delete=False) as tmp_file:
                # Save the file data to the temporary file
                tmp_file.write(file_field.file.read())

                # Get the temporary file path
                tmp_file_path = tmp_file.name

                # Extract the filename from the uploaded file field
                filename = os.path.basename(file_field.filename)

                # Move the temporary file to the new filename
                new_filename = reqdir + filename
                os.rename(tmp_file_path, new_filename)

            try:
                run(
                    logging=logging,
                    source_filename=new_filename,
                    dir=reqdir,
                )

                if not os.path.exists(reqdir + "/result.json"):
                    self.send_response(200)  # Send 200 OK status code
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {'message': 'Nothing found'}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    
                    subprocess.call(["rm", "-rf", reqdir])
                    return

                with open(reqdir + "/result.json", 'r') as file:
                    response = file.read()
                # response = {'message': 'File processed successfully', 'result': result}

                subprocess.call(["rm", "-rf", reqdir])

                self.send_response(200)  # Send 200 OK status code
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(response.encode('utf-8'))
            except Exception as e:
                logging.exception(e)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'message': 'Error', 'result': str(e)}).encode('utf-8'))

# Create an HTTP server with the request handler
server_address = ('', 8540)  # Listen on all available interfaces, port 8700
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Start the server
print('Server running on port 8540...')
httpd.serve_forever()
