from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import cgi
import logging
import numpy as np
import gc  # Added for memory optimization
from src.DeepBee.software.detection_and_classification import run

# Configure logging with memory-conscious settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

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

    def do_POST(self):
        content_type = self.headers['Content-Type']

        if content_type.startswith('multipart/form-data'):
            try:
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

                # Check if it's a valid file upload
                if not isinstance(file_field, cgi.FieldStorage) or not file_field.filename:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {"message": "'file' field is not a valid file upload"}
                    self.wfile.write(json.dumps(response).encode("utf-8"))
                    return

                # Read file content into memory
                logging.info(f"Processing uploaded image: {file_field.filename}")
                image_data = file_field.file.read()

                # Clear form data from memory immediately after reading
                del form_data, file_field
                gc.collect()

                # Call run with image_buffer
                result = run(
                    logging=logging,
                    image_buffer=image_data,
                )

                # Clear image data from memory after processing
                del image_data
                gc.collect()

                if result is not None and len(result) > 0:
                    # Define NumpyEncoder locally for serialization
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

                # Clear result from memory
                del result
                gc.collect()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                logging.exception(f"Error processing image: {e}")

                # Force garbage collection on error
                gc.collect()

                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'message': 'Error processing image', 'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(415)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'message': 'Unsupported content type. Please use multipart/form-data.'}
            self.wfile.write(json.dumps(response).encode('utf-8'))


# Create an HTTP server with memory-optimized settings
server_address = ('', 8540)
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Configure server for better memory management
httpd.timeout = 300  # 5 minute timeout to prevent hanging requests
httpd.allow_reuse_address = True

print('Sserver running on port 8540...')

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("Shutting down server...")
    httpd.shutdown()
