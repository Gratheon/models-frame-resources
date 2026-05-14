from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import cgi
import numpy as np
import gc  # Added for memory optimization
import time

from gratheon_log_lib import bind_context, clear_context, configure, error_enriched, info, warn
from src.DeepBee.software.detection_and_classification import run

configure()


class LogAdapter:
    def info(self, message, meta=None):
        info(str(message), meta)

    def warn(self, message, meta=None):
        warn(str(message), meta)

    def warning(self, message, meta=None):
        warn(str(message), meta)

    def error(self, message, *args):
        if args:
            meta = {"args": [str(arg) for arg in args]}
        else:
            meta = None
        error_enriched(str(message), Exception(str(message)), meta)

    def exception(self, message, *args):
        if args and isinstance(args[-1], BaseException):
            meta = {"args": [str(arg) for arg in args[:-1]]} if len(args) > 1 else None
            error_enriched(str(message), args[-1], meta)
            return
        error_enriched(str(message), Exception(str(message)))


logger = LogAdapter()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        info(
            "http access log",
            {
                "remote_addr": self.address_string(),
                "request_line": self.requestline,
                "message": format % args,
            },
        )

    def do_GET(self):
        request_id = str(time.time_ns())[-8:]
        bind_context(request_id=request_id)
        info(
            "serving upload form",
            {
                "path": self.path,
                "method": "GET",
                "remote_addr": self.client_address[0] if self.client_address else None,
            },
        )
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
        clear_context()

    def do_POST(self):
        request_id = str(time.time_ns())[-8:]
        bind_context(request_id=request_id)
        started_at = time.perf_counter()
        content_type = self.headers['Content-Type']
        info(
            "incoming frame-resources request",
            {
                "path": self.path,
                "method": "POST",
                "remote_addr": self.client_address[0] if self.client_address else None,
                "content_type": content_type,
                "content_length": self.headers.get('Content-Length'),
            },
        )

        if content_type and content_type.startswith('multipart/form-data'):
            try:
                # Parse the form data
                form_data = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )

                # Check if 'file' field exists
                if "file" not in form_data:
                    warn("rejecting request, missing file field")
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {"message": "Missing 'file' field in form data"}
                    self.wfile.write(json.dumps(response).encode("utf-8"))
                    return

                file_field = form_data['file']

                # Check if it's a valid file upload
                if not isinstance(file_field, cgi.FieldStorage) or not file_field.filename:
                    warn("rejecting request, invalid file upload")
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {"message": "'file' field is not a valid file upload"}
                    self.wfile.write(json.dumps(response).encode("utf-8"))
                    return

                # Read file content into memory
                info(
                    "processing uploaded image",
                    {
                        "filename": file_field.filename,
                    },
                )
                image_data = file_field.file.read()
                info("image payload loaded", {"image_bytes": len(image_data)})

                # Clear form data from memory immediately after reading
                del form_data, file_field
                gc.collect()

                # Call run with image_buffer
                result = run(
                    logging=logger,
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

                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                info(
                    "frame-resources request processed",
                    {
                        "result_count": len(response_data.get('result', [])),
                        "duration_ms": duration_ms,
                    },
                )

                # Clear result from memory
                del result
                gc.collect()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                error_enriched("error processing image", e)

                # Force garbage collection on error
                gc.collect()

                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'message': 'Error processing image', 'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
            finally:
                clear_context()
        else:
            warn("rejecting request, unsupported content type", {"content_type": content_type})
            self.send_response(415)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'message': 'Unsupported content type. Please use multipart/form-data.'}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            clear_context()


# Create an HTTP server with memory-optimized settings
server_address = ('', 8540)
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Configure server for better memory management
httpd.timeout = 300  # 5 minute timeout to prevent hanging requests
httpd.allow_reuse_address = True

info('starting frame-resources server', {"port": 8540})

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    info("shutting down frame-resources server")
    httpd.shutdown()
