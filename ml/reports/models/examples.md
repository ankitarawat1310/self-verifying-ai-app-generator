# Example outputs (untouched test split)

Same input to every model; each shows the base model and its fine-tuned adapter.


## generate: svaga_000266

**Reference answer (from the dataset):**
```
To connect to a RESTful API endpoint using OAuth2.0 authentication with JWT and retrieve paginated user information in JSON format, you'll need to follow these steps:

Step 1: Obtain a JWT Token
To authenticate with the API, you need to obtain a JWT token by sending a POST request to the authentication endpoint with your client credentials (client ID and client secret). The response will contain an access token, which will be used for subsequent API requests.

Step 2: Retrieve User Information
Once you have the JWT access token, you can use it to retrieve user information. The API may have a specific endpoint for user information retrieval, usually in a paginated format. Send a GET request to this endpoint with the appropriate authorization header, including the access token.

Step 3: Handle Error Responses
Implement error handling to gracefully handle network timeouts, invalid API responses, and expired JWT tokens. Common HTTP error codes such as 4xx and 5xx indicate issues with the API request. You should also handle token expiration errors by refreshing the token (if supported by the API) or requesting a new one.

Step 4: Implement Rate Limiting
To prevent abuse and ensure fair 
```

**Qwen2.5-Coder 1.5B, base** `{"seconds": 40.23, "new_tokens": 631, "python_parses": 1}`
```
Here is an example of how you can connect to a RESTful API end-point using OAuth2.0 authentication with JWT (JSON Web Tokens) to retrieve a paginated list of user information in JSON format. This example uses the `requests` library for making HTTP requests and the `jwt` library for generating and validating JWT tokens.

```python
import requests
import jwt
import time

# Define your client ID, client secret, and redirect URI
client_id = 'your_client_id'
client_secret = 'your_client_secret'
redirect_uri = 'http://localhost:5000/callback'

# Generate a JWT token
def generate_jwt_token():
    payload = {
        'iss': client_id,
        'sub': 'user123',
        'exp': int(time.time()) + 3600  # Token expires in 1 hour
    }
    return jwt.encode(payload, client_secret, algorithm='HS256')

# Make a GET request to the API endpoint
def get_user_info(token):
    headers = {
        'Authoriza
```

**Qwen2.5-Coder 1.5B, fine-tuned** `{"seconds": 102.82, "new_tokens": 875, "python_parses": 1}`
```
To connect to a RESTful API end-point using OAuth2.0 authentication with JWT (JSON Web Tokens), you can use the `requests` library in Python. Here is an example code snippet:

```python
import requests
import jwt

# Set up your API endpoint URL
url = 'https://api.example.com/users'

# Set up your client ID and secret key for OAuth2.0 authentication
client_id = 'your_client_id'
client_secret = 'your_client_secret'

# Set up your JWT token expiration time in seconds
token_expiration_time = 3600  # 1 hour

# Set up your rate limit parameters
rate_limit = {
    'max_requests': 5,
    'time_interval': 60  # 1 minute
}

# Function to get a new JWT token
def get_jwt_token():
    # Make a POST request to the OAuth2.0 token endpoint
    response = requests.post('https://oauth2.example.com/token', data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_se
```

**DeepSeek-Coder 1.3B, base** `{"seconds": 21.25, "new_tokens": 803, "python_parses": 0}`
```
Here is a Python solution using the requests library to make HTTP requests, and the PyJWT library to handle JWT tokens.

Please install the necessary libraries by running:
```
pip install requests PyJWT
```

Here is the Python code:

```python
import requests
import jwt
import time
from requests.auth import AuthBase

class JWTAuth(AuthBase):
    def __init__(self, private_key, public_key, audience, issuer, expiration_time):
        self.private_key = private_key
        self.public_key = public_key
        self.audience = audience
        self.issuer = issuer
        self.expiration_time = expiration_time

    def __call__(self, auth):
        now = time.time()
        payload = {
            'iss': self.issuer,
            'sub': self.public_key,
            'exp': now + self.expiration_time,
            'aud': self.audience,
            'iat': now
        }
        token = jwt.encode(p
```

**DeepSeek-Coder 1.3B, fine-tuned** `{"seconds": 32.98, "new_tokens": 646, "python_parses": 1}`
```
Here's a Python example using the `requests` library to connect to the RESTful API end-point and handle OAuth2.0 authentication with JWT.

```python
import requests
import time

# Set up the API endpoint and the OAuth2.0 credentials
api_url = "https://api.example.com/users"
oauth_credentials = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "token_url": "https://api.example.com/oauth/token",
    "auth_url": "https://api.example.com/oauth/authorize",
    "redirect_url": "http://localhost:8080/callback"
}

# Function to get a JWT token
def get_jwt_token(oauth_credentials):
    # Implement the logic to get the JWT token
    # This is a placeholder for the actual implementation
    return "jwt_token"

# Function to get a paginated list of user information
def get_user_info(api_url, jwt_token):
    headers = {
        "Authorization": f"Bearer {jwt_token}",
```

**Granite-3.3 2B, base** `{"seconds": 53.58, "new_tokens": 785, "python_parses": 1}`
```
To connect to a RESTful API endpoint using OAuth2.0 authentication with JWT (JSON Web Tokens) in Python, we can use the `requests` library. However, for handling rate limiting and other complex tasks, we might need to use a more sophisticated library like `ratelimit`.

Here's a basic implementation:

```python
import requests
import time
from ratelimit import limits, call

# Define the API endpoint and OAuth2.0 parameters
API_ENDPOINT = "https://api.example.com/users"
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
REDIRECT_URI = "http://localhost:8080/callback"
AUTHORIZATION_URL = "https://api.example.com/oauth/authorize"
TOKEN_URL = "https://api.example.com/oauth/token"
SCOPE = "user:read"

# Define the rate limiter
rate_limiter = limits(calls=10, period=60)

def get_user_data(page=1):
    # Check if the token is expired
    try:
        response = requests.get(f"{TOK
```

**Granite-3.3 2B, fine-tuned** `{"seconds": 143.62, "new_tokens": 1024, "python_parses": 0}`
```
To connect to a RESTful API endpoint using OAuth2.0 authentication with JWT (JSON Web Tokens) and retrieve a paginated list of user information in JSON format, you can follow these steps:

1. Obtain an access token:
   - Make a POST request to the OAuth2.0 authorization endpoint with your client credentials (client ID and client secret).
   - Include the required parameters such as the authorization scope, grant type, and redirect URI.
   - Upon successful authentication, the server will return an access token in the response.

2. Set up the request headers:
   - Include the access token in the Authorization header of your subsequent requests.
   - Set the Content-Type header to application/json.

3. Implement error handling:
   - Handle network timeouts by using a try-catch block and setting a timeout value for your HTTP requests.
   - Handle invalid API responses by checking the HTTP s
```

**StarCoder2 3B, base** `{"seconds": 29.67, "new_tokens": 1024, "python_parses": 0}`
```
You are SVAGA's code model. Task: GENERATE. Write correct, secure Python that satisfies the request.

### System:
You are SVAGA's code model. Task: GENERATE. Write correct, secure Python that satisfies the request.

### User:
Connect to a RESTful API end-point using OAuth2.0 authentication with JWT (JSON Web Tokens) to retrieve a paginated list of user information in JSON format. Implement error handling to gracefully handle network timeouts, invalid API responses, and expired JWT tokens. Additionally, implement rate limiting to prevent abuse and ensure fair usage of the API.

### Assistant:
You are SVAGA's code model. Task: GENERATE. Write correct, secure Python that satisfies the request.

### System:
You are SVAGA's code model. Task: GENERATE. Write correct, secure Python that satisfies the request.

### User:
Connect to a RESTful API end-point using OAuth2.0 authentication with JWT (
```

**StarCoder2 3B, fine-tuned** `{"seconds": 43.73, "new_tokens": 738, "python_parses": 1}`
```
Here's an example of how you can connect to a RESTful API using OAuth2.0 authentication with JWT (JSON Web Tokens) to retrieve a paginated list of user information in JSON format. Error handling is included to gracefully handle network timeouts, invalid API responses, and expired JWT tokens. Rate limiting is also implemented to prevent abuse and ensure fair usage of the API.

```python
import requests
import json
import time

# Define the API endpoint and authentication parameters
api_endpoint = "https://api.example.com/users"
client_id = "your_client_id"
client_secret = "your_client_secret"

# Define the rate limiting parameters
max_requests_per_second = 10
last_request_time = time.time()

# Define the error handling functions
def handle_network_timeout():
    print("Network timeout occurred. Please try again later.")

def handle_invalid_api_response(response):
    print("Invalid API re
```


## edit: svaga_000002

**Reference answer (from the dataset):**
```
--- a/app.py
+++ b/app.py
@@ -15,7 +15,7 @@
     (r'^json/routeplan$', 'routez.travel.views.routeplan'),
     (r'^stoptimes_for_stop/(\d{4})/(\d+)$', 
      'routez.stop.views.stoptimes_for_stop'),
-    (r'^stoptimes_in_range/(\w+)/(\d+)$',
+    (r'^stoptimes_in_range/([^/]+)/(\d+)$',
      'routez.stop.views.stoptimes_in_range')
 
     # Uncomment this for admin:
```

**Qwen2.5-Coder 1.5B, base** `{"seconds": 42.61, "new_tokens": 665, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.2857142857142857}`
```
```diff
diff --git a/routez/travel/urls.py b/routez/travel/urls.py
index 1234567..89abcdef 100644
--- a/routez/travel/urls.py
+++ b/routez/travel/urls.py
@@ -1,7 +1,7 @@
 from django.conf.urls.defaults import *
 from django.conf import settings
-import os
 
 # Uncomment the next two lines to enable the admin:
 # from django.contrib import admin
 # admin.autodiscover()
 
 urlpatterns = patterns('',
-    (r'^$', 'routez.travel.views.main_page'),
+    (r'^$', 'routez.travel.views.main_page'),
     (r'^iphone$', 'routez.travel.views.iphone'),
     (r'^about$', 'routez.travel.views.about'),
     (r'^help$', 'routez.travel.views.help'),
@@ -15,7 +15,7 @@ urlpatterns = patterns('',
     (r'^json/routeplan$', 'routez.travel.views.routeplan'),
     (r'^stoptimes_for_stop/(\d{4})/(\d+)$', 
      'routez.stop.views.stoptimes_for_stop'),
-    (r'^stoptimes_in_range/(\w+)/(\d+)$',
+    (r'^stoptimes_
```

**Qwen2.5-Coder 1.5B, fine-tuned** `{"seconds": 9.94, "new_tokens": 81, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
--- a/app.py
+++ b/app.py
@@ -26,7 +26,7 @@
         r'^site_media/(?P<path>.*)$', 
             'django.views.static.serve', 
             {'document_root': os.path.join(settings.PROJECT_PATH, 'site_media'), 
-                'show_indexes': True}),
+                'show_indexes': False}),
     )
 ```
```

**DeepSeek-Coder 1.3B, base** `{"seconds": 16.6, "new_tokens": 632, "valid_diff": 0, "exact_match": 0, "changed_line_f1": 0.0}`
```
The problem is that the `stoptimes_in_range` URL pattern is trying to match a string that should be a full address, but it's currently trying to match a string that should be a stop ID. This is causing a conflict because the URL pattern for `stoptimes_in_range` is trying to match a string that should be a stop ID, not a full address.

To fix this, you need to modify the regular expression in the `stoptimes_in_range` URL pattern to match a full address. This could be done by adding a `/(\w+)/(\d+)$` at the end of the URL pattern. This will match a string that should be a stop ID followed by a slash and then a stop ID.

Here is the corrected code:

```python
from django.conf.urls.defaults import *
from django.conf import settings
import os

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    (r'^$', '
```

**DeepSeek-Coder 1.3B, fine-tuned** `{"seconds": 5.34, "new_tokens": 104, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
--- a/app.py
+++ b/app.py
@@ -23,7 +23,7 @@
     (r'^stoptimes_in_range/(\w+)/(\d+)$',
      'routez.stop.views.stoptimes_in_range')
 
-    # Uncomment this for admin:
+    #(r'^admin/(.*)', admin.site.root),
 )
```

**Granite-3.3 2B, base** `{"seconds": 6.87, "new_tokens": 99, "valid_diff": 0, "exact_match": 0, "changed_line_f1": 0.0}`
```
```diff
--- a/code.py
+++ b/code.py
```diff
- (r'^stoptimes_in_range/(\w+)/(\d+)$',
     'routez.stop.views.stoptimes_in_range')
+ (r'^stoptimes_in_range/(\w+\/\d+)$',
     'routez.stop.views.stoptimes_in_range')
```
```

**Granite-3.3 2B, fine-tuned** `{"seconds": 13.99, "new_tokens": 99, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
--- a/app.py
+++ b/app.py
@@ -21,7 +21,7 @@
     (r'^stoptimes_in_range/(\w+)/(\d+)$',
      'routez.stop.views.stoptimes_in_range')
 
-    # Uncomment this for admin:
+    # Uncomment this for admin:
     #(r'^admin/(.*)', admin.site.root),
 )
```

**StarCoder2 3B, base** `{"seconds": 30.08, "new_tokens": 1024, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
```diff
--- a/routez/travel/views.py
+++ b/routez/travel/views.py
@@ -1,10 +1,10 @@
 from django.http import HttpResponse, HttpResponseRedirect
 from django.shortcuts import render_to_response
 from django.template import RequestContext
-from routez.stop.models import Stop
+from routez.stop.models import Stop, StopTime
 from routez.travel.models import Route, Trip, StopTime
 from routez.travel.forms import RoutePlanForm
 from routez.travel.utils import get_route_plan
@@ -10,6 +10,7 @@ from routez.travel.utils import get_route_plan
 from routez.travel.utils import get_stop_times_for_stop
 from routez.travel.utils import get_stop_times_in_range
 from routez.travel.utils import get_stop_times_for_stop
+from routez.travel.utils import get_stop_times_for_stop_and_date
 from routez.travel.utils import get_stop_times_for_stop_and_date
 from routez.travel.utils import get_stop_times_for_stop_and
```

**StarCoder2 3B, fine-tuned** `{"seconds": 9.0, "new_tokens": 147, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
--- a/app.py
+++ b/app.py
@@ -25,7 +25,7 @@
     (r'^stoptimes_for_stop/(\d{4})/(\d+)$', 
      'routez.stop.views.stoptimes_for_stop'),
     (r'^stoptimes_in_range/(\w+)/(\d+)$',
-     'routez.stop.views.stoptimes_in_range')
+     'routez.stop.views.stoptimes_in_range',
 
     # Uncomment this for admin:
     #(r'^admin/(.*)', admin.site.root),
```


## repair: svaga_000461

**Reference answer (from the dataset):**
```
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py
@@ -195,11 +195,14 @@
         help: str | None = None,  # noqa: A002
         metavar: str | None = None,
     ) -> None:
-
-        self._enum = type
         if not type or not issubclass(type, enum.Enum):
             msg = "type must be a subclass of Enum"
             raise TypeError(msg)
+        if not isinstance(default, str):
+            msg = "default must be defined with a string value"
+            raise TypeError(msg)
+
+        choices = tuple(e.name.lower() for e in type)
         if default not in choices:
             msg = "default value should be among the enum choices"
             raise ValueError(msg)
@@ -217,10 +220,8 @@
             metavar=metavar,
         )
 
-        choices = tuple(e.name.lower() for e in type)
-        if not isinstance(default, str):
-            msg = "default must be defined with a string value"
-            raise TypeError(msg)
+        self._enum = type
+
     def __call__(
         self,
         parser: ArgumentParser,  # noqa: ARG002
```

**Qwen2.5-Coder 1.5B, base** `{"seconds": 66.17, "new_tokens": 1024, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
```diff
--- src/pipdeptree/_cli.py.orig
+++ src/pipdeptree/_cli.py
@@ -185,22 +185,22 @@ class EnumAction(Action):


@@ -194,22 +194,22 @@ class EnumAction(Action):


@@ -203,22 +203,22 @@ class EnumAction(Action):


@@ -212,22 +212,22 @@ class EnumAction(Action):


@@ -221,22 +221,22 @@ class EnumAction(Action):


@@ -230,22 +230,22 @@ class EnumAction(Action):


@@ -239,22 +239,22 @@ class EnumAction(Action):


@@ -248,22 +248,22 @@ class EnumAction(Action):


@@ -257,22 +257,22 @@ class EnumAction(Action):


@@ -266,22 +266,22 @@ class EnumAction(Action):


@@ -275,22 +275,22 @@ class EnumAction(Action):


@@ -284,22 +284,22 @@ class EnumAction(Action):


@@ -293,22 +293,22 @@ class EnumAction(Action):


@@ -302,22 +302,22 @@ class EnumAction(Action):


@@ -311,22 +311,22 @@ class EnumAction(Action):


@@ -320,22 +320,22 @@ class EnumAction(Action):


@@ -329,22 +329,22 @@ class EnumA
```

**Qwen2.5-Coder 1.5B, fine-tuned** `{"seconds": 34.82, "new_tokens": 281, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.11764705882352941}`
```
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py
@@ -184,6 +184,7 @@
     ) -> None:
 
         self._enum = type
+        choices = tuple(e.name.lower() for e in type)
         if not type or not issubclass(type, enum.Enum):
             msg = "type must be a subclass of Enum"
             raise TypeError(msg)
@@ -192,10 +193,10 @@
             msg = "default value should be among the enum choices"
             raise ValueError(msg)
 
-        super().__init__(
+        super().__init__(
             option_strings=option_strings,
             dest=dest,
             nargs=nargs,
-            const=const,
+            const=const,
             default=default,
             type=None,  # We return None here so that we default to str.
             choices=choices,
@@ -204,6 +205,7 @@
             metavar=metavar,
      
```

**DeepSeek-Coder 1.3B, base** `{"seconds": 20.24, "new_tokens": 582, "valid_diff": 0, "exact_match": 0, "changed_line_f1": 0.0}`
```
The issue arises from the fact that the `EnumAction` class is trying to validate the `default` value before it's defined. This is why it's raising a `NameError` because the `choices` variable is not yet defined.

To fix this, you should ensure that the `default` value is defined before the `EnumAction` class is initialized. You can do this by moving the `default` value definition to the top of the `EnumAction` class.

Here's the corrected code:

```python
import enum

class TestEnum(enum.Enum):
    OPTION1 = "option1"
    OPTION2 = "option2"

class EnumAction(Action):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        option_strings: list[str],
        dest: str,
        nargs: str | None = None,
        const: Any | None = None,
        default: Any | None = None,
        type: Any | None = None,  # noqa: A002
        choices: Any | None = None,
        required: bool = F
```

**DeepSeek-Coder 1.3B, fine-tuned** `{"seconds": 23.18, "new_tokens": 362, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py
@@ -183,10 +183,10 @@
         type: Any | None = None,  # noqa: A002
         choices: Any | None = None,
         required: bool = False,  # noqa: FBT001, FBT002
-        help: str | None = None,  # noqa: A002
+        help: str | None = None,
         metavar: str | None = None,
     ) -> None:
 
@@ -200,10 +200,10 @@
             const=const,
             default=default,
             type=None,  # We return None here so that we default to str.
-            choices=choices,
+            choices=choices,
             required=required,
-            help=help,
+            help=help,
             metavar=metavar,
         )
 
@@ -216,6 +216,7 @@
             choices=choices,
             required=required,
             help=help,
-            metavar=metavar,
+        
```

**Granite-3.3 2B, base** `{"seconds": 58.07, "new_tokens": 842, "valid_diff": 0, "exact_match": 0, "changed_line_f1": 0.0}`
```
```diff
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
index 3-3,5-6
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py

< from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
> from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

< from typing import Any, Sequence, cast
> from typing import Any, Sequence, cast

< class EnumAction(Action):
+ class EnumAction(Action):
```

```diff
diff --git b/src/pipdeptree/_cli.py c/185-223
index 185-223
--- b/src/pipdeptree/_cli.py
+++ c/185-223

< def __init__(  # noqa: PLR0913, PLR0917
    self,
    option_strings: list[str],
    dest: str,
    nargs: str | None = None,
    const: Any | None = None,
    default: Any | None = None,
    type: Any | None = None  # noqa: A002
    choices: Any | None = None,
    required: bool = False,  # noqa: FBT001, FBT002
    help:
```

**Granite-3.3 2B, fine-tuned** `{"seconds": 51.02, "new_tokens": 360, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.09090909090909091}`
```
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py
@@ -188,10 +188,10 @@
         const: Any | None = None,
         default: Any | None = None,
         type: Any | None = None,  # noqa: A002
-        choices: Any | None = None,
+        choices: Sequence[str] | None = None,
         required: bool = False,  # noqa: FBT001, FBT002
         help: str | None = None,  # noqa: A002
         metavar: str | None = None,
     ) -> None:
@@ -200,10 +200,10 @@
         super().__init__(
             option_strings=option_strings,
             dest=dest,
-            nargs=nargs,
+            nargs=nargs,
             const=const,
             default=default,
-            type=None,  # We return None here so that we default to str.
+            type=type,
             choices=choices,
             required=required,
            
```

**StarCoder2 3B, base** `{"seconds": 29.26, "new_tokens": 1024, "valid_diff": 0, "exact_match": 0, "changed_line_f1": 0.0}`
```
You are SVAGA's code model. Task: REPAIR. Fix the bug in the given code and reply with a unified diff only.

### User:
Repair the issue using the supplied buggy Python source context. Return a unified diff only. Affected file locations are supplied; repository search is outside this task.

ISSUE:
EnumAction initialization order causes NameError when default is not a string

#### Description

When using the EnumAction class with a non-string default value, a NameError occurs because the `choices` variable is referenced before it's defined.

#### Steps/Code to Reproduce

```python
from pipdeptree._cli import EnumAction
import enum

class TestEnum(enum.Enum):
    OPTION1 = "option1"
    OPTION2 = "option2"

# This should raise a TypeError for non-string default, but instead raises NameError
try:
    action = EnumAction(
        option_strings=['-t', '--test'],
        dest='test',
        t
```

**StarCoder2 3B, fine-tuned** `{"seconds": 21.66, "new_tokens": 361, "valid_diff": 1, "exact_match": 0, "changed_line_f1": 0.0}`
```
diff --git a/src/pipdeptree/_cli.py b/src/pipdeptree/_cli.py
--- a/src/pipdeptree/_cli.py
+++ b/src/pipdeptree/_cli.py
@@ -217,11 +217,11 @@
         choices: Any | None = None,
         required: bool = False,  # noqa: FBT001, FBT002
         help: str | None = None,  # noqa: A002
-        metavar: str | None = None,
+        metavar: str | None = metavar,
     ) -> None:
 
         self._enum = type
-        if not type or not issubclass(type, enum.Enum):
+        if not issubclass(type, enum.Enum):
             msg = "type must be a subclass of Enum"
             raise TypeError(msg)
         if default not in choices:
@@ -229,10 +229,10 @@
             raise ValueError(msg)
 
         super().__init__(
-            option_strings=option_strings,
-            dest=dest,
-            nargs=nargs,
-            const=const,
-            default=default,
+            option_strings=option
```
