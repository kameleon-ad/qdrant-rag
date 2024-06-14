from flask import Flask, request, render_template_string
from config import Config

from .extensions import NCL_AGENT

INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Flask App</title>
</head>
<body>
    <h1>Nice Classification</h1>
    <form method="POST">
        <label for="description">Enter the production description:</label>
        <textarea id="description" name="description" required></textarea>
        <button type="submit">Submit</button>
    </form>
    {% if description %}
        <h3> NCL Class of the <b> {{ description }} </b> </h3>
        <h3> Result </h3>
        <p>{{ result }}</p>
    {% endif %}
</body>
</html>
"""


def create_app(config=None):
    if config is None:
        config = Config

    app = Flask(__name__)

    @app.route("/", methods=["get", "post"])
    def index():
        description = None
        result = None
        if request.method == "POST":
            description = request.form.get("description")
            if description:
                result = NCL_AGENT(description)
        return render_template_string(
            INDEX_TEMPLATE, description=description, result=result
        )

    return app
