from flask import Flask
from src.analysis_api import bp_analysis # absolute imports
from src.core_api import bp_core
'''
Flask API app factory
'''

def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp_core, url_prefix="/api") # routes will be under /api/...
    app.register_blueprint(bp_analysis, url_prefix="/api/analytics")
    return app

if __name__ == "__main__":
    create_app().run(debug=True, port=5000)