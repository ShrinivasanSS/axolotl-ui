import os
from flask import Flask, current_app
from flask_migrate import Migrate

from config import Config
from .extensions import db
from .models import User


migrate = Migrate()


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True, static_folder="static", template_folder="templates")
    app.config.from_object(config_class)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CONFIG_FOLDER"], exist_ok=True)
    os.makedirs(app.config["LOG_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMPLATE_FOLDER"], exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        db.create_all()
        ensure_default_user()

    from .routes import main_bp, api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


def ensure_default_user() -> None:
    email = current_app.config["DEFAULT_SUPERUSER_EMAIL"]
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, name=current_app.config["DEFAULT_SUPERUSER_NAME"])
        db.session.add(user)
        db.session.commit()
