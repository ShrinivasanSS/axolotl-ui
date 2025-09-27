import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(os.path.dirname(__file__), 'instance', 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    TRAINING_ROOT = os.environ.get(
        "TRAINING_ROOT",
        os.path.join(os.path.dirname(__file__), "data")
    )
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER",
        os.path.join(TRAINING_ROOT, "datasets")
    )
    CONFIG_FOLDER = os.environ.get(
        "CONFIG_FOLDER",
        os.path.join(TRAINING_ROOT, "configs")
    )
    TEMPLATE_FOLDER = os.environ.get(
        "TEMPLATE_FOLDER",
        os.path.join(TRAINING_ROOT, "templates"),
    )
    LOG_FOLDER = os.environ.get(
        "LOG_FOLDER",
        os.path.join(TRAINING_ROOT, "logs")
    )
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GB uploads
    JOB_POLL_INTERVAL = float(os.environ.get("JOB_POLL_INTERVAL", 2.0))
    DATASET_ALLOWED_EXTENSIONS = {"json", "jsonl", "yaml", "yml"}
    DOCKER_CONTAINER_NAME = os.environ.get("DOCKER_CONTAINER_NAME", "axolotl")
    DEFAULT_SUPERUSER_EMAIL = os.environ.get("DEFAULT_SUPERUSER_EMAIL", "admin@example.com")
    DEFAULT_SUPERUSER_NAME = os.environ.get("DEFAULT_SUPERUSER_NAME", "Administrator")
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"
