# Axolotl Fine-Tuning UI

A Flask-based dashboard that orchestrates Axolotl fine-tuning jobs for open-source language models. The interface mirrors the OpenAI fine-tuning experience while adding support for methods such as LoRA, QLoRA, Direct Preference Optimization (DPO), and reinforcement learning (RL) style trainers.

## Features

- Launch fine-tuning jobs for supported open-source models with LoRA, QLoRA, DPO, and RL workflows.
- Upload chat_template formatted datasets (see [Axolotl dataset format](https://docs.axolotl.ai/docs/dataset-formats/)).
- Auto-generate Axolotl YAML configuration files with method-specific defaults.
- Persist job metadata, status history, and execution logs in SQLite.
- Stream job events and logs from background threads without blocking the UI.
- Tail docker-executed training runs using the existing `axolotl` container.
- Docker Compose environment that reuses the provided Axolotl service definition.

## Architecture Overview

| Layer | Description |
| --- | --- |
| **Flask app (`app/`)** | Renders the dashboard, handles form submissions, and exposes REST endpoints for jobs, events, and logs. |
| **SQLite database (`instance/app.db`)** | Stores users, training jobs, and event history via SQLAlchemy models. |
| **Background worker** | Spawns a Python thread per job to run `docker exec axolotl accelerate launch -m axolotl.cli.train <config>` while streaming output to log files. |
| **Config builder** | Creates Axolotl YAML configs under `data/configs/` named like `gpt-oss-20b-20250101-lora-1.yaml` with embedded dataset references. |
| **Static assets** | Custom CSS/JS replicating the OpenAI dashboard experience with live polling of job status, events, and logs. |

## Getting Started (Local Python)

1. **Clone & install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the Flask app**

   ```bash
   flask --app wsgi run --debug
   ```

3. **Open** `http://localhost:5000` to access the dashboard.

The app automatically provisions the SQLite database (`instance/app.db`), creates a default admin user (configurable via environment variables), and prepares the shared `data/` directories for datasets, configs, outputs, and logs.

## Running with Docker Compose

The repository contains a `docker-compose.yaml` that expects the existing Axolotl service definition in `existing-axolotl-docker.yaml` (as provided).

```bash
docker compose up --build
```

The `web` service exposes the UI on port 5000 and depends on the `axolotl` service. Shared volumes under `./data` ensure that datasets, configs, and logs are visible to both containers at `/workspace/axolotl-ui/data`. Update environment variables in `docker-compose.yaml` if your Axolotl container uses a different mount point.

### Axolotl Container Assumptions

- The Axolotl service is named `axolotl` and already available through the provided compose file.
- Required dependencies inside the container should be installed according to [Axolotl's getting started guide](https://docs.axolotl.ai/docs/getting-started.html).
- Training commands follow the documented CLI: `accelerate launch -m axolotl.cli.train <config>`.

## Usage Flow

1. **Select a training method** (LoRA, QLoRA, DPO, RL) and **base model** (pre-populated from Axolotl examples).
2. **Upload a dataset** formatted with the Axolotl `chat_template` schema. Validation and optional hyperparameters are accessible via the “Advanced hyperparameters” section.
3. **Launch training**. The system:
   - Stores the dataset under `data/datasets/`.
   - Generates an Axolotl YAML config under `data/configs/`.
   - Creates a per-job log file under `data/logs/`.
   - Saves job metadata in SQLite.
   - Starts a background thread that executes `docker exec axolotl …` and streams output into the log file.
4. **Monitor progress** from the dashboard or job detail page, which polls job metadata, events, and log tail output.

Completed jobs remain in the database, providing a history of successes and failures. Log files are retained for auditability.

## Configuration

| Environment Variable | Purpose | Default |
| --- | --- | --- |
| `DOCKER_CONTAINER_NAME` | Name of the Axolotl container to target with `docker exec`. | `axolotl` |
| `TRAINING_ROOT` | Base directory (in both containers) for datasets, configs, outputs, and logs. | `<repo>/data` |
| `UPLOAD_FOLDER` | Dataset upload directory. | `<TRAINING_ROOT>/datasets` |
| `CONFIG_FOLDER` | Generated config directory. | `<TRAINING_ROOT>/configs` |
| `LOG_FOLDER` | Log directory. | `<TRAINING_ROOT>/logs` |
| `DATABASE_URL` | SQLAlchemy database URI. | `sqlite:///instance/app.db` |
| `DEFAULT_SUPERUSER_EMAIL` | Seed admin email. | `admin@example.com` |
| `DEFAULT_SUPERUSER_NAME` | Seed admin name. | `Administrator` |

Configure these in `.env`, shell exports, or the compose file. Ensure the same `TRAINING_ROOT` path is mounted into the Axolotl container so the training process can read the dataset and config files generated by the UI.

## Project Structure

```
app/
  __init__.py          # Flask app factory
  models.py            # SQLAlchemy models (User, TrainingJob, TrainingEvent)
  routes.py            # UI routes and JSON APIs
  services/
    config_builder.py  # Generates Axolotl YAML configs
    training.py        # Background job orchestration & docker execution
  static/              # CSS/JS assets for the dashboard UI
  templates/           # HTML templates (dashboard & detail views)
config.py              # Application configuration
wsgi.py                # Entrypoint for Flask/Gunicorn
requirements.txt       # Python dependencies
Dockerfile             # Container image for the web UI
docker-compose.yaml    # Web UI + existing Axolotl service
existing-axolotl-docker.yaml # Provided Axolotl service definition
```

## Dataset Format Reference

The uploader expects data that conforms to the `chat_template` schema documented by Axolotl. Refer to the [chat_template documentation](https://docs.axolotl.ai/docs/dataset-formats/) and the [gpt-oss examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss) for practical guidance. Validation and evaluation datasets can be registered via the advanced settings.

## Logging & Monitoring

- Logs are stored in `data/logs/job-<id>.log`. The UI displays the live tail (400 lines by default) with manual refresh controls.
- Events capture key lifecycle changes (creation, configuration paths, command execution, completion) and are stored in the `training_events` table.
- All metadata is queryable via `/api/jobs`, `/api/jobs/<id>`, `/api/jobs/<id>/events`, and `/api/jobs/<id>/logs` endpoints, enabling external automation.

## Contributing

1. Fork and clone the repository.
2. Install dependencies and run the Flask app locally.
3. Submit pull requests describing feature additions or fixes.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
