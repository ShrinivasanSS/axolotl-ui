class TrainingFormController {
  constructor(form) {
    this.form = form;
    this.statusEl = document.getElementById('form-status');
    this.form.addEventListener('submit', (event) => this.handleSubmit(event));

    this.datasetRadios = Array.from(this.form.querySelectorAll('input[name="dataset_mode"]'));
    this.datasetUploadGroup = this.form.querySelector('[data-dataset-upload]');
    this.datasetExistingGroup = this.form.querySelector('[data-dataset-existing]');
    this.datasetInput = this.form.querySelector('#dataset');
    this.existingDatasetSelect = this.form.querySelector('#existing_dataset');
    this.datasetEmptyHint = this.form.querySelector('[data-existing-empty]');

    this.datasetRadios.forEach((radio) =>
      radio.addEventListener('change', () => this.updateDatasetMode())
    );

    this.updateDatasetMode();
    this.loadDatasets();
  }

  async handleSubmit(event) {
    event.preventDefault();
    const formData = new FormData(this.form);
    const mode = this.getDatasetMode();

    if (mode === 'existing') {
      if (!this.existingDatasetSelect || !this.existingDatasetSelect.value) {
        this.setStatus('Please select a stored dataset or choose Upload new dataset.', 'error');
        return;
      }
      formData.delete('dataset');
    } else if (!this.datasetInput || this.datasetInput.files.length === 0) {
      this.setStatus('Please choose a dataset file to upload.', 'error');
      return;
    }

    this.setStatus('Submitting fine-tuning job…');
    try {
      const response = await fetch('/train', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Failed to submit job');
      }
      this.setStatus('Job created. Redirecting…', 'success');
      if (payload.redirect) {
        window.location.href = payload.redirect;
      }
    } catch (error) {
      this.setStatus(error.message, 'error');
    }
  }

  setStatus(message, type = 'info') {
    if (!this.statusEl) return;
    this.statusEl.textContent = message;
    this.statusEl.dataset.state = type;
  }

  getDatasetMode() {
    const checked = this.datasetRadios.find((radio) => radio.checked);
    return checked ? checked.value : 'upload';
  }

  updateDatasetMode() {
    const mode = this.getDatasetMode();
    const existingRadio = this.datasetRadios.find((radio) => radio.value === 'existing');
    const uploadRadio = this.datasetRadios.find((radio) => radio.value === 'upload');
    if (this.datasetUploadGroup) {
      this.datasetUploadGroup.hidden = mode !== 'upload';
    }
    if (this.datasetExistingGroup) {
      const hasOptions = this.existingDatasetSelect && !this.existingDatasetSelect.disabled;
      this.datasetExistingGroup.hidden = mode !== 'existing' || !hasOptions;
      if (existingRadio) {
        existingRadio.disabled = !hasOptions;
        if (!hasOptions && existingRadio.checked && uploadRadio) {
          uploadRadio.checked = true;
        }
      }
    }
    if (this.datasetInput) {
      this.datasetInput.required = mode === 'upload';
    }
    if (this.existingDatasetSelect) {
      this.existingDatasetSelect.required = mode === 'existing';
    }
  }

  async loadDatasets() {
    if (!this.existingDatasetSelect) return;
    try {
      const response = await fetch('/api/datasets');
      const datasets = await response.json();
      this.populateExistingDatasets(Array.isArray(datasets) ? datasets : []);
    } catch (error) {
      console.error('Failed to load datasets', error);
      this.populateExistingDatasets([]);
    }
  }

  populateExistingDatasets(datasets) {
    if (!this.existingDatasetSelect) return;
    const select = this.existingDatasetSelect;
    const existingRadio = this.datasetRadios.find((radio) => radio.value === 'existing');
    select.innerHTML = '';

    if (!Array.isArray(datasets) || datasets.length === 0) {
      const placeholder = select.dataset.emptyOption || 'No stored datasets found';
      const option = new Option(placeholder, '');
      option.disabled = true;
      select.appendChild(option);
      select.disabled = true;
      if (existingRadio) {
        existingRadio.disabled = true;
        if (existingRadio.checked) {
          const uploadRadio = this.datasetRadios.find((radio) => radio.value === 'upload');
          if (uploadRadio) uploadRadio.checked = true;
        }
      }
      if (this.datasetEmptyHint) {
        this.datasetEmptyHint.hidden = false;
      }
      this.updateDatasetMode();
      return;
    }

    select.disabled = false;
    if (existingRadio) {
      existingRadio.disabled = false;
    }

    const defaultOption = new Option('Select a dataset…', '');
    select.appendChild(defaultOption);

    for (const dataset of datasets) {
      const size = typeof dataset.size_bytes === 'number' ? this.formatBytes(dataset.size_bytes) : null;
      const label = size ? `${dataset.filename} (${size})` : dataset.filename;
      const option = new Option(label, dataset.id);
      if (dataset.updated_at) {
        option.dataset.updatedAt = dataset.updated_at;
      }
      select.appendChild(option);
    }

    if (this.datasetEmptyHint) {
      this.datasetEmptyHint.hidden = true;
    }
    this.updateDatasetMode();
  }

  formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / 1024 ** exponent;
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[exponent]}`;
  }
}

class JobsListController {
  constructor(container) {
    this.container = container;
    this.refreshBtn = document.getElementById('refresh-jobs');
    if (this.refreshBtn) {
      this.refreshBtn.addEventListener('click', () => this.refresh());
    }
    this.refresh();
    this.startPolling();
  }

  async refresh() {
    try {
      const response = await fetch('/api/jobs');
      const jobs = await response.json();
      this.render(jobs);
    } catch (error) {
      console.error('Failed to load jobs', error);
    }
  }

  render(jobs) {
    if (!Array.isArray(jobs)) return;
    this.container.innerHTML = '';
    if (jobs.length === 0) {
      this.container.innerHTML = `<div class="empty">${this.container.dataset.empty}</div>`;
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const job of jobs) {
      const modelName = job.model_label || job.base_model;
      const article = document.createElement('article');
      article.className = 'job-card';
      article.dataset.jobId = job.id;
      article.innerHTML = `
        <header>
          <h3>${job.display_name}</h3>
          <span class="status status-${job.status}">${job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
        </header>
        <dl>
          <div><dt>Model</dt><dd>${modelName}</dd></div>
          <div><dt>Method</dt><dd>${job.training_method.toUpperCase()}</dd></div>
          <div><dt>Created</dt><dd>${new Date(job.created_at).toLocaleString()}</dd></div>
        </dl>
        <a class="job-link" href="/jobs/${job.id}">View details →</a>
      `;
      fragment.appendChild(article);
    }
    this.container.appendChild(fragment);
  }

  startPolling() {
    setInterval(() => this.refresh(), 15000);
  }
}

class JobDetailController {
  constructor(jobId) {
    this.jobId = jobId;
    this.logEl = document.getElementById('job-log');
    this.eventsEl = document.getElementById('event-list');
    this.refreshLogsBtn = document.getElementById('refresh-logs');
    if (this.refreshLogsBtn) {
      this.refreshLogsBtn.addEventListener('click', () => this.refreshLogs());
    }
    this.refreshLogs();
    this.refreshEvents();
    this.startPolling();
  }

  async refreshLogs() {
    try {
      const response = await fetch(`/api/jobs/${this.jobId}/logs?tail=400`);
      const payload = await response.json();
      if (this.logEl) {
        this.logEl.textContent = payload.log || '';
        this.logEl.scrollTop = this.logEl.scrollHeight;
      }
    } catch (error) {
      console.error('Failed to load logs', error);
    }
  }

  async refreshEvents() {
    try {
      const response = await fetch(`/api/jobs/${this.jobId}/events`);
      const events = await response.json();
      if (!Array.isArray(events) || !this.eventsEl) return;
      this.eventsEl.innerHTML = '';
      for (const event of events) {
        const item = document.createElement('li');
        const timestamp = new Date(event.created_at).toLocaleTimeString();
        item.innerHTML = `<span class="timestamp">${timestamp}</span> ${event.message}`;
        this.eventsEl.appendChild(item);
      }
    } catch (error) {
      console.error('Failed to load events', error);
    }
  }

  startPolling() {
    this.interval = setInterval(() => {
      this.refreshLogs();
      this.refreshEvents();
    }, 5000);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('training-form');
  if (form) {
    new TrainingFormController(form);
  }
  const jobsContainer = document.getElementById('jobs-list');
  if (jobsContainer) {
    new JobsListController(jobsContainer);
  }
  window.JobDetailController = JobDetailController;
});
