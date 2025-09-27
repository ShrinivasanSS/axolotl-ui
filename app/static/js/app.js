class TrainingFormController {
  constructor(form) {
    this.form = form;
    this.statusEl = document.getElementById('form-status');
    this.form.addEventListener('submit', (event) => this.handleSubmit(event));

    this.booleanCheckboxes = Array.from(
      this.form.querySelectorAll('input[type="checkbox"][data-boolean]')
    );

    this.templateRadios = Array.from(this.form.querySelectorAll('input[name="template_mode"]'));
    this.templateExistingGroup = this.form.querySelector('[data-template-existing]');
    this.templateUploadGroup = this.form.querySelector('[data-template-upload]');
    this.templateSelect = this.form.querySelector('#template_choice');
    this.templateFileInput = this.form.querySelector('#template_file');
    this.templateStatusEls = Array.from(this.form.querySelectorAll('[data-template-status]'));
    this.templateEmptyHint = this.form.querySelector('[data-template-empty]');
    this.autoSelectedTemplate = false;

    this.datasetRadios = Array.from(this.form.querySelectorAll('input[name="dataset_mode"]'));
    this.datasetUploadGroup = this.form.querySelector('[data-dataset-upload]');
    this.datasetExistingGroup = this.form.querySelector('[data-dataset-existing]');
    this.datasetInput = this.form.querySelector('#dataset');
    this.existingDatasetSelect = this.form.querySelector('#existing_dataset');
    this.datasetEmptyHint = this.form.querySelector('[data-existing-empty]');

    this.datasetRadios.forEach((radio) =>
      radio.addEventListener('change', () => this.updateDatasetMode())
    );

    this.templateRadios.forEach((radio) =>
      radio.addEventListener('change', () => this.updateTemplateMode())
    );
    if (this.templateSelect) {
      this.templateSelect.addEventListener('change', () => this.handleTemplateSelection());
    }
    if (this.templateFileInput) {
      this.templateFileInput.addEventListener('change', () => this.handleTemplateUpload());
    }

    this.updateDatasetMode();
    this.updateTemplateMode();
    this.loadDatasets();
    this.loadTemplates();
  }

  async handleSubmit(event) {
    event.preventDefault();
    const formData = new FormData(this.form);
    const mode = this.getDatasetMode();
    const templateMode = this.getTemplateMode();

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

    if (templateMode === 'existing') {
      if (!this.templateSelect || !this.templateSelect.value) {
        this.setStatus('Please choose a template from the library or upload a new file.', 'error');
        return;
      }
    } else if (templateMode === 'upload') {
      if (!this.templateFileInput || this.templateFileInput.files.length === 0) {
        this.setStatus('Select a template YAML file to upload.', 'error');
        return;
      }
    }

    this.booleanCheckboxes.forEach((checkbox) => {
      formData.set(checkbox.name, checkbox.checked ? 'true' : 'false');
    });

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

  getTemplateMode() {
    const checked = this.templateRadios.find((radio) => radio.checked);
    return checked ? checked.value : 'existing';
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

  updateTemplateMode() {
    const mode = this.getTemplateMode();
    const existingRadio = this.templateRadios.find((radio) => radio.value === 'existing');
    const uploadRadio = this.templateRadios.find((radio) => radio.value === 'upload');

    if (this.templateExistingGroup) {
      const hasOptions = this.templateSelect && !this.templateSelect.disabled;
      this.templateExistingGroup.hidden = mode !== 'existing' || !hasOptions;
      if (existingRadio) {
        existingRadio.disabled = !hasOptions;
        if (!hasOptions && existingRadio.checked && uploadRadio) {
          uploadRadio.checked = true;
        }
      }
    }

    if (this.templateUploadGroup) {
      this.templateUploadGroup.hidden = mode !== 'upload';
    }

    if (this.templateSelect) {
      this.templateSelect.required = mode === 'existing';
    }
    if (this.templateFileInput) {
      this.templateFileInput.required = mode === 'upload';
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

  async loadTemplates() {
    if (!this.templateSelect) return;
    try {
      const response = await fetch('/api/templates');
      const templates = await response.json();
      this.populateTemplateOptions(Array.isArray(templates) ? templates : []);
    } catch (error) {
      console.error('Failed to load templates', error);
      this.populateTemplateOptions([]);
    }
  }

  populateTemplateOptions(templates) {
    if (!this.templateSelect) return;
    const select = this.templateSelect;
    select.innerHTML = '';

    if (!Array.isArray(templates) || templates.length === 0) {
      const option = new Option(select.dataset.emptyOption || 'No templates available', '');
      option.disabled = true;
      select.appendChild(option);
      select.disabled = true;
      if (this.templateEmptyHint) {
        this.templateEmptyHint.hidden = false;
      }
      this.updateTemplateMode();
      return;
    }

    select.disabled = false;
    const groups = new Map();
    for (const template of templates) {
      const groupLabel = template.group || 'Templates';
      if (!groups.has(groupLabel)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = groupLabel;
        groups.set(groupLabel, optgroup);
      }
      const option = new Option(template.label, template.id);
      option.dataset.source = template.source || '';
      option.dataset.filename = template.filename || '';
      groups.get(groupLabel).appendChild(option);
    }

    const placeholder = new Option('Select a template…', '');
    select.appendChild(placeholder);
    for (const [, groupEl] of groups) {
      select.appendChild(groupEl);
    }

    if (this.templateEmptyHint) {
      this.templateEmptyHint.hidden = true;
    }

    if (!this.autoSelectedTemplate) {
      const firstTemplate = select.querySelector('option[value]:not([value=""])');
      if (firstTemplate) {
        firstTemplate.selected = true;
        this.autoSelectedTemplate = true;
        this.handleTemplateSelection();
      }
    }

    this.updateTemplateMode();
  }

  async handleTemplateSelection() {
    if (!this.templateSelect || !this.templateSelect.value) {
      this.setTemplateStatus('Select a template to load configuration hints.', 'info');
      return;
    }

    const id = this.templateSelect.value;
    const existingRadio = this.templateRadios.find((radio) => radio.value === 'existing');
    if (existingRadio && !existingRadio.checked) {
      existingRadio.checked = true;
      this.updateTemplateMode();
    }
    this.setTemplateStatus('Loading template details…');
    try {
      const response = await fetch(`/api/templates/info?id=${encodeURIComponent(id)}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Failed to load template details');
      }
      this.applyTemplateMetadata(payload.metadata, payload);
      this.setTemplateStatus('Template settings applied.', 'success');
    } catch (error) {
      console.error('Failed to inspect template', error);
      this.setTemplateStatus(error.message || 'Failed to inspect template', 'error');
    }
  }

  async handleTemplateUpload() {
    if (!this.templateFileInput || this.templateFileInput.files.length === 0) {
      this.setTemplateStatus('Select a template YAML file to analyze.', 'info');
      return;
    }

    const file = this.templateFileInput.files[0];
    const uploadRadio = this.templateRadios.find((radio) => radio.value === 'upload');
    if (uploadRadio && !uploadRadio.checked) {
      uploadRadio.checked = true;
      this.updateTemplateMode();
    }
    const data = new FormData();
    data.append('template', file);
    this.setTemplateStatus('Analyzing uploaded template…');
    try {
      const response = await fetch('/api/templates/inspect', {
        method: 'POST',
        body: data,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Failed to analyze template');
      }
      this.applyTemplateMetadata(payload.metadata, { label: file.name, source: 'upload' });
      this.setTemplateStatus('Template analyzed. Settings updated.', 'success');
    } catch (error) {
      console.error('Failed to analyze uploaded template', error);
      this.setTemplateStatus(error.message || 'Failed to analyze template', 'error');
    }
  }

  applyTemplateMetadata(metadata, descriptor = {}) {
    if (!metadata || typeof metadata !== 'object') return;
    const trainingSelect = this.form.querySelector('#training_method');
    const baseModelSelect = this.form.querySelector('#base_model');

    if (metadata.training_method && trainingSelect) {
      const option = Array.from(trainingSelect.options).find(
        (item) => item.value === metadata.training_method
      );
      if (option) {
        trainingSelect.value = metadata.training_method;
      }
    }

    if (baseModelSelect) {
      const modelValue = metadata.model_choice || metadata.resolved_base_model || metadata.base_model;
      if (modelValue) {
        const label = metadata.model_choice ? null : descriptor.label || modelValue;
        this.ensureSelectOption(baseModelSelect, modelValue, label || modelValue);
        baseModelSelect.value = modelValue;
      }
    }

    const params = metadata.parameters || {};
    this.applyFieldValue('learning_rate', params.learning_rate);
    this.applyFieldValue('num_epochs', params.num_epochs);
    this.applyFieldValue('max_steps', params.max_steps);
    this.applyFieldValue('micro_batch_size', params.micro_batch_size);
    this.applyFieldValue('gradient_accumulation_steps', params.gradient_accumulation_steps);
    this.applyFieldValue('save_steps', params.save_steps);
    this.applyFieldValue('logging_steps', params.logging_steps);
    this.applyFieldValue('warmup_steps', params.warmup_steps);
    this.applyFieldValue('chat_template', params.chat_template);
    this.applyFieldValue('wandb_project', params.wandb_project);
    this.applyFieldValue('validation_path', params.validation_path);
    this.applyFieldValue('seed', params.seed);
    this.applyBooleanField('sample_packing', params.sample_packing);
    this.applyBooleanField('flash_attention', params.flash_attention);
    this.applyBooleanField('bf16', params.bf16);
  }

  ensureSelectOption(select, value, label) {
    if (!value) return;
    let option = Array.from(select.options).find((item) => item.value === value);
    if (!option) {
      option = new Option(label || value, value);
      option.dataset.templateInserted = 'true';
      select.add(option);
    }
  }

  applyFieldValue(name, value) {
    if (value === undefined || value === null) return;
    const field = this.form.querySelector(`[name="${name}"]`);
    if (!field) return;
    field.value = String(value);
  }

  applyBooleanField(name, value) {
    if (value === undefined || value === null) return;
    const field = this.form.querySelector(`input[name="${name}"][type="checkbox"]`);
    if (!field) return;
    field.checked = Boolean(value);
  }

  setTemplateStatus(message, state = 'info') {
    if (!this.templateStatusEls.length) return;
    this.templateStatusEls.forEach((el) => {
      el.textContent = message || '';
      el.dataset.state = state;
    });
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
