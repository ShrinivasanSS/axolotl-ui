class TrainingFormController {
  constructor(form) {
    this.form = form;
    this.statusEl = document.getElementById('form-status');
    this.form.addEventListener('submit', (event) => this.handleSubmit(event));
  }

  async handleSubmit(event) {
    event.preventDefault();
    const formData = new FormData(this.form);
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
      const article = document.createElement('article');
      article.className = 'job-card';
      article.dataset.jobId = job.id;
      article.innerHTML = `
        <header>
          <h3>${job.display_name}</h3>
          <span class="status status-${job.status}">${job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
        </header>
        <dl>
          <div><dt>Model</dt><dd>${job.base_model}</dd></div>
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
