const promptInput = document.getElementById("prompt-input");
const contextInput = document.getElementById("context-input");
const preferencesInput = document.getElementById("preferences-input");
const validationMessage = document.getElementById("validation-message");
const understandingPanel = document.getElementById("understanding-panel");
const progressPanel = document.getElementById("progress-panel");
const cancelOptimizationButton = document.getElementById("cancel-optimization");
const resultsPanel = document.getElementById("results-panel");
const understandingContent = document.getElementById("understanding-content");
const progressSummary = document.getElementById("progress-summary");
const stepper = document.getElementById("stepper");
const progressIteration = document.getElementById("progress-iteration");
const progressCandidates = document.getElementById("progress-candidates");
const progressModelCalls = document.getElementById("progress-model-calls");
const progressBestQuality = document.getElementById("progress-best-quality");
const progressBestTokens = document.getElementById("progress-best-tokens");
const rateLimitAlert = document.getElementById("rate-limit-alert");
const providerStatusList = document.getElementById("provider-status-list");
const providerStatusMessage = document.getElementById(
  "provider-status-message",
);
const resultsContent = document.getElementById("results-content");
const charCount = document.getElementById("char-count");
const intentEditor = document.getElementById("intent-editor");
const intentEditorWrap = document.getElementById("intent-editor-wrap");
const reportActions = document.getElementById("report-actions");
const sessionView = document.getElementById("session-view");
const sessionEmpty = document.getElementById("session-empty");
const sessionContent = document.getElementById("session-content");
const sessionSummary = document.getElementById("session-summary");
const sessionPrompt = document.getElementById("session-prompt");
const sessionTask = document.getElementById("session-task");
const sessionRubric = document.getElementById("session-rubric");
const sessionCandidates = document.getElementById("session-candidates");
const sessionIterations = document.getElementById("session-iterations");
const sessionEvaluations = document.getElementById("session-evaluations");
const sessionInsights = document.getElementById("session-insights");
const sessionReportActions = document.getElementById("session-report-actions");

const steps = [
  ["understanding_task", "Understanding task"],
  ["building_rubric", "Building rubric"],
  ["generating_candidates", "Generating candidates"],
  ["ucb_selection", "UCB selection"],
  ["performer_execution", "Performer execution"],
  ["judge_evaluation", "Judge evaluation"],
  ["stage_a_optimization", "Stage A optimization"],
  ["insight_update", "Insight update"],
  ["next_iteration", "Next iteration"],
  ["final_evaluation", "Final evaluation"],
  ["report_generation", "Report generation"],
];
let currentSessionId = null;
let currentTaskSpec = null;
let pollingTimer = null;
let optimizationRunning = false;
let activeView = "optimize";
let progressRequestInFlight = false;
let providerCheckInFlight = false;
const optimizationDefaults = {
  active_candidates: null,
  edited_candidates: null,
  max_iterations: null,
  final_top_n: null,
  ucb_c: null,
  stagnation_limit: null,
  min_improvement: null,
  max_llm_calls: null,
};

function applyOptimizationDefaults(defaults = {}) {
  const mapping = {
    active_candidates: "active-candidates",
    edited_candidates: "edited-candidates",
    max_iterations: "max-iterations",
    max_llm_calls: "max-llm-calls",
  };
  Object.entries(mapping).forEach(([key, id]) => {
    const value = defaults[key];
    if (value == null || Number.isNaN(Number(value))) return;
    const field = document.getElementById(id);
    if (field) field.value = Number(value);
    optimizationDefaults[key] = Number(value);
  });
}

async function loadOptimizationDefaults() {
  try {
    const response = await fetch("/api/optimization/defaults");
    if (!response.ok) return;
    const payload = await response.json();
    const defaults = payload?.data || {};
    applyOptimizationDefaults(defaults);
  } catch {
    // The backend config contract is authoritative when available; no hardcoded JS fallback is used.
  }
}

function getOptimizationFieldValue(id, fallback) {
  const field = document.getElementById(id);
  if (!field) return fallback;
  const raw = field.value;
  if (raw === "" || raw === null) return fallback;
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : fallback;
}

document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    switchView(button.dataset.view);
  });
});

document
  .getElementById("refresh-session")
  .addEventListener("click", refreshCurrentSession);
document
  .getElementById("refresh-provider-status")
  .addEventListener("click", () => refreshProviderStatus(true));

function switchView(viewName) {
  activeView = viewName;
  document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
    const selected = button.dataset.view === viewName;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-current", selected ? "page" : "false");
  });
  document.querySelectorAll(".app-view").forEach((view) => {
    view.classList.toggle("hidden", view.id !== `${viewName}-view`);
  });
  if (viewName === "session") refreshCurrentSession();
}

function restoreCurrentSession() {
  currentSessionId = window.sessionStorage.getItem("promptlab-current-session");
  if (!currentSessionId) return;
  refreshCurrentSession().then((session) => {
    if (!session) {
      window.sessionStorage.removeItem("promptlab-current-session");
      currentSessionId = null;
      return;
    }
    currentTaskSpec = session.task_spec || null;
    if (session.status === "awaiting_confirmation" && currentTaskSpec) {
      renderUnderstanding(currentTaskSpec);
      understandingPanel.classList.remove("hidden");
    }
    if (
      [
        "initialized",
        "ready",
        "running",
        "paused_rate_limit",
        "final_evaluation",
        "report_generation",
      ].includes(session.status)
    ) {
      optimizationRunning = true;
      cancelOptimizationButton.disabled = false;
      progressPanel.classList.remove("hidden");
      startPolling();
    } else if (["completed", "failed", "cancelled"].includes(session.status)) {
      optimizationRunning = false;
      cancelOptimizationButton.disabled = true;
      progressPanel.classList.remove("hidden");
      startPolling();
      if (session.status === "completed" && session.final_evaluation) {
        renderResults({
          finalEvaluation: session.final_evaluation,
          stopReason: session.stop_reason,
        });
        if (session.report_path) {
          document.getElementById("download-report").href =
            `/api/optimization/${currentSessionId}/report`;
          reportActions.classList.remove("hidden");
        }
      }
    }
  });
}

restoreCurrentSession();
loadOptimizationDefaults();
refreshProviderStatus();

promptInput.addEventListener("input", () => {
  charCount.textContent = `${promptInput.value.length} characters`;
});

document
  .getElementById("optimize-button")
  .addEventListener("click", async () => {
    hideValidation();
    const prompt = promptInput.value.trim();
    if (!prompt) return showValidation("Please enter a prompt to optimize.");
    if (prompt.length < 5)
      return showValidation(
        "Please enter at least 5 characters of meaningful text.",
      );
    if (prompt.length > 12000) {
      return showValidation(
        "Your prompt is too long for reliable optimization. Please shorten it or provide the essential requirements.",
      );
    }
    progressPanel.classList.remove("hidden");
    cancelOptimizationButton.disabled = true;
    resultsPanel.classList.add("hidden");
    reportActions.classList.add("hidden");
    progressSummary.textContent = "Starting task understanding…";
    renderStepper({});
    setProgressMetrics(null);
    const payload = {
      prompt,
      optional_context: contextInput.value.trim() || null,
      preferences: preferencesInput.value.trim() || null,
      active_candidates: getOptimizationFieldValue(
        "active-candidates",
        optimizationDefaults.active_candidates,
      ),
      max_iterations: getOptimizationFieldValue(
        "max-iterations",
        optimizationDefaults.max_iterations,
      ),
      edited_candidates: getOptimizationFieldValue(
        "edited-candidates",
        optimizationDefaults.edited_candidates,
      ),
      max_llm_calls: getOptimizationFieldValue(
        "max-llm-calls",
        optimizationDefaults.max_llm_calls,
      ),
    };
    try {
      const response = await fetch("/api/optimization/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        progressPanel.classList.add("hidden");
        return showValidation(
          data.error?.message || "Unable to start optimization.",
        );
      }
      currentSessionId = data.data.sessionId;
      window.sessionStorage.setItem(
        "promptlab-current-session",
        currentSessionId,
      );
      currentTaskSpec = data.data.taskSpec;
      renderUnderstanding(currentTaskSpec);
      understandingPanel.classList.remove("hidden");
      progressSummary.textContent =
        "Task understanding is ready. Review it, then confirm to begin optimization.";
      startPolling();
    } catch {
      progressPanel.classList.add("hidden");
      showValidation(
        "Unable to reach PromptLab. Check that the server is running and retry.",
      );
    }
  });

document.getElementById("edit-understanding").addEventListener("click", () => {
  if (!currentTaskSpec) return;
  intentEditor.value = JSON.stringify(currentTaskSpec, null, 2);
  intentEditorWrap.classList.toggle("hidden");
});

document
  .getElementById("confirm-understanding")
  .addEventListener("click", async () => {
    if (!currentSessionId || !currentTaskSpec) return;
    hideValidation();
    let taskSpec = currentTaskSpec;
    if (!intentEditorWrap.classList.contains("hidden")) {
      try {
        taskSpec = JSON.parse(intentEditor.value);
      } catch {
        return showValidation("TaskIntent edits must be valid JSON.");
      }
    }
    try {
      const response = await fetch(
        `/api/optimization/${currentSessionId}/confirm-understanding`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_spec: taskSpec }),
        },
      );
      const data = await response.json();
      if (!response.ok || !data.success) {
        return showValidation(
          data.error?.message || "Unable to confirm task understanding.",
        );
      }
      currentTaskSpec = taskSpec;
      renderUnderstanding(taskSpec);
      intentEditorWrap.classList.add("hidden");
      understandingPanel.classList.add("hidden");
      progressPanel.classList.remove("hidden");
      optimizationRunning = true;
      cancelOptimizationButton.disabled = false;
      progressSummary.textContent =
        "Optimization started. Waiting for backend progress…";
      startPolling();
    } catch {
      showValidation("Unable to confirm task understanding. Please retry.");
    }
  });

document
  .getElementById("cancel-optimization")
  .addEventListener("click", async () => {
    if (!currentSessionId || !optimizationRunning) return;
    if (
      !window.confirm(
        "Cancel the optimization now? Completed evaluations will remain available.",
      )
    )
      return;
    await fetch(`/api/optimization/${currentSessionId}/cancel`, {
      method: "POST",
    });
  });

document.getElementById("reset-button").addEventListener("click", () => {
  if (
    optimizationRunning &&
    !window.confirm("An optimization is running. Leave this session?")
  )
    return;
  if (pollingTimer) window.clearInterval(pollingTimer);
  currentSessionId = null;
  window.sessionStorage.removeItem("promptlab-current-session");
  currentTaskSpec = null;
  optimizationRunning = false;
  promptInput.value = "";
  contextInput.value = "";
  preferencesInput.value = "";
  charCount.textContent = "0 characters";
  understandingPanel.classList.add("hidden");
  progressPanel.classList.add("hidden");
  cancelOptimizationButton.disabled = true;
  resultsPanel.classList.add("hidden");
  reportActions.classList.add("hidden");
  hideValidation();
});

function startPolling() {
  if (pollingTimer) window.clearInterval(pollingTimer);
  refreshProgress();
  pollingTimer = window.setInterval(refreshProgress, 1000);
}

async function refreshProgress() {
  if (!currentSessionId || progressRequestInFlight) return;
  progressRequestInFlight = true;
  try {
    const progressResponse = await fetch(
      `/api/optimization/${currentSessionId}/status`,
    );
    const progressPayload = await progressResponse.json();
    if (!progressResponse.ok || !progressPayload.success) return;
    const p = progressPayload.data;
    renderStepper(p.progress_stages || {});
    const current = Number(p.current_iteration || 0);
    const max = Number(p.max_iterations || 0);
    const active = Number(p.active_candidate_count || 0);
    const total = Number(p.candidate_count || 0);
    const modelCalls = Number(p.llm_call_count || 0);
    const callBudget = p.max_llm_calls == null ? "—" : p.max_llm_calls;
    progressIteration.textContent = `${current} / ${max}`;
    progressCandidates.textContent = `${active} active · ${total} total`;
    progressModelCalls.textContent = `${modelCalls} / ${callBudget}`;
    progressBestQuality.textContent =
      p.current_best_quality == null
        ? "—"
        : `${Math.round(Number(p.current_best_quality) * 100)}%`;
    progressBestTokens.textContent =
      p.current_best_prompt_token_count == null
        ? p.token_count_error || "—"
        : String(p.current_best_prompt_token_count);
    progressSummary.textContent =
      p.status === "awaiting_confirmation"
        ? "Task understanding is ready. Review it, then confirm to begin optimization."
        : p.status === "paused_rate_limit"
          ? "Optimization paused because the provider rate limit was reached."
          : p.user_message ||
            `Status: ${p.status}.` +
              (p.stop_reason ? ` Stop reason: ${p.stop_reason}.` : "") +
              (p.report_error ? ` ${p.report_error}` : "");
    renderRateLimitState(p);
    if (["completed", "failed", "cancelled"].includes(p.status)) {
      optimizationRunning = false;
      cancelOptimizationButton.disabled = true;
      window.clearInterval(pollingTimer);
      pollingTimer = null;
      const resultResponse = await fetch(
        `/api/optimization/${currentSessionId}/results`,
      );
      const resultPayload = await resultResponse.json();
      if (resultPayload.success) renderResults(resultPayload.data);
      if (p.status === "completed" && p.report_ready) {
        document.getElementById("download-report").href =
          `/api/optimization/${currentSessionId}/report`;
        reportActions.classList.remove("hidden");
      }
    }
    if (activeView === "session") await refreshCurrentSession();
  } catch {
    progressSummary.textContent = "Progress update failed; retrying shortly.";
  } finally {
    progressRequestInFlight = false;
  }
}

async function refreshCurrentSession() {
  if (!currentSessionId) {
    sessionEmpty.textContent =
      "No current session yet. Start an optimization to see its prompt, TaskIntent, rubric, candidates, evaluations, and progress here.";
    sessionEmpty.classList.remove("hidden");
    sessionContent.classList.add("hidden");
    return null;
  }
  try {
    const response = await fetch(`/api/optimization/${currentSessionId}`);
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      sessionEmpty.textContent =
        payload.detail ||
        payload.error?.message ||
        "Could not load this session. The server may have restarted and cleared its in-memory data.";
      sessionEmpty.classList.remove("hidden");
      sessionContent.classList.add("hidden");
      return null;
    }
    const session = payload.data;
    renderCurrentSession(session);
    return session;
  } catch {
    sessionEmpty.textContent =
      "Could not reach PromptLab to load the current session.";
    sessionEmpty.classList.remove("hidden");
    sessionContent.classList.add("hidden");
    return null;
  }
}

function renderCurrentSession(session) {
  sessionEmpty.classList.add("hidden");
  sessionContent.classList.remove("hidden");
  sessionPrompt.textContent = session.original_prompt || "";
  renderInfoGrid(sessionSummary, [
    ["Session ID", session.session_id],
    ["Status", session.status],
    [
      "Iteration",
      `${session.current_iteration || 0} / ${session.configuration?.max_iterations || session.max_iterations || 0}`,
    ],
    ["Candidates", (session.candidates || []).length],
    ["Evaluations", (session.evaluations || []).length],
    [
      "LLM calls",
      `${session.llm_call_count || 0} / ${session.configuration?.max_llm_calls ?? "—"}`,
    ],
    ["Stop reason", session.stop_reason || "Not stopped"],
    ["Created", session.created_at || "—"],
  ]);

  const task = session.task_spec || {};
  const taskRows = Object.entries(task).map(([key, value]) => [
    key.replaceAll("_", " "),
    Array.isArray(value)
      ? value.join(", ") || "None"
      : (value ?? "Not specified"),
  ]);
  renderInfoGrid(sessionTask, taskRows);

  sessionRubric.replaceChildren();
  const criteria = session.rubric?.criteria || [];
  if (!criteria.length) {
    sessionRubric.append(
      makeEmptyMessage(
        "Rubric is not available until the TaskIntent is confirmed.",
      ),
    );
  } else {
    criteria.forEach((criterion) => {
      sessionRubric.append(
        makeRecordCard(
          criterion.criterion,
          `${Math.round(Number(criterion.weight) * 100)}% weight`,
          [criterion.description, criterion.scoring_scale].filter(Boolean),
        ),
      );
    });
  }

  sessionCandidates.replaceChildren();
  const candidates = session.candidates || [];
  if (!candidates.length) {
    sessionCandidates.append(
      makeEmptyMessage("No candidates have been generated yet."),
    );
  } else {
    candidates.forEach((candidate) => {
      const parent = candidates.find(
        (item) => item.candidate_id === candidate.parent_id,
      );
      const details = [
        `Status: ${candidate.status || "unknown"} · Generation: ${candidate.generation ?? 0}`,
        `Parent: ${candidate.parent_id || "Original prompt"}${parent ? ` — ${parent.prompt_text}` : ""}`,
        `Pulls: ${candidate.pull_count ?? 0} · Mean reward: ${Number(candidate.mean_reward || 0).toFixed(3)}`,
        candidate.generation_reason || "",
        `Preserved requirements: ${(candidate.preserved_requirements || []).join(", ") || "—"}`,
      ];
      sessionCandidates.append(
        makeRecordCard(
          candidate.candidate_id || "Candidate",
          candidate.source || "source unknown",
          [...details, candidate.prompt_text || ""].filter(Boolean),
          true,
        ),
      );
    });
  }

  sessionIterations.replaceChildren();
  const iterations = session.iterations || [];
  if (!iterations.length) {
    sessionIterations.append(
      makeEmptyMessage("No optimization iterations have completed yet."),
    );
  } else {
    iterations
      .slice()
      .reverse()
      .forEach((iteration) => {
        sessionIterations.append(
          makeRecordCard(
            `Iteration ${iteration.iteration}`,
            `Best reward: ${iteration.best_score == null ? "unavailable" : Number(iteration.best_score).toFixed(3)}`,
            [
              `Sample: ${iteration.sample_id || "—"}`,
              `Selected: ${(iteration.active_candidates || []).join(", ") || "None"}`,
              `Evaluated: ${(iteration.evaluated_candidates || []).join(", ") || "None"}`,
              `New children: ${(iteration.edited_candidates || []).join(", ") || "None"}`,
              `UCB: ${(iteration.ucb_decisions || []).map((item) => `${item.candidate_id}=${item.ucb == null ? "—" : Number(item.ucb).toFixed(3)}${item.selected ? " (selected)" : ""}`).join(" · ") || "No decisions"}`,
            ],
          ),
        );
      });
  }

  sessionEvaluations.replaceChildren();
  const evaluations = session.evaluations || [];
  if (!evaluations.length) {
    sessionEvaluations.append(
      makeEmptyMessage("No candidate evaluations have been recorded yet."),
    );
  } else {
    evaluations
      .slice(-10)
      .reverse()
      .forEach((evaluation) => {
        sessionEvaluations.append(
          makeRecordCard(
            `${evaluation.candidate_id || "Candidate"} · ${evaluation.status || "unknown"}`,
            `Reward: ${evaluation.authoritative_score == null ? "unavailable" : Number(evaluation.authoritative_score).toFixed(3)} · ${evaluation.evaluation_source || evaluation.performer_source || "source unknown"}`,
            [
              `Sample: ${evaluation.sample_id || "—"} · Iteration: ${evaluation.iteration ?? "—"}`,
              evaluation.failure_reason || evaluation.error || "",
              `Strengths: ${(evaluation.strengths || []).join("; ") || "—"}`,
              `Weaknesses: ${(evaluation.weaknesses || []).join("; ") || "—"}`,
              evaluation.response || "",
            ].filter(Boolean),
            true,
          ),
        );
      });
  }

  sessionInsights.replaceChildren();
  const insights = session.insights || [];
  if (!insights.length) {
    sessionInsights.append(
      makeEmptyMessage("No insights have been recorded yet."),
    );
  } else {
    insights
      .slice()
      .reverse()
      .forEach((insight) => {
        sessionInsights.append(
          makeRecordCard(
            insight.text || "Insight",
            `Confidence: ${Number(insight.confidence || 0).toFixed(2)} · ${insight.category || "general"}`,
            [
              `Evidence: ${(insight.evidence || []).join("; ") || "—"}`,
              `Source candidates: ${(insight.provenance || []).map((item) => `${item.candidate_id} (iteration ${item.iteration})`).join(", ") || insight.source_candidate_id || "—"}`,
            ],
          ),
        );
      });
  }

  const reportReady =
    Boolean(session.report_path) &&
    ["completed", "failed", "cancelled"].includes(session.status);
  sessionReportActions.classList.toggle("hidden", !reportReady);
  if (reportReady) {
    document.getElementById("session-download-report").href =
      `/api/optimization/${session.session_id}/report`;
  }
}

function renderInfoGrid(container, rows) {
  container.replaceChildren();
  rows.forEach(([label, value]) => {
    const box = document.createElement("div");
    box.className = "info-box";
    const title = document.createElement("strong");
    title.textContent = String(label);
    const content = document.createElement("span");
    content.textContent = Array.isArray(value)
      ? value.join(", ")
      : String(value ?? "—");
    box.append(title, content);
    container.append(box);
  });
}

function makeRecordCard(titleText, subtitleText, lines, preformatted = false) {
  const card = document.createElement("article");
  card.className = "result-card";
  const title = document.createElement("h3");
  title.textContent = titleText;
  const subtitle = document.createElement("p");
  subtitle.className = "muted-copy";
  subtitle.textContent = subtitleText;
  card.append(title, subtitle);
  lines.forEach((line) => {
    const content = document.createElement(preformatted ? "pre" : "p");
    if (preformatted && line === lines[lines.length - 1])
      content.className = "session-pre";
    content.textContent = String(line);
    card.append(content);
  });
  return card;
}

function makeEmptyMessage(text) {
  const message = document.createElement("p");
  message.className = "muted-copy";
  message.textContent = text;
  return message;
}

function renderResults(data) {
  const finalEvaluation = data.finalEvaluation;
  if (!finalEvaluation) return;
  resultsPanel.classList.remove("hidden");
  resultsContent.replaceChildren();
  const heading = document.createElement("p");
  const terminationReason =
    finalEvaluation.termination_reason ||
    finalEvaluation.stop_reason ||
    data.stopReason;
  const completionText =
    terminationReason &&
    terminationReason !== "max_iterations" &&
    terminationReason !== "stagnation"
      ? `Optimization complete. Stop reason: ${terminationReason}.`
      : "Optimization complete.";
  heading.textContent = completionText;
  resultsContent.append(heading);
  const baseline = finalEvaluation.baseline;
  if (baseline) {
    const baselineEl = document.createElement("p");
    baselineEl.textContent = `Original prompt baseline: ${baseline.score == null ? "unavailable" : Number(baseline.score).toFixed(3)}`;
    resultsContent.append(baselineEl);
  }
  (finalEvaluation.top3 || []).forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const title = document.createElement("h3");
    title.textContent = `Top Candidate ${index + 1}`;
    const score = document.createElement("p");
    score.textContent = `Final quality: ${Number(item.score).toFixed(3)}${item.generation_reason ? ` — ${item.generation_reason}` : ""}`;
    const prompt = document.createElement("pre");
    prompt.textContent = item.prompt_text;
    const copy = document.createElement("button");
    copy.className = "secondary-button";
    copy.textContent = "Copy prompt";
    copy.addEventListener("click", () =>
      navigator.clipboard.writeText(item.prompt_text),
    );
    card.append(title, score, prompt, copy);
    resultsContent.append(card);
  });
}

function renderUnderstanding(taskSpec) {
  understandingContent.replaceChildren();
  const items = [
    ["Primary intent", taskSpec.primary_intent || "Not specified"],
    ["Audience", taskSpec.audience || "Not specified"],
    ["Desired complexity", taskSpec.desired_complexity || "Not specified"],
    ["Style", taskSpec.style || "Not specified"],
    ["Output format", taskSpec.output_format || "Not specified"],
    [
      "Explicit requirements",
      (taskSpec.explicit_requirements || []).join(", ") || "None specified",
    ],
    [
      "Constraints",
      (taskSpec.constraints || []).join(", ") || "None specified",
    ],
    [
      "Potential ambiguities",
      (taskSpec.ambiguities || []).join(", ") || "None identified",
    ],
    [
      "Assumptions to avoid",
      (taskSpec.forbidden_assumptions || []).join(", ") || "None listed",
    ],
  ];
  const grid = document.createElement("div");
  grid.className = "understanding-grid";
  items.forEach(([label, value]) => {
    const box = document.createElement("div");
    box.className = "info-box";
    const title = document.createElement("strong");
    title.textContent = label;
    const content = document.createElement("span");
    content.textContent = String(value || "");
    box.append(title, content);
    grid.append(box);
  });
  understandingContent.append(grid);
}

function renderStepper(progressStages) {
  stepper.replaceChildren();
  steps.forEach(([stage, label]) => {
    const state = progressStages[stage] || "pending";
    const marker =
      state === "completed"
        ? "✓"
        : state === "running"
          ? "●"
          : state === "failed"
            ? "!"
            : "○";
    const item = document.createElement("li");
    item.className = `stepper-item ${state}`;
    item.textContent = `${marker} ${label}`;
    item.setAttribute("aria-label", `${label}: ${state}`);
    stepper.append(item);
  });
}

function setProgressMetrics(status) {
  progressIteration.textContent = status
    ? `${status.current_iteration || 0} / ${status.max_iterations || 0}`
    : "—";
  progressCandidates.textContent = status
    ? `${status.active_candidate_count || 0} active · ${status.candidate_count || 0} total`
    : "—";
  progressModelCalls.textContent = status
    ? `${status.llm_call_count || 0} / ${status.max_llm_calls ?? "—"}`
    : "—";
  progressBestQuality.textContent =
    status?.current_best_quality == null
      ? "—"
      : `${Math.round(Number(status.current_best_quality) * 100)}%`;
  progressBestTokens.textContent =
    status?.current_best_prompt_token_count == null
      ? status?.token_count_error || "—"
      : String(status.current_best_prompt_token_count);
}

function renderRateLimitState(status) {
  const state = status.rate_limit_status;
  if (!state) {
    rateLimitAlert.classList.add("hidden");
    rateLimitAlert.textContent = "";
    return;
  }
  rateLimitAlert.classList.remove("hidden");
  if (state === "waiting" || status.status === "paused_rate_limit") {
    const deadline = status.rate_limit_retry_at
      ? Date.parse(status.rate_limit_retry_at)
      : NaN;
    const remaining = Number.isFinite(deadline)
      ? Math.max(0, Math.ceil((deadline - Date.now()) / 1000))
      : Number(status.rate_limit_retry_after || 0);
    rateLimitAlert.textContent =
      `! Rate limit reached\n${status.rate_limit_provider || "Provider"} is temporarily rate limited.\n` +
      `Retrying in ${remaining} seconds…\n` +
      `Completed: Iteration ${status.current_iteration || 0} / ${status.max_iterations || 0}`;
  } else if (state === "retrying") {
    rateLimitAlert.textContent = "Retrying…";
  } else if (state === "cleared") {
    rateLimitAlert.textContent =
      "✓ Rate limit cleared\nContinuing optimization.";
  } else if (state === "exhausted") {
    rateLimitAlert.textContent = `! ${status.rate_limit_message || "The provider is still rate limited."}`;
  } else {
    rateLimitAlert.classList.add("hidden");
  }
}

async function refreshProviderStatus(forceRefresh = false) {
  if (providerCheckInFlight) return;
  providerCheckInFlight = true;
  const refreshButton = document.getElementById("refresh-provider-status");
  refreshButton.disabled = true;
  providerStatusMessage.textContent = forceRefresh
    ? "Refreshing provider connections…"
    : "Checking provider connections…";
  try {
    const query = forceRefresh ? "?force_refresh=true" : "";
    const response = await fetch(`/api/providers/status${query}`);
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(
        payload.error?.message || "Provider status request failed.",
      );
    }
    renderProviderStatuses(payload.data);
    providerStatusMessage.textContent =
      "Provider health checks are complete. Working means a live provider request succeeded.";
  } catch {
    providerStatusList.replaceChildren();
    providerStatusMessage.textContent =
      "Could not load provider status. Retry when the server is available.";
  } finally {
    providerCheckInFlight = false;
    refreshButton.disabled = false;
  }
}

function renderProviderStatuses(statuses) {
  const providers = [
    ["optimizer", "Gemini Optimizer"],
    ["performer", "Gemini Performer"],
    ["judge", "Groq Judge"],
  ];
  const warningStates = new Set([
    "mock",
    "not_verified",
    "rate_limited",
    "temporarily_unavailable",
    "timeout",
    "network_error",
  ]);
  providerStatusList.replaceChildren();
  providers.forEach(([key, name]) => {
    const details = statuses[key] || {};
    const state = details.working
      ? "working"
      : warningStates.has(details.status)
        ? "warning"
        : "failed";
    const card = document.createElement("article");
    card.className = "provider-status-card";
    card.dataset.state = state;
    const heading = document.createElement("div");
    heading.className = "provider-status-heading";
    const indicator = document.createElement("span");
    indicator.className = "provider-status-indicator";
    indicator.setAttribute("aria-hidden", "true");
    const title = document.createElement("strong");
    title.textContent = `${state === "working" ? "✓" : state === "failed" ? "✕" : "●"} ${name}`;
    heading.append(indicator, title);
    const configured = document.createElement("p");
    configured.textContent = `Configured: ${details.configured ? "Yes" : "No"}`;
    const model = document.createElement("p");
    model.textContent = `Model: ${details.model || "—"}`;
    const status = document.createElement("p");
    status.className = "provider-status-state";
    status.textContent =
      state === "working"
        ? "Working"
        : details.status === "mock"
          ? "Not verified (mock mode)"
          : details.status === "not_verified"
            ? "Configured, not verified"
            : details.status === "rate_limited"
              ? "Rate limited"
              : details.status === "temporarily_unavailable"
                ? "Temporarily unavailable"
                : details.status === "timeout" ||
                    details.status === "network_error"
                  ? "Temporarily unavailable"
                  : details.status === "missing_credential"
                    ? "Not configured"
                    : details.status || "Provider test failed";
    card.append(heading, configured, model, status);
    if (details.message && state !== "working") {
      const message = document.createElement("p");
      message.textContent = details.message;
      card.append(message);
    }
    providerStatusList.append(card);
  });
}

function showValidation(message) {
  validationMessage.textContent = message;
  validationMessage.classList.remove("hidden");
}

function hideValidation() {
  validationMessage.textContent = "";
  validationMessage.classList.add("hidden");
}

window.addEventListener("beforeunload", (event) => {
  if (optimizationRunning) {
    event.preventDefault();
    event.returnValue =
      "An optimization is in progress. Are you sure you want to leave?";
  }
});
