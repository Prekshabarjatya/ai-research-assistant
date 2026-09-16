(function () {
  "use strict";

  var EXAMPLE_QUESTIONS = [
    "What separators does the chunker try first?",
    "Why TF-IDF instead of a neural embedding model?",
    "What happens if no passage is relevant enough?",
  ];

  var activeTagFilter = null;
  var webSearchConfigured = false;
  var researchToggleInitialized = false;

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
  }

  function fmtScore(score) {
    return score.toFixed(3);
  }

  // ---------- status pill ----------
  function refreshStatus() {
    var dot = document.getElementById("status-dot");
    var pill = document.getElementById("status-pill");
    fetch("/health")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        webSearchConfigured = !!data.web_search_configured;
        var dotState = data.groq_configured ? "ok" : "warn";
        dot.className = "dot " + dotState;
        var parts = [
          data.groq_configured ? "groq connected" : "groq not configured",
          data.web_search_configured ? "tavily connected" : "tavily not configured",
          data.documents_indexed + " docs",
          data.chunks_indexed + " chunks",
        ];
        pill.textContent = parts.join(" · ");
        syncResearchToggle();
      })
      .catch(function () {
        dot.className = "dot bad";
        pill.textContent = "service unreachable";
      });
  }

  function syncResearchToggle() {
    var toggle = document.getElementById("research-use-web");
    var label = toggle ? toggle.nextElementSibling : null;
    if (!toggle) return;
    if (!researchToggleInitialized) {
      toggle.checked = webSearchConfigured;
      researchToggleInitialized = true;
    }
    toggle.disabled = !webSearchConfigured;
    if (label) {
      label.textContent = webSearchConfigured
        ? "Include live web search (Tavily)"
        : "Live web search unavailable (TAVILY_API_KEY not configured)";
    }
  }

  // ---------- documents list ----------
  function refreshDocuments() {
    var url = "/api/documents" + (activeTagFilter ? "?tag=" + encodeURIComponent(activeTagFilter) : "");
    return fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderTagFilterRow(data.tags);

        var list = document.getElementById("doc-list");
        var total = document.getElementById("doc-total");
        total.textContent = data.total_chunks + " chunks total";
        if (!data.documents.length) {
          list.innerHTML = '<li class="doc-list-empty">' +
            (activeTagFilter ? "No documents tagged “" + escapeHtml(activeTagFilter) + "”." : "Nothing indexed yet.") +
            "</li>";
          return;
        }
        list.innerHTML = data.documents
          .map(function (d) {
            var tags = (d.tags || [])
              .map(function (t) { return '<span class="doc-tag">' + escapeHtml(t) + "</span>"; })
              .join("");
            return (
              '<li><span class="doc-name-group"><span class="doc-name">' + escapeHtml(d.document) + "</span>" +
              (tags ? '<span class="doc-tags">' + tags + "</span>" : "") + "</span>" +
              '<span class="doc-chunks mono tnum">' + d.chunks + " chunk" + (d.chunks === 1 ? "" : "s") + "</span></li>"
            );
          })
          .join("");
      })
      .catch(function () {
        document.getElementById("doc-list").innerHTML =
          '<li class="doc-list-empty">Couldn’t load the document list.</li>';
      });
  }

  function renderTagFilterRow(tags) {
    var row = document.getElementById("tag-filter-row");
    if (!tags || !tags.length) {
      row.innerHTML = "";
      return;
    }
    var chips = tags.map(function (t) {
      var active = t === activeTagFilter;
      return '<button type="button" class="tag-chip' + (active ? " is-active" : "") + '" data-tag="' +
        escapeHtml(t) + '">' + escapeHtml(t) + "</button>";
    });
    if (activeTagFilter) {
      chips.unshift('<button type="button" class="tag-chip" data-tag="">All</button>');
    }
    row.innerHTML = chips.join("");
    row.querySelectorAll(".tag-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        var tag = chip.getAttribute("data-tag");
        activeTagFilter = tag || null;
        refreshDocuments();
      });
    });
  }

  // ---------- example chips ----------
  function renderExampleChips() {
    var wrap = document.getElementById("example-chips");
    wrap.innerHTML = EXAMPLE_QUESTIONS.map(function (q) {
      return '<button type="button" class="example-chip">' + escapeHtml(q) + "</button>";
    }).join("");
    wrap.querySelectorAll(".example-chip").forEach(function (chip, i) {
      chip.addEventListener("click", function () {
        var textarea = document.getElementById("question");
        textarea.value = EXAMPLE_QUESTIONS[i];
        textarea.focus();
      });
    });
  }

  // ---------- diagram <-> pipeline hover link ----------
  function wirePipelineHover() {
    var items = document.querySelectorAll(".pipeline-item[data-node]");
    var nodes = document.querySelectorAll(".node[data-node]");
    function setActive(name) {
      nodes.forEach(function (n) {
        n.classList.toggle("is-active", n.getAttribute("data-node") === name);
      });
    }
    items.forEach(function (item) {
      var name = item.getAttribute("data-node");
      item.addEventListener("mouseenter", function () { setActive(name); });
      item.addEventListener("focus", function () { setActive(name); });
      item.addEventListener("mouseleave", function () { setActive(null); });
      item.addEventListener("blur", function () { setActive(null); });
    });
  }

  // ---------- ask form ----------
  function renderAnswerSkeleton() {
    var area = document.getElementById("answer-area");
    area.innerHTML =
      '<div class="answer-card">' +
      '<div class="skeleton w-3-5" style="height:20px;"></div>' +
      '<div class="skeleton w-full" style="margin-top:16px;"></div>' +
      '<div class="skeleton w-full"></div>' +
      '<div class="skeleton w-4-5"></div>' +
      "</div>";
  }

  function renderSources(sources) {
    if (!sources.length) return "";
    return (
      '<div class="sources-title">Retrieved passages</div>' +
      sources
        .map(function (s) {
          return (
            '<div class="source-item">' +
            '<div class="source-meta"><span class="mono">' + escapeHtml(s.document) + "</span>" +
            '<span>chunk ' + s.chunk_index + "</span>" +
            '<span class="score tnum">relevance ' + fmtScore(s.score) + "</span></div>" +
            '<div class="source-text">' + escapeHtml(s.text) + "</div>" +
            "</div>"
          );
        })
        .join("")
    );
  }

  function renderAnswer(body) {
    var area = document.getElementById("answer-area");
    var statusClass = body.sources.length === 0 ? "empty" : body.grounded ? "grounded" : "ungrounded";
    var statusLabel = body.sources.length === 0 ? "no match" : body.grounded ? "grounded answer" : "passages only";

    var html = '<div class="answer-card">';
    html += '<span class="answer-status ' + statusClass + '">' + statusLabel + "</span>";
    if (body.answer) {
      html += '<p class="answer-text">' + escapeHtml(body.answer) + "</p>";
    } else if (body.note) {
      html += '<p class="answer-note">' + escapeHtml(body.note) + "</p>";
    }
    html += renderSources(body.sources);
    html += "</div>";
    area.innerHTML = html;
  }

  function renderAnswerError(message) {
    var area = document.getElementById("answer-area");
    area.innerHTML =
      '<div class="answer-card">' +
      '<span class="answer-status empty">request failed</span>' +
      '<p class="answer-note">' + escapeHtml(message) + "</p>" +
      "</div>";
  }

  function wireAskForm() {
    var form = document.getElementById("ask-form");
    var submitBtn = document.getElementById("ask-submit");
    var errorEl = document.getElementById("question-error");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var question = document.getElementById("question").value.trim();
      if (!question) {
        errorEl.textContent = "Type a question first.";
        errorEl.hidden = false;
        return;
      }
      errorEl.hidden = true;

      submitBtn.disabled = true;
      submitBtn.textContent = "Asking…";
      renderAnswerSkeleton();

      fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      })
        .then(function (r) {
          if (!r.ok) {
            return r.json().then(function (err) {
              throw new Error(err.detail || "The server returned an error.");
            });
          }
          return r.json();
        })
        .then(renderAnswer)
        .catch(function (err) {
          renderAnswerError(err.message || "Couldn’t reach the service. Try again.");
        })
        .finally(function () {
          submitBtn.disabled = false;
          submitBtn.textContent = "Ask";
        });
    });
  }

  // ---------- research form (gather -> write -> review) ----------
  function renderResearchSteps() {
    var area = document.getElementById("research-area");
    area.innerHTML =
      '<div class="answer-card">' +
      '<div class="step-indicator">' +
      '<span class="step is-active" id="step-gather">1. Gathering sources</span>' +
      '<span class="sep">→</span>' +
      '<span class="step" id="step-write">2. Drafting report</span>' +
      '<span class="sep">→</span>' +
      '<span class="step" id="step-review">3. Reviewing claims</span>' +
      "</div>" +
      '<div class="skeleton w-full" style="margin-top:18px;"></div>' +
      '<div class="skeleton w-full"></div>' +
      '<div class="skeleton w-4-5"></div>' +
      "</div>";
    // There's no server-sent progress channel, so this just advances the
    // three labels on a timer as a rough sense of where a multi-call
    // pipeline usually is — not a claim about the exact current step.
    var t1 = setTimeout(function () {
      var g = document.getElementById("step-gather");
      var w = document.getElementById("step-write");
      if (g) g.className = "step is-done";
      if (w) w.className = "step is-active";
    }, 3000);
    var t2 = setTimeout(function () {
      var w = document.getElementById("step-write");
      var rv = document.getElementById("step-review");
      if (w) w.className = "step is-done";
      if (rv) rv.className = "step is-active";
    }, 11000);
    return [t1, t2];
  }

  function renderResearchSources(sources) {
    if (!sources.length) return "";
    return (
      '<div class="sources-title">Sources</div>' +
      sources
        .map(function (s) {
          var ref = s.kind === "web"
            ? '<a href="' + escapeHtml(s.reference) + '" target="_blank" rel="noopener">' + escapeHtml(s.reference) + "</a>"
            : "<span class=\"mono\">" + escapeHtml(s.reference) + "</span>";
          return (
            '<div class="source-item">' +
            '<div class="source-meta"><span class="mono">[' + s.kind + "] " + escapeHtml(s.title) + "</span>" + ref + "</div>" +
            '<div class="source-text">' + escapeHtml(s.text) + "</div>" +
            "</div>"
          );
        })
        .join("")
    );
  }

  function downloadReport(question, report) {
    var blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    var slug = question.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "report";
    a.href = url;
    a.download = slug + ".md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function renderResearchResult(question, body) {
    var area = document.getElementById("research-area");
    var statusClass = body.verified ? "grounded" : "needs-revision";
    var statusLabel = body.verified ? "verified" : "needs revision";

    var html = '<div class="answer-card">';
    html += '<span class="answer-status ' + statusClass + '">' + statusLabel + "</span>";
    if (body.revision_count > 0) {
      html += ' <span class="mono" style="font-size:12px; color:var(--ink-faint);">(' + body.revision_count + " revision" + (body.revision_count === 1 ? "" : "s") + ")</span>";
    }
    html += '<div class="report-text">' + escapeHtml(body.report) + "</div>";
    html += '<div class="report-actions"><button type="button" class="btn-secondary" id="download-report-btn">Download report (.md)</button></div>';
    html += renderResearchSources(body.sources);
    html += "</div>";
    area.innerHTML = html;

    var btn = document.getElementById("download-report-btn");
    if (btn) btn.addEventListener("click", function () { downloadReport(question, body.report); });
  }

  function renderResearchError(message) {
    var area = document.getElementById("research-area");
    area.innerHTML =
      '<div class="answer-card">' +
      '<span class="answer-status empty">request failed</span>' +
      '<p class="answer-note">' + escapeHtml(message) + "</p>" +
      "</div>";
  }

  function wireResearchForm() {
    var form = document.getElementById("research-form");
    if (!form) return;
    var submitBtn = document.getElementById("research-submit");
    var errorEl = document.getElementById("research-question-error");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var question = document.getElementById("research-question").value.trim();
      if (!question) {
        errorEl.textContent = "Type a question first.";
        errorEl.hidden = false;
        return;
      }
      errorEl.hidden = true;

      var useWeb = document.getElementById("research-use-web").checked && webSearchConfigured;

      submitBtn.disabled = true;
      submitBtn.textContent = "Researching…";
      var timers = renderResearchSteps();

      fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, use_web: useWeb }),
      })
        .then(function (r) {
          if (!r.ok) {
            return r.json().then(function (err) {
              throw new Error(err.detail || "The server returned an error.");
            });
          }
          return r.json();
        })
        .then(function (body) { renderResearchResult(question, body); })
        .catch(function (err) {
          renderResearchError(err.message || "Couldn’t reach the service. Try again.");
        })
        .finally(function () {
          timers.forEach(clearTimeout);
          submitBtn.disabled = false;
          submitBtn.textContent = "Research";
        });
    });
  }

  // ---------- ingest: paste text ----------
  function wireIngestTextForm() {
    var form = document.getElementById("ingest-text-form");
    var note = document.getElementById("ingest-text-note");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var title = document.getElementById("ingest-title").value.trim();
      var text = document.getElementById("ingest-text").value.trim();
      var tagsRaw = document.getElementById("ingest-text-tags").value;
      var tags = tagsRaw.split(",").map(function (t) { return t.trim(); }).filter(Boolean);
      if (!title || !text) return;

      var btn = form.querySelector("button");
      btn.disabled = true;
      note.className = "field-note";
      note.textContent = "Indexing…";

      fetch("/api/ingest/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title, text: text, tags: tags }),
      })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (err) { throw new Error(err.detail || "Failed to index."); });
          return r.json();
        })
        .then(function (data) {
          note.className = "field-note ok";
          note.textContent = "Added " + data.chunks_added + " chunk" + (data.chunks_added === 1 ? "" : "s") + " from “" + title + "”.";
          form.reset();
          refreshDocuments();
          refreshStatus();
        })
        .catch(function (err) {
          note.className = "field-note bad";
          note.textContent = err.message;
        })
        .finally(function () { btn.disabled = false; });
    });
  }

  // ---------- ingest: file upload ----------
  function wireIngestFileForm() {
    var form = document.getElementById("ingest-file-form");
    var note = document.getElementById("ingest-file-note");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var fileInput = document.getElementById("ingest-file");
      var file = fileInput.files[0];
      var tagsRaw = document.getElementById("ingest-file-tags").value;
      if (!file) return;

      var btn = form.querySelector("button");
      btn.disabled = true;
      note.className = "field-note";
      note.textContent = "Uploading…";

      var formData = new FormData();
      formData.append("file", file);
      formData.append("tags", tagsRaw);

      fetch("/api/ingest/file", { method: "POST", body: formData })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (err) { throw new Error(err.detail || "Failed to index."); });
          return r.json();
        })
        .then(function (data) {
          note.className = "field-note ok";
          note.textContent = "Added " + data.chunks_added + " chunk" + (data.chunks_added === 1 ? "" : "s") + " from “" + data.document + "”.";
          form.reset();
          refreshDocuments();
          refreshStatus();
        })
        .catch(function (err) {
          note.className = "field-note bad";
          note.textContent = err.message;
        })
        .finally(function () { btn.disabled = false; });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderExampleChips();
    wirePipelineHover();
    wireAskForm();
    wireResearchForm();
    wireIngestTextForm();
    wireIngestFileForm();
    refreshStatus();
    refreshDocuments();
  });
})();
