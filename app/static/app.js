(function () {
  "use strict";

  var EXAMPLE_QUESTIONS = [
    "What separators does the chunker try first?",
    "Why TF-IDF instead of a neural embedding model?",
    "What happens if no passage is relevant enough?",
  ];

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
        var groqState = data.groq_configured ? "ok" : "warn";
        dot.className = "dot " + groqState;
        pill.textContent =
          (data.groq_configured ? "groq connected" : "groq not configured") +
          " · " + data.documents_indexed + " docs · " + data.chunks_indexed + " chunks";
      })
      .catch(function () {
        dot.className = "dot bad";
        pill.textContent = "service unreachable";
      });
  }

  // ---------- documents list ----------
  function refreshDocuments() {
    return fetch("/api/documents")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var list = document.getElementById("doc-list");
        var total = document.getElementById("doc-total");
        total.textContent = data.total_chunks + " chunks total";
        if (!data.documents.length) {
          list.innerHTML = '<li class="doc-list-empty">Nothing indexed yet.</li>';
          return;
        }
        list.innerHTML = data.documents
          .map(function (d) {
            return (
              '<li><span class="doc-name">' + escapeHtml(d.document) + "</span>" +
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

  // ---------- ingest: paste text ----------
  function wireIngestTextForm() {
    var form = document.getElementById("ingest-text-form");
    var note = document.getElementById("ingest-text-note");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var title = document.getElementById("ingest-title").value.trim();
      var text = document.getElementById("ingest-text").value.trim();
      if (!title || !text) return;

      var btn = form.querySelector("button");
      btn.disabled = true;
      note.className = "field-note";
      note.textContent = "Indexing…";

      fetch("/api/ingest/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title, text: text }),
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
      if (!file) return;

      var btn = form.querySelector("button");
      btn.disabled = true;
      note.className = "field-note";
      note.textContent = "Uploading…";

      var formData = new FormData();
      formData.append("file", file);

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
    wireIngestTextForm();
    wireIngestFileForm();
    refreshStatus();
    refreshDocuments();
  });
})();
