/* swipesort - swipe deck. Vanilla JS, no build step, no dependencies. */
(() => {
  "use strict";

  const THRESHOLD_X = 0.22;   // fraction of card width before a horizontal swipe commits
  const THRESHOLD_Y = 0.18;   // fraction of card height for vertical
  const REFILL_AT = 5;        // fetch more once the deck is this short
  const BATCH = 20;

  const DIRECTIONS = {
    right: "keep",
    left: "drop",
    up: "love",
    down: "later",
  };

  const MODE_COPY = {
    learn:   ["Learn", "Whatever the model is least sure about. Teaches it fastest."],
    clean:   ["Clean", "Likely rubbish, biggest files first. Frees space fastest."],
    keepers: ["Keepers", "Likely favourites first, for picking the good ones out."],
    backlog: ["Backlog", "Oldest first. The straightforward chronological grind."],
  };

  const el = (id) => document.getElementById(id);
  const deckEl = el("deck");

  const state = {
    mode: localStorage.getItem("swipesort.mode") || "learn",
    includeLater: localStorage.getItem("swipesort.includeLater") === "1",
    year: "",
    bucket: "",
    deck: [],
    offset: 0,
    counts: null,
    info: null,
    busy: false,
    exhausted: false,
    lastDecision: null,
  };

  // ------------------------------------------------------------------ API --

  async function api(path, options) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`${response.status} ${detail.slice(0, 200)}`);
    }
    return response.json();
  }

  const post = (path, body) =>
    api(path, { method: "POST", body: JSON.stringify(body || {}) });

  // -------------------------------------------------------------- helpers --

  function bytes(n) {
    if (!n) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
    const value = n / Math.pow(1024, i);
    return `${value >= 100 || i === 0 ? Math.round(value) : value.toFixed(1)} ${units[i]}`;
  }

  function when(item) {
    if (item.taken_at) {
      const stamp = new Date(item.taken_at);
      if (!Number.isNaN(stamp.getTime())) {
        return stamp.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
      }
    }
    // No capture date, but the year may still have come from the filename -
    // say so rather than claiming there is no date while filing it under one.
    if (item.year && item.year !== "Unclassified") {
      return item.date_source === "filename" ? `${item.year} (from filename)` : item.year;
    }
    return "no date";
  }

  function buzz(ms) {
    if (navigator.vibrate) { try { navigator.vibrate(ms); } catch (_) {} }
  }

  function toast(text, actionLabel, onAction, timeout = 5000) {
    const box = el("toast");
    el("toastText").textContent = text;
    const button = el("toastAction");
    button.hidden = !actionLabel;
    if (actionLabel) {
      button.textContent = actionLabel;
      button.onclick = () => { box.hidden = true; onAction(); };
    }
    box.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { box.hidden = true; }, timeout);
  }

  // ----------------------------------------------------------------- cards --

  function scoreBadge(item) {
    if (item.score === null || item.score === undefined) return null;
    const pct = Math.round(item.score * 100);
    const tone = item.score >= 0.65 ? "score-keep" : item.score <= 0.35 ? "score-drop" : "score-unsure";
    const verdict = item.score >= 0.65 ? "likely keep" : item.score <= 0.35 ? "likely bin" : "unsure";
    return { cls: tone, text: `${pct}% · ${verdict}` };
  }

  function buildCard(item) {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.id = item.id;

    const figure = document.createElement("figure");
    if (item.kind === "video") {
      const video = document.createElement("video");
      video.src = item.media_url;
      video.poster = item.thumb_url;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.preload = "metadata";
      figure.appendChild(video);
    } else {
      const img = document.createElement("img");
      img.src = item.thumb_url;
      img.alt = item.filename;
      img.decoding = "async";
      figure.appendChild(img);
    }
    card.appendChild(figure);

    for (const key of ["keep", "drop", "love", "later"]) {
      const hint = document.createElement("div");
      hint.className = `hint ${key}`;
      hint.innerHTML = `<span>${{ keep: "KEEP", drop: "BIN", love: "LOVE", later: "LATER" }[key]}</span>`;
      card.appendChild(hint);
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "meta-name";
    name.textContent = item.filename;
    meta.appendChild(name);

    const line = document.createElement("div");
    line.className = "meta-line muted";
    const bits = [when(item), bytes(item.size_bytes), item.bucket];
    if (item.width && item.height) bits.push(`${item.width}×${item.height}`);
    if (item.duration_s) bits.push(`${Math.round(item.duration_s)}s`);
    line.append(document.createTextNode(bits.join(" · ")));
    meta.appendChild(line);

    const badges = document.createElement("div");
    badges.className = "meta-line";
    const score = scoreBadge(item);
    if (score) badges.appendChild(badge(score.text, score.cls));
    if (item.dup_group_size > 1) badges.appendChild(badge(`${item.dup_group_size} near-identical`, "dup"));
    if (item.kind === "video") badges.appendChild(badge("video", "video"));
    if (item.deferred) badges.appendChild(badge("deferred", ""));
    badges.appendChild(badge(`→ ${item.target_folder}`, ""));
    if (badges.childElementCount) meta.appendChild(badges);

    card.appendChild(meta);
    return card;
  }

  function badge(text, cls) {
    const span = document.createElement("span");
    span.className = `badge ${cls || ""}`.trim();
    span.textContent = text;
    return span;
  }

  function renderDeck() {
    const visible = state.deck.slice(0, 3);
    const wanted = new Set(visible.map((i) => String(i.id)));

    for (const card of [...deckEl.querySelectorAll(".card")]) {
      if (!wanted.has(card.dataset.id)) card.remove();
    }

    visible.forEach((item, index) => {
      let card = deckEl.querySelector(`.card[data-id="${item.id}"]`);
      if (!card) {
        card = buildCard(item);
        deckEl.appendChild(card);
      }
      card.className = `card${index ? ` behind-${index}` : ""}`;
      card.style.zIndex = String(10 - index);
      if (index === 0) {
        card.style.transform = "";
        const video = card.querySelector("video");
        if (video) video.play().catch(() => {});
      } else {
        const video = card.querySelector("video");
        if (video) video.pause();
      }
    });

    // Warm the next few thumbnails so the deck never shows a blank card.
    state.deck.slice(3, 7).forEach((item) => { new Image().src = item.thumb_url; });

    const nothing = state.deck.length === 0;
    el("empty").hidden = !nothing;
    if (nothing) showEmpty();
    updateChrome();
  }

  function showEmpty() {
    const counts = state.counts || {};
    const filtered = state.year || state.bucket;
    if (filtered) {
      el("emptyTitle").textContent = "Nothing left in this filter";
      el("emptyBody").textContent = "Clear the year or type filter to keep going.";
      const button = el("emptyAction");
      button.hidden = false;
      button.textContent = "Clear filters";
      button.onclick = () => {
        state.year = ""; state.bucket = "";
        el("yearFilter").value = ""; el("bucketFilter").value = "";
        reload();
      };
      return;
    }
    if (!counts.total) {
      el("emptyTitle").textContent = "Nothing indexed yet";
      el("emptyBody").textContent = "Run: python -m swipesort ingest --library <folder>";
      el("emptyAction").hidden = true;
      return;
    }
    el("emptyTitle").textContent = "Backlog cleared";
    el("emptyBody").textContent =
      `${counts.decided.toLocaleString()} decided · ${bytes(counts.drop_bytes)} marked for the bin.`;
    const button = el("emptyAction");
    button.hidden = false;
    button.textContent = "Apply decisions";
    button.onclick = openApply;
  }

  function updateChrome() {
    const counts = state.counts;
    const [label] = MODE_COPY[state.mode] || [state.mode];
    el("modeLabel").textContent = label;
    if (!counts) return;
    const done = counts.decided;
    const total = counts.total - counts.exact_duplicates;
    el("progressFill").style.width = `${total ? (100 * done) / total : 0}%`;
    el("queueNote").textContent =
      `${counts.remaining.toLocaleString()} left · ${bytes(counts.drop_bytes)} binned`;
  }

  // ----------------------------------------------------------------- queue --

  async function fetchMore(reset) {
    if (state.busy) return;
    state.busy = true;
    try {
      const params = new URLSearchParams({
        mode: state.mode,
        limit: String(BATCH),
        offset: String(reset ? 0 : state.offset),
        include_later: state.includeLater ? "true" : "false",
      });
      if (state.year) params.set("year", state.year);
      if (state.bucket) params.set("bucket", state.bucket);
      const data = await api(`/api/queue?${params}`);
      state.counts = data.counts;
      const seen = new Set(state.deck.map((i) => i.id));
      const fresh = data.items.filter((i) => !seen.has(i.id));
      state.exhausted = data.items.length === 0;
      if (reset) {
        state.deck = data.items;
        state.offset = data.items.length;
      } else {
        state.deck.push(...fresh);
        state.offset += data.items.length;
      }
    } catch (error) {
      toast(`Could not load the queue: ${error.message}`);
    } finally {
      state.busy = false;
      renderDeck();
    }
  }

  async function reload() {
    state.offset = 0;
    state.exhausted = false;
    state.deck = [];
    deckEl.querySelectorAll(".card").forEach((c) => c.remove());
    await fetchMore(true);
  }

  // ------------------------------------------------------------- decisions --

  async function decide(action, item) {
    item = item || state.deck[0];
    if (!item) return;

    state.deck.shift();
    state.lastDecision = { item, action, at: Date.now() };
    renderDeck();
    if (state.deck.length < REFILL_AT && !state.exhausted) fetchMore(false);

    try {
      const result = await post("/api/decide", {
        id: item.id,
        action,
        latency_ms: shown ? Date.now() - shown : null,
      });
      state.counts = result.counts;
      updateChrome();
      if (result.model) {
        const accuracy = result.model.holdout_accuracy;
        toast(accuracy != null
          ? `Model retrained · ${Math.round(accuracy * 100)}% on held-out swipes`
          : `Model retrained on ${result.model.n_labels} swipes`);
        // Scores changed underneath us, so the ordering is stale.
        if (state.mode !== "backlog") refreshScoresSoon();
      } else if ((action === "keep" || action === "love") && item.dup_group_size > 1) {
        const others = item.dup_group_size - 1;
        toast(`${others} near-identical shot${others > 1 ? "s" : ""} left`,
              "Bin them", () => dropGroup(item));
      }
    } catch (error) {
      toast(`Swipe not saved: ${error.message}`);
    }
    shown = Date.now();
  }

  async function dropGroup(item) {
    try {
      const result = await post("/api/decide", {
        id: item.id, action: "keep", drop_rest_of_group: true,
      });
      state.counts = result.counts;
      state.deck = state.deck.filter((i) => i.dup_group !== item.dup_group);
      toast(`Binned ${result.also_dropped} near-identical shot(s)`);
      renderDeck();
      if (state.deck.length < REFILL_AT) fetchMore(false);
    } catch (error) {
      toast(`Could not bin the group: ${error.message}`);
    }
  }

  let scoreTimer = null;
  function refreshScoresSoon() {
    clearTimeout(scoreTimer);
    // Let the user keep swiping the cards already on screen; re-rank behind them.
    scoreTimer = setTimeout(() => {
      const keep = state.deck.slice(0, 2);
      state.deck = keep;
      state.offset = 0;
      fetchMore(false);
    }, 1200);
  }

  async function undo() {
    try {
      const result = await post("/api/undo");
      if (!result.ok) { toast("Nothing to undo"); return; }
      state.counts = result.counts;
      if (result.item) {
        state.deck.unshift(result.item);
        renderDeck();
      }
      buzz(10);
      toast(`Undid "${result.undone.action}"`);
    } catch (error) {
      toast(`Undo failed: ${error.message}`);
    }
  }

  // -------------------------------------------------------------- gestures --

  let drag = null;
  let shown = Date.now();

  function topCard() {
    return deckEl.querySelector('.card:not([class*="behind"])');
  }

  deckEl.addEventListener("pointerdown", (event) => {
    const card = topCard();
    if (!card || !card.contains(event.target)) return;
    drag = {
      card,
      id: event.pointerId,
      x0: event.clientX,
      y0: event.clientY,
      w: card.offsetWidth,
      h: card.offsetHeight,
    };
    card.classList.remove("snap");
    card.setPointerCapture(event.pointerId);
  });

  deckEl.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.x0;
    const dy = event.clientY - drag.y0;
    drag.dx = dx;
    drag.dy = dy;
    const horizontal = Math.abs(dx) > Math.abs(dy);
    drag.card.style.transform =
      `translate(${dx}px, ${dy}px) rotate(${(dx / drag.w) * 12}deg)`;

    const ratios = {
      keep:  horizontal && dx > 0 ? dx / (drag.w * THRESHOLD_X) : 0,
      drop:  horizontal && dx < 0 ? -dx / (drag.w * THRESHOLD_X) : 0,
      love:  !horizontal && dy < 0 ? -dy / (drag.h * THRESHOLD_Y) : 0,
      later: !horizontal && dy > 0 ? dy / (drag.h * THRESHOLD_Y) : 0,
    };
    for (const [key, ratio] of Object.entries(ratios)) {
      const hint = drag.card.querySelector(`.hint.${key}`);
      if (hint) hint.style.opacity = String(Math.min(1, Math.max(0, ratio)));
    }
  });

  function endDrag(event) {
    if (!drag || event.pointerId !== drag.id) return;
    const { card, dx = 0, dy = 0, w, h } = drag;
    const horizontal = Math.abs(dx) > Math.abs(dy);
    let direction = null;
    if (horizontal && Math.abs(dx) > w * THRESHOLD_X) direction = dx > 0 ? "right" : "left";
    else if (!horizontal && Math.abs(dy) > h * THRESHOLD_Y) direction = dy > 0 ? "down" : "up";

    drag = null;
    if (!direction) {
      card.classList.add("snap");
      card.style.transform = "";
      card.querySelectorAll(".hint").forEach((hint) => { hint.style.opacity = "0"; });
      return;
    }
    flingAndDecide(card, direction);
  }

  deckEl.addEventListener("pointerup", endDrag);
  deckEl.addEventListener("pointercancel", endDrag);

  function flingAndDecide(card, direction) {
    const action = DIRECTIONS[direction];
    const offsets = {
      right: [window.innerWidth * 1.2, -40, 22],
      left: [-window.innerWidth * 1.2, -40, -22],
      up: [0, -window.innerHeight * 1.2, 0],
      down: [0, window.innerHeight * 1.2, 0],
    }[direction];
    card.classList.add("gone");
    card.style.transform = `translate(${offsets[0]}px, ${offsets[1]}px) rotate(${offsets[2]}deg)`;
    card.addEventListener("transitionend", () => card.remove(), { once: true });
    setTimeout(() => card.remove(), 400);
    buzz(action === "drop" ? 18 : 8);
    decide(action);
  }

  // Buttons mirror the swipes exactly, for one-handed use and for desktop.
  document.querySelector(".actions").addEventListener("click", (event) => {
    const button = event.target.closest(".act");
    if (!button) return;
    const action = button.dataset.act;
    if (action === "undo") { undo(); return; }
    const card = topCard();
    if (!card) return;
    const direction = Object.keys(DIRECTIONS).find((d) => DIRECTIONS[d] === action);
    flingAndDecide(card, direction);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    const map = {
      ArrowRight: "right", ArrowLeft: "left", ArrowUp: "up", ArrowDown: "down",
    };
    if (map[event.key]) {
      event.preventDefault();
      const card = topCard();
      if (card) flingAndDecide(card, map[event.key]);
    } else if (event.key === "z" || event.key === "u") {
      undo();
    } else if (event.key === " ") {
      event.preventDefault();
      const video = topCard()?.querySelector("video");
      if (video) video.paused ? video.play() : video.pause();
    }
  });

  // ---------------------------------------------------------------- sheets --

  function wireSheet(sheet) {
    sheet.querySelectorAll("[data-close]").forEach((button) => {
      button.onclick = () => sheet.close();
    });
  }
  ["modeSheet", "applySheet", "statsSheet"].forEach((id) => wireSheet(el(id)));

  el("modeBtn").onclick = () => {
    const list = el("modeList");
    list.innerHTML = "";
    for (const mode of state.info.modes) {
      const [label, why] = MODE_COPY[mode] || [mode, ""];
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(mode === state.mode));
      button.innerHTML = `<strong>${label}</strong><span class="why">${why}</span>`;
      button.onclick = () => {
        state.mode = mode;
        [...list.querySelectorAll("button")].forEach((b) =>
          b.setAttribute("aria-pressed", String(b === button)));
      };
      li.appendChild(button);
      list.appendChild(li);
    }
    el("includeLater").checked = state.includeLater;
    fillFilters();
    el("modeSheet").showModal();
  };

  function fillFilters() {
    const years = el("yearFilter");
    const buckets = el("bucketFilter");
    if (years.options.length <= 1) {
      for (const year of state.info.years) years.add(new Option(year, year));
      for (const bucket of state.info.buckets) buckets.add(new Option(bucket, bucket));
    }
    years.value = state.year;
    buckets.value = state.bucket;
  }

  el("modeApply").onclick = () => {
    state.includeLater = el("includeLater").checked;
    state.year = el("yearFilter").value;
    state.bucket = el("bucketFilter").value;
    localStorage.setItem("swipesort.mode", state.mode);
    localStorage.setItem("swipesort.includeLater", state.includeLater ? "1" : "0");
    el("modeSheet").close();
    reload();
  };

  async function openApply() {
    const sheet = el("applySheet");
    el("applyPreview").textContent = "checking…";
    el("applyConfirm").disabled = true;
    sheet.showModal();
    try {
      const dry = await post("/api/apply", { confirm: false });
      el("applyPreview").textContent =
        `${dry.summary}\n\nBin:    ${dry.quarantined} file(s), ${bytes(dry.quarantined_bytes)}\n` +
        `File:   ${dry.sorted} keeper(s) into year folders\n` +
        `Merge:  ${dry.collapsed} exact duplicate(s)\n` +
        (dry.errors.length ? `\nProblems:\n${dry.errors.join("\n")}` : "");
      el("applyConfirm").disabled = dry.quarantined + dry.sorted + dry.collapsed === 0;
    } catch (error) {
      el("applyPreview").textContent = `Failed: ${error.message}`;
    }
  }
  el("applyBtn").onclick = openApply;

  el("applyConfirm").onclick = async () => {
    el("applyConfirm").disabled = true;
    try {
      const done = await post("/api/apply", { confirm: true });
      el("applySheet").close();
      toast(`${done.summary}. Undo: swipesort undo --batch ${done.batch}`, null, null, 9000);
      reload();
    } catch (error) {
      el("applyPreview").textContent = `Failed: ${error.message}`;
    }
  };

  el("modeLabel").onclick = openStats;
  function openStats() {
    const body = el("statsBody");
    const model = state.info?.model;
    const counts = state.counts || {};
    const rows = [
      ["Indexed", (counts.total || 0).toLocaleString()],
      ["Decided", (counts.decided || 0).toLocaleString()],
      ["Marked for the bin", `${counts.drop || 0} · ${bytes(counts.drop_bytes)}`],
      ["Exact duplicates", `${counts.exact_duplicates || 0} · ${bytes(counts.exact_duplicate_bytes)}`],
    ];
    if (model) {
      rows.push(["Trained on", `${model.n_labels} swipes`]);
      if (model.holdout_accuracy != null) {
        rows.push(["Held-out accuracy", `${Math.round(model.holdout_accuracy * 100)}%`]);
      }
      if (model.baseline_accuracy != null) {
        rows.push(["Always-guess baseline", `${Math.round(model.baseline_accuracy * 100)}%`]);
      }
      if (model.holdout_auc != null) rows.push(["AUC", model.holdout_auc.toFixed(2)]);
      rows.push(["Features", `${model.n_features} (${model.feat_kind})`]);
    } else {
      rows.push(["Model", `not trained (needs ~${state.info?.min_labels || 25} swipes)`]);
    }
    body.innerHTML = rows
      .map(([k, v]) => `<div class="kv"><span class="muted">${k}</span><span>${v}</span></div>`)
      .join("");
    el("statsSheet").showModal();
  }

  el("trainNow").onclick = async () => {
    el("trainNow").disabled = true;
    try {
      const result = await post("/api/train");
      state.info = await api("/api/state");
      toast(result.ok ? "Model retrained" : result.reason);
      el("statsSheet").close();
      if (result.ok) reload();
    } catch (error) {
      toast(`Retrain failed: ${error.message}`);
    } finally {
      el("trainNow").disabled = false;
    }
  };

  // ------------------------------------------------------------------ boot --

  (async function start() {
    try {
      state.info = await api("/api/state");
      state.counts = state.info.counts;
      if (!state.info.modes.includes(state.mode)) state.mode = "learn";
    } catch (error) {
      toast(`Could not reach swipesort: ${error.message}`, null, null, 20000);
    }
    await fetchMore(true);
    shown = Date.now();
  })();
})();
