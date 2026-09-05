/* ============================================================
   THE REVIEW ROOM — client
   A hash-routed, dependency-free front end over the NLP API.
   ============================================================ */

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const STACKED = window.matchMedia('(max-width: 68rem)');

/* ---------- helpers ---------------------------------------- */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const num = (value) => Number(value ?? 0).toLocaleString('en-US');
const pct = (value, digits = 1) => `${(Number(value ?? 0) * 100).toFixed(digits)}%`;
const fixed = (value, digits = 3) => Number(value ?? 0).toFixed(digits);

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const titleCase = (value) => String(value ?? '')
  .replace(/\b[a-z]/g, (c) => c.toUpperCase());

/** Trim to a whole word so cards never end mid-word. */
function truncate(text, limit = 260) {
  const value = String(text ?? '').trim();
  if (value.length <= limit) return value;
  const cut = value.slice(0, limit);
  return `${cut.slice(0, cut.lastIndexOf(' ')) || cut}…`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
    method: options.body ? (options.method ?? 'POST') : (options.method ?? 'GET'),
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = Array.isArray(payload.detail)
          ? payload.detail.map((d) => d.msg ?? d).join(', ')
          : payload.detail;
      }
    } catch { /* keep the generic message */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ---------- shared fragments ------------------------------- */

const skeleton = (rows = 4) => `<div class="skeleton">${'<i></i>'.repeat(rows)}</div>`;

const empty = (title, body) =>
  `<div class="empty"><strong>${esc(title)}</strong><p>${esc(body)}</p></div>`;

const failure = (title, body) =>
  `<div class="error"><strong>${esc(title)}</strong><p>${esc(body)}</p></div>`;

const TONES = ['positive', 'neutral', 'negative'];

/** The signature instrument: a stacked polarity band with an axis. */
function ribbon(name, counts, note = '') {
  const total = TONES.reduce((sum, tone) => sum + (counts[tone] ?? 0), 0) || 1;
  const segments = TONES
    .filter((tone) => (counts[tone] ?? 0) > 0)
    .map((tone) => {
      const share = (counts[tone] ?? 0) / total;
      const label = share > 0.11 ? `${titleCase(tone)} ${pct(share, 0)}` : '';
      return `<span class="ribbon__seg" data-tone="${tone}" style="--w:${share * 100}%"
        title="${titleCase(tone)} — ${num(counts[tone])} (${pct(share)})">${esc(label)}</span>`;
    }).join('');

  const ticks = TONES
    .map((tone) => `<li>${titleCase(tone)} ${num(counts[tone] ?? 0)}</li>`)
    .join('');

  return `
    <div class="ribbon">
      <div class="ribbon__head">
        <h4 class="ribbon__name">${esc(name)}</h4>
        ${note ? `<p class="ribbon__note">${esc(note)}</p>` : ''}
      </div>
      <div class="ribbon__band">${segments}</div>
      <ul class="ribbon__ticks">${ticks}</ul>
    </div>`;
}

/**
 * A ranked corpus entry — used by retrieval, neighbours and topic samples.
 * Topic samples carry no score or corpus index, so both are opt-in.
 */
function resultRow(item, { scoreLabel = 'Similarity', showScore = true, showIndex = true } = {}) {
  const body = item.summary?.trim() || item.review?.trim() || '';
  const width = clamp(Number(item.score ?? 0), 0, 1) * 100;
  const meta = [
    showScore ? `<span>${esc(scoreLabel)} ${fixed(item.score, 4)}</span>
      <span class="result__meter"><i style="--w:${width}%"></i></span>` : '',
    item.sentiment ? `<span class="dot" data-tone="${esc(item.sentiment)}">${esc(item.sentiment)}</span>` : '',
    showIndex ? `<span>Index ${num(item.index)}</span>` : '',
    item.topic != null ? `<span>Topic ${num(item.topic)}</span>` : '',
  ].filter(Boolean).join('');

  return `
    <article class="result">
      <p class="result__rank">${String(item.rank).padStart(2, '0')}</p>
      <div class="result__main">
        ${item.title ? `<h4 class="result__title">${esc(titleCase(item.title))}</h4>` : ''}
        <p class="result__text">${esc(truncate(body, 340))}</p>
        ${meta ? `<div class="result__meta">${meta}</div>` : ''}
      </div>
    </article>`;
}

/* ---------- reveal choreography ---------------------------- */

function choreograph(view) {
  $$('[data-reveal]', view).forEach((node, index) => {
    node.style.setProperty('--i', index);
  });
}

/** Re-run the entrance animation on freshly injected content. */
function animateIn(container) {
  if (REDUCED || !container) return;
  container.querySelectorAll('.ribbon__seg, .track__fill, .result__meter i')
    .forEach((node) => {
      node.style.animation = 'none';
      void node.offsetWidth;
      node.style.animation = '';
    });
}

function countUp(node, target) {
  const value = Number(target) || 0;
  if (REDUCED || value < 2) { node.textContent = num(value); return; }

  const duration = 900;
  const start = performance.now();
  const tick = (now) => {
    const progress = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    node.textContent = num(Math.round(value * eased));
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ---------- 01 · Corpus ------------------------------------ */

const corpus = {
  loaded: false,
  async enter(view) {
    if (this.loaded) return;
    this.loaded = true;

    const stats = $('#corpus-stats', view);
    stats.innerHTML = skeleton(2);

    let data;
    try {
      data = await api('/api/overview');
    } catch (error) {
      stats.innerHTML = failure('The corpus could not be read', error.message);
      return;
    }

    const cards = [
      { label: 'Reviews analysed', value: data.total_reviews, note: 'full pipeline run' },
      { label: 'Indexed for retrieval', value: data.indexed_reviews, note: 'MiniLM · 384 dimensions' },
      { label: 'Aspects extracted', value: data.aspects.length, note: 'product facets' },
      { label: 'Topics discovered', value: data.topics.length, note: 'latent clusters' },
    ];

    stats.innerHTML = cards.map((card) => `
      <div class="stat">
        <span class="stat__label">${esc(card.label)}</span>
        <span class="stat__value" data-count="${card.value}">0</span>
        <span class="stat__note">${esc(card.note)}</span>
      </div>`).join('') + (data.agreement == null ? '' : `
      <div class="stat stat--signal">
        <span class="stat__label">Model agreement</span>
        <span class="stat__value">${pct(data.agreement, 1)}</span>
        <span class="stat__note">VADER vs. human label</span>
      </div>`);

    $$('[data-count]', stats).forEach((node) => countUp(node, node.dataset.count));

    const ribbons = $('#corpus-ribbons', view);
    const bands = [];
    if (data.labelled) bands.push(ribbon('Human labels', data.labelled, 'as shipped with the dataset'));
    if (data.predicted) bands.push(ribbon('VADER prediction', data.predicted, 'what the engine returned'));
    ribbons.innerHTML = bands.length
      ? bands.join('')
      : empty('No polarity columns', 'The sentiment table did not contain a label column.');
    $('#ribbon-meta', view).textContent = data.agreement == null
      ? ''
      : `${pct(data.agreement, 1)} of predictions match the human label`;
    animateIn(ribbons);

    const top = data.aspects.slice(0, 8);
    const ceiling = Math.max(...top.map((a) => a.total), 1);
    $('#corpus-aspects', view).innerHTML = top.map((aspect) => `
      <li>
        <div class="leaders__top">
          <span class="leaders__name">${esc(titleCase(aspect.aspect))}</span>
          <span class="leaders__count">${num(aspect.total)} · ${pct(aspect.positive_share, 0)} positive</span>
        </div>
        <span class="track">
          <i class="track__fill" data-tone="positive" style="--w:${(aspect.positive / ceiling) * 100}%"></i>
          <i class="track__fill" data-tone="neutral"  style="--w:${(aspect.neutral / ceiling) * 100}%"></i>
          <i class="track__fill" data-tone="negative" style="--w:${(aspect.negative / ceiling) * 100}%"></i>
        </span>
      </li>`).join('');

    const topicCeiling = Math.max(...data.topics.map((t) => t.count), 1);
    $('#corpus-topics', view).innerHTML = data.topics.map((topic) => `
      <li>
        <div class="bars__top">
          <span class="bars__name">Topic ${topic.topic}</span>
          <span class="bars__count">${num(topic.count)} reviews</span>
        </div>
        <span class="track">
          <i class="track__fill" data-tone="ink" style="--w:${(topic.count / topicCeiling) * 100}%"></i>
        </span>
      </li>`).join('');

    $('#colophon-corpus').textContent = `${num(data.total_reviews)} reviews · ${num(data.indexed_reviews)} indexed`;
    animateIn(view);
  },
};

/* ---------- 02 · Verdict ----------------------------------- */

/** Merge overlapping evidence spans, longest and strongest first. */
function markupSpans(evidence) {
  const collected = [];
  const push = (items, kind, weak) => items.forEach(({ term, spans }) => {
    spans.forEach(([start, end]) => collected.push({ start, end, kind, weak, term }));
  });

  push(evidence.negative_phrases, 'neg', 0);
  push(evidence.positive_phrases, 'pos', 0);
  push(evidence.negative_words, 'neg', 1);
  push(evidence.positive_words, 'pos', 1);

  collected.sort((a, b) => (a.weak - b.weak) || (b.end - b.start) - (a.end - a.start) || a.start - b.start);

  const kept = [];
  collected.forEach((span) => {
    const overlaps = kept.some((other) => span.start < other.end && other.start < span.end);
    if (!overlaps) kept.push(span);
  });
  return kept.sort((a, b) => a.start - b.start);
}

function manuscript(text, evidence) {
  const spans = markupSpans(evidence);
  let cursor = 0;
  let html = '';

  spans.forEach((span) => {
    html += esc(text.slice(cursor, span.start));
    html += `<mark data-kind="${span.kind}" data-weak="${span.weak}" title="${esc(span.term)}">`
      + `${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  });
  html += esc(text.slice(cursor));
  return html;
}

function evidenceRows(evidence) {
  const groups = [
    ['Negative phrases', evidence.negative_phrases, 'neg'],
    ['Positive phrases', evidence.positive_phrases, 'pos'],
    ['Negative keywords', evidence.negative_words, 'neg'],
    ['Positive keywords', evidence.positive_words, 'pos'],
  ].filter(([, items]) => items.length);

  if (!groups.length) {
    return `<p class="result__text">No lexicon phrase or keyword fired. The verdict rests on
      polarity scoring alone — typically a factual, non-evaluative review.</p>`;
  }

  return `<div class="evidence">${groups.map(([label, items, tone]) => `
    <div class="evidence__row">
      <span class="evidence__key">${esc(label)}</span>
      <span class="evidence__list">${items.map((item) =>
        `<span class="evidence__item" data-tone="${tone}">${esc(item.term)}</span>`).join('')}</span>
    </div>`).join('')}</div>`;
}

function renderVerdict(result) {
  const tone = result.sentiment.toLowerCase();
  const arrow = { positive: '↑', negative: '↓', neutral: '→' }[tone];

  return `
    <div class="verdict" data-tone="${tone}">
      <div class="verdict__banner">
        <div>
          <span class="verdict__label">Predicted sentiment</span>
          <strong class="verdict__word">${esc(result.sentiment)}</strong>
        </div>
        <div class="verdict__score">
          <span class="verdict__label">Final score</span>
          <b>${arrow} ${fixed(result.final_score, 3)}</b>
        </div>
      </div>

      <div class="verdict__body">
        <div>
          <p class="micro" style="margin-bottom:.55rem">The review, marked up</p>
          <div class="manuscript">${manuscript(result.text, result.evidence)}</div>
          <div class="legend" style="margin-top:.75rem">
            <span><i style="background:var(--neg)"></i> negative language</span>
            <span><i style="background:var(--pos)"></i> positive language</span>
            <span><i style="background:var(--ink-soft);height:2px"></i> dotted = single keyword</span>
          </div>
        </div>

        <dl class="readout">
          <div><dt>Positive</dt><dd>${fixed(result.positive)}</dd></div>
          <div><dt>Negative</dt><dd>${fixed(result.negative)}</dd></div>
          <div><dt>Neutral</dt><dd>${fixed(result.neutral)}</dd></div>
          <div><dt>Compound</dt><dd>${fixed(result.compound)}</dd></div>
          <div><dt>Rule score</dt><dd>${fixed(result.rule_score)}</dd></div>
        </dl>

        <div>
          <p class="micro" style="margin-bottom:.7rem">Evidence · decided by ${esc(result.basis)}</p>
          ${evidenceRows(result.evidence)}
        </div>

        ${result.engine === 'rules-only' ? `<p class="notice">Running on the rule lexicon only.
          Install <b>vaderSentiment</b> to restore full hybrid scoring.</p>` : ''}
      </div>
    </div>`;
}

const verdict = {
  enter(view) {
    const form = $('#verdict-form', view);
    const field = $('#verdict-text', view);
    const out = $('#verdict-out', view);
    if (form.dataset.bound) return;
    form.dataset.bound = '1';

    $$('[data-sample]', view).forEach((button) => {
      button.addEventListener('click', () => {
        field.value = button.dataset.sample;
        field.focus();
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = field.value.trim();
      if (!text) {
        out.innerHTML = empty('Nothing to read', 'Paste a customer review, or pick one of the samples above.');
        return;
      }

      const button = $('.btn', form);
      button.disabled = true;
      out.innerHTML = skeleton(4);
      try {
        const result = await api('/api/sentiment', { body: { text } });
        out.innerHTML = renderVerdict(result);
      } catch (error) {
        out.innerHTML = failure('Analysis failed', error.message);
      } finally {
        button.disabled = false;
      }
    });
  },
};

/* ---------- 03 · Aspects ----------------------------------- */

const aspects = {
  rows: null,
  sort: 'total',
  filter: '',
  selected: null,

  async enter(view) {
    const ledger = $('#aspect-ledger', view);
    if (!this.bound) {
      this.bound = true;
      $('#aspect-filter', view).addEventListener('input', (event) => {
        this.filter = event.target.value.trim().toLowerCase();
        this.paint(view);
      });
      $$('[data-sort]', view).forEach((button) => {
        button.addEventListener('click', () => {
          this.sort = button.dataset.sort;
          $$('[data-sort]', view).forEach((b) => b.classList.toggle('is-active', b === button));
          this.paint(view);
        });
      });
    }

    if (this.rows) return;
    ledger.innerHTML = skeleton(6);
    try {
      this.rows = (await api('/api/aspects')).rows;
    } catch (error) {
      ledger.innerHTML = failure('Aspect table unavailable', error.message);
      return;
    }
    this.selected = this.rows[0]?.aspect ?? null;
    this.paint(view);
    if (this.selected) this.detail(view, this.selected);
  },

  paint(view) {
    const ledger = $('#aspect-ledger', view);
    const visible = this.rows
      .filter((row) => row.aspect.toLowerCase().includes(this.filter))
      .sort((a, b) => (this.sort === 'net' ? b.net - a.net : b[this.sort] - a[this.sort]));

    if (!visible.length) {
      ledger.innerHTML = empty('No aspect matches', `Nothing in the ledger contains “${this.filter}”.`);
      return;
    }

    // Bars show composition, not volume — otherwise the long tail of small
    // aspects collapses into invisible slivers. Volume is the numeral.
    ledger.innerHTML = visible.map((row) => `
      <button type="button" class="ledger__row${row.aspect === this.selected ? ' is-active' : ''}"
              data-aspect="${esc(row.aspect)}">
        <span class="ledger__name">${esc(titleCase(row.aspect))}</span>
        <span class="ledger__num">${num(row.total)}<br>${row.net >= 0 ? '+' : ''}${pct(row.net, 0)}</span>
        <span class="track">
          <i class="track__fill" data-tone="positive" style="--w:${row.positive_share * 100}%"></i>
          <i class="track__fill" data-tone="neutral"  style="--w:${row.neutral_share * 100}%"></i>
          <i class="track__fill" data-tone="negative" style="--w:${row.negative_share * 100}%"></i>
        </span>
      </button>`).join('');

    $$('[data-aspect]', ledger).forEach((button) => {
      button.addEventListener('click', () => {
        this.selected = button.dataset.aspect;
        $$('.ledger__row', ledger).forEach((row) => row.classList.toggle('is-active', row === button));
        this.detail(view, this.selected);
        // Stacked layouts push the dossier below a 28-row ledger; follow the click.
        if (STACKED.matches) {
          $('#aspect-dossier', view).scrollIntoView({
            block: 'start', behavior: REDUCED ? 'auto' : 'smooth',
          });
        }
      });
    });
    animateIn(ledger);
  },

  async detail(view, aspect) {
    const dossier = $('#aspect-dossier', view);
    const row = this.rows.find((item) => item.aspect === aspect);
    if (!row) return;

    dossier.innerHTML = `
      <div class="dossier__head">
        <p class="micro">Aspect dossier</p>
        <h3 class="dossier__name">${esc(titleCase(row.aspect))}</h3>
      </div>
      <div class="dossier__body">
        ${ribbon(`${num(row.total)} mentions`, row, `net ${row.net >= 0 ? '+' : ''}${pct(row.net, 1)}`)}
        <div>
          <p class="micro" style="margin-bottom:.7rem">In their own words</p>
          <div class="dossier__quotes">${skeleton(3)}</div>
        </div>
      </div>`;
    animateIn(dossier);

    try {
      const { reviews } = await api(`/api/aspects/${encodeURIComponent(aspect)}/reviews?limit=4`);
      const quotes = $('.dossier__quotes', dossier);
      quotes.innerHTML = reviews.length
        ? reviews.map((review) => `
          <blockquote class="quote" data-tone="${esc(review.sentiment || 'neutral')}">
            <em>${esc(review.sentiment || 'unlabelled')}</em>
            ${esc(truncate(review.text, 240))}
          </blockquote>`).join('')
        : empty('No excerpts', 'No review rows were tagged with this aspect.');
    } catch (error) {
      $('.dossier__quotes', dossier).innerHTML = failure('Excerpts unavailable', error.message);
    }
  },
};

/* ---------- 04 · Topics ------------------------------------ */

const topics = {
  list: null,
  selected: null,

  async enter(view) {
    const chips = $('#topic-chips', view);
    if (this.list) return;

    chips.innerHTML = skeleton(1);
    try {
      this.list = (await api('/api/topics')).topics;
    } catch (error) {
      chips.innerHTML = failure('Topics unavailable', error.message);
      return;
    }

    chips.innerHTML = this.list.map((topic) => `
      <button type="button" class="chip" data-topic="${topic.topic}">
        <span class="chip__name">Topic ${topic.topic}</span>
        <span class="chip__count">${num(topic.count)} reviews</span>
      </button>`).join('');

    $$('[data-topic]', chips).forEach((button) => {
      button.addEventListener('click', () => this.select(view, Number(button.dataset.topic)));
    });

    this.select(view, this.list[0].topic);
  },

  async select(view, topic) {
    this.selected = topic;
    $$('[data-topic]', view).forEach((button) => {
      button.classList.toggle('is-active', Number(button.dataset.topic) === topic);
    });

    const out = $('#topic-out', view);
    out.innerHTML = skeleton(4);

    try {
      const data = await api(`/api/topics/${topic}/reviews?limit=8`);
      const meta = this.list.find((item) => item.topic === topic);
      out.innerHTML = `
        <section class="block">
          <div class="block__head">
            <h3 class="block__title">Topic ${topic}</h3>
            <p class="block__meta">${num(data.count)} reviews · ${pct(data.count / (this.total() || 1), 1)} of corpus</p>
          </div>
          ${meta?.sentiment ? ribbon('Polarity within this topic', meta.sentiment) : ''}
        </section>
        <section class="block">
          <div class="block__head"><h3 class="block__title">Representative reviews</h3></div>
          <div class="results">${data.reviews.map((review, index) => resultRow({
            rank: index + 1,
            title: review.title,
            review: review.text,
            sentiment: review.sentiment,
          }, { showScore: false, showIndex: false })).join('')}</div>
        </section>`;
      animateIn(out);
    } catch (error) {
      out.innerHTML = failure('Topic could not be opened', error.message);
    }
  },

  total() {
    return (this.list ?? []).reduce((sum, topic) => sum + topic.count, 0);
  },
};

/* ---------- 05 · Retrieval --------------------------------- */

const retrieval = {
  enter(view) {
    const form = $('#search-form', view);
    if (form.dataset.bound) return;
    form.dataset.bound = '1';

    const input = $('#search-query', view);
    const out = $('#search-out', view);

    $$('[data-query]', view).forEach((button) => {
      button.addEventListener('click', () => {
        input.value = button.dataset.query;
        form.requestSubmit();
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = input.value.trim();
      if (!query) {
        out.innerHTML = empty('Nothing to search for', 'Describe the reviews you want to surface.');
        return;
      }

      const button = $('button[type="submit"]', form);
      button.disabled = true;
      out.innerHTML = skeleton(5);

      try {
        const data = await api('/api/search', {
          body: { query, top_k: Number($('#search-k', view).value) },
        });
        out.innerHTML = data.results.length ? `
          <section class="block">
            <div class="block__head">
              <h3 class="block__title">${data.results.length} matches for “${esc(query)}”</h3>
              <p class="block__meta">${data.mode} ranking</p>
            </div>
            <div class="results">${data.results.map((item) => resultRow(item)).join('')}</div>
            ${data.mode === 'lexical' ? `<p class="notice">Ranked lexically by weighted term overlap.
              Install <b>sentence-transformers</b> to encode queries into the same embedding space
              as the corpus and search by meaning.</p>` : ''}
          </section>`
          : empty('No matches', 'Nothing in the indexed corpus came close. Try describing the sentiment differently.');
        animateIn(out);
      } catch (error) {
        out.innerHTML = failure('Search failed', error.message);
      } finally {
        button.disabled = false;
      }
    });
  },
};

/* ---------- 06 · Neighbours -------------------------------- */

const neighbours = {
  enter(view) {
    const form = $('#neighbour-form', view);
    if (form.dataset.bound) return;
    form.dataset.bound = '1';

    const field = $('#neighbour-index', view);
    const source = $('#neighbour-source', view);
    const out = $('#neighbour-out', view);

    const showSource = (record) => {
      source.innerHTML = `
        <article class="source">
          <p class="source__label">Review ${num(record.index)} of ${num(record.corpus ?? 0)}</p>
          ${record.title ? `<h3 class="source__title">${esc(titleCase(record.title))}</h3>` : ''}
          <p class="source__text">${esc(truncate(record.review, 700))}</p>
          <div class="result__meta">
            ${record.sentiment ? `<span class="dot" data-tone="${esc(record.sentiment)}">${esc(record.sentiment)}</span>` : ''}
            ${record.score != null ? `<span>VADER ${fixed(record.score, 4)}</span>` : ''}
            ${record.topic != null ? `<span>Topic ${num(record.topic)}</span>` : ''}
          </div>
        </article>`;
      animateIn(source);
    };

    $('#neighbour-random', view).addEventListener('click', async () => {
      try {
        const record = await api('/api/reviews/random/pick');
        field.value = record.index;
        showSource(record);
        out.innerHTML = '';
      } catch (error) {
        source.innerHTML = failure('Could not draw a review', error.message);
      }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const index = Number(field.value);
      if (!Number.isInteger(index) || index < 0) {
        out.innerHTML = empty('Invalid index', 'Enter a whole number, or press Random.');
        return;
      }

      const button = $('button[type="submit"]', form);
      button.disabled = true;
      out.innerHTML = skeleton(5);

      try {
        const data = await api('/api/similar', {
          body: { index, top_k: Number($('#neighbour-k', view).value) },
        });
        if (data.source) showSource({ ...data.source, corpus: data.source.corpus });
        out.innerHTML = `
          <section class="block">
            <div class="block__head">
              <h3 class="block__title">Nearest neighbours</h3>
              <p class="block__meta">cosine distance in embedding space</p>
            </div>
            <div class="results">${data.results.map((item) => resultRow(item)).join('')}</div>
          </section>`;
        animateIn(out);
      } catch (error) {
        out.innerHTML = failure('Lookup failed', error.message);
      } finally {
        button.disabled = false;
      }
    });

    // Open on a real review rather than an empty frame.
    api('/api/reviews/0').then(showSource).catch(() => {
      source.innerHTML = empty('Corpus not loaded', 'The embedding index could not be opened.');
    });
  },
};

/* ---------- 07 · Condense ---------------------------------- */

const condense = {
  enter(view) {
    const form = $('#condense-form', view);
    if (form.dataset.bound) return;
    form.dataset.bound = '1';

    const field = $('#condense-text', view);
    const out = $('#condense-out', view);

    $('#condense-sample', view).addEventListener('click', async () => {
      try {
        let record = await api('/api/reviews/random/pick');
        // Prefer something long enough to be worth condensing.
        for (let attempt = 0; attempt < 6 && record.review.split(' ').length < 90; attempt += 1) {
          record = await api('/api/reviews/random/pick');
        }
        field.value = record.review;
        field.focus();
      } catch { /* the textarea simply stays as it was */ }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = field.value.trim();
      if (!text) {
        out.innerHTML = empty('Nothing to condense', 'Paste a review, or load one from the corpus.');
        return;
      }

      const button = $('button[type="submit"]', form);
      button.disabled = true;
      out.innerHTML = skeleton(4);

      try {
        const data = await api('/api/summarize', {
          body: { text, sentences: Number($('#condense-n', view).value) },
        });
        out.innerHTML = `
          <div class="statrow">
            <div class="stat"><span class="stat__label">Original</span>
              <span class="stat__value">${num(data.original_words)}</span>
              <span class="stat__note">words · ${num(data.original_sentences)} sentences</span></div>
            <div class="stat"><span class="stat__label">Summary</span>
              <span class="stat__value">${num(data.summary_words)}</span>
              <span class="stat__note">words · ${num(data.summary_sentences)} sentences</span></div>
            <div class="stat stat--signal"><span class="stat__label">Compression</span>
              <span class="stat__value">${pct(data.compression, 0)}</span>
              <span class="stat__note">${esc(data.engine)}</span></div>
          </div>
          <div class="compare">
            <div class="compare__pane">
              <p class="compare__label">As written</p>
              <p class="compare__text">${esc(data.original)}</p>
            </div>
            <div class="compare__pane">
              <p class="compare__label">The gist</p>
              <p class="compare__text">${esc(data.summary)}</p>
            </div>
          </div>`;
        animateIn(out);
      } catch (error) {
        out.innerHTML = failure('Summarisation failed', error.message);
      } finally {
        button.disabled = false;
      }
    });
  },
};

/* ---------- 08 · Atlas ------------------------------------- */

const atlas = {
  loaded: false,
  async enter(view) {
    if (this.loaded) return;
    this.loaded = true;

    const grid = $('#atlas', view);
    grid.innerHTML = skeleton(3);

    let plates;
    try {
      plates = (await api('/api/overview')).plates;
    } catch (error) {
      grid.innerHTML = failure('Plates unavailable', error.message);
      return;
    }

    if (!plates.length) {
      grid.innerHTML = empty('No plates rendered', 'The visuals directory is empty — run the analytics notebook to generate them.');
      return;
    }

    grid.innerHTML = plates.map((plate, index) => `
      <button type="button" class="plate" data-src="${esc(plate.src)}" data-title="${esc(plate.title)}">
        <span class="plate__figure"><img src="${esc(plate.src)}" alt="${esc(plate.title)}" loading="lazy"></span>
        <span class="plate__body">
          <span class="plate__no">Plate ${String(index + 1).padStart(2, '0')}</span>
          <span class="plate__title">${esc(plate.title)}</span>
          <span class="plate__cap">${esc(plate.caption)}</span>
        </span>
      </button>`).join('');

    const box = $('#lightbox');
    $$('.plate', grid).forEach((plate) => {
      plate.addEventListener('click', () => {
        $('#lightbox-img').src = plate.dataset.src;
        $('#lightbox-img').alt = plate.dataset.title;
        $('#lightbox-cap').textContent = plate.dataset.title;
        box.showModal();
      });
    });
  },
};

/* ---------- lightbox & drawer ------------------------------ */

function wireChrome() {
  const box = $('#lightbox');
  $('.lightbox__close', box).addEventListener('click', () => box.close());
  box.addEventListener('click', (event) => { if (event.target === box) box.close(); });

  const rail = $('#rail');
  const scrim = $('.scrim');
  const toggle = $('[data-drawer]');

  const setDrawer = (open) => {
    rail.classList.toggle('is-open', open);
    scrim.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
  };

  toggle.addEventListener('click', () => setDrawer(!rail.classList.contains('is-open')));
  scrim.addEventListener('click', () => setDrawer(false));
  rail.addEventListener('click', (event) => {
    if (event.target.closest('.nav__item')) setDrawer(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setDrawer(false);
  });

  // Steppers drive their paired <output>.
  $$('[data-stepper]').forEach((stepper) => {
    const output = $(`#${stepper.dataset.stepper}`);
    const bounds = output.id === 'condense-n' ? [1, 8] : [1, 25];
    $$('[data-step]', stepper).forEach((button) => {
      button.addEventListener('click', () => {
        const next = clamp(Number(output.value ?? output.textContent) + Number(button.dataset.step), ...bounds);
        output.value = next;
        output.textContent = next;
      });
    });
  });
}

async function wireStatus() {
  const list = $('#engines');
  try {
    const data = await api('/api/status');
    const state = (value, live) => (value === live ? 'live' : 'fallback');
    const rows = [
      ['sentiment', data.engines.sentiment, state(data.engines.sentiment, 'vader+rules')],
      ['retrieval', data.engines.search, state(data.engines.search, 'semantic')],
      ['similarity', data.engines.similarity, data.engines.similarity === 'embeddings' ? 'live' : 'down'],
      ['summariser', data.engines.summarizer, data.engines.summarizer.includes('sumy') ? 'live' : 'fallback'],
    ];
    list.innerHTML = rows.map(([name, value, dot]) =>
      `<li><span class="engines__dot" data-state="${dot}"></span><span>${esc(name)}: ${esc(value)}</span></li>`
    ).join('');
  } catch {
    list.innerHTML = '<li><span class="engines__dot" data-state="down"></span><span>API offline</span></li>';
  }
}

/* ---------- router ----------------------------------------- */

const VIEWS = {
  corpus, verdict, aspects, topics, retrieval, neighbours, condense, atlas,
};

function route() {
  const name = (location.hash.replace('#/', '') || 'corpus');
  const target = VIEWS[name] ? name : 'corpus';

  $$('.view').forEach((view) => {
    const active = view.dataset.view === target;
    view.hidden = !active;
    view.toggleAttribute('data-active', active);
    if (active) choreograph(view);
  });

  $$('.nav__item').forEach((link) => {
    link.classList.toggle('is-active', link.dataset.route === target);
    link.setAttribute('aria-current', link.dataset.route === target ? 'page' : 'false');
  });

  const view = $(`.view[data-view="${target}"]`);
  document.title = `${titleCase(target)} — The Review Room`;
  VIEWS[target].enter(view);
  window.scrollTo({ top: 0, behavior: REDUCED ? 'auto' : 'smooth' });
}

window.addEventListener('hashchange', route);

wireChrome();
wireStatus();
route();
document.documentElement.dataset.boot = 'ready';
