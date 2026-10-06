/**
 * debunk.js — Checa-AI frontend logic
 * ─────────────────────────────────────────────────────────────────
 * Gerencia o formulário de verificação de boatos:
 *  1. POST para /api/debunk
 *  2. Renderiza card de resultado (matched | abstained | erro)
 *  3. Chips de exemplos pré-preenchidos
 */

/* ── Contagem de caracteres ─────────────────────────────────── */
const textarea  = document.getElementById('claim-input');
const charCount = document.getElementById('char-count');

textarea.addEventListener('input', () => {
  charCount.textContent = textarea.value.length;
});

/* ── Preenche exemplos ──────────────────────────────────────── */
function fillExample(btn) {
  textarea.value = btn.textContent.trim();
  charCount.textContent = textarea.value.length;
  textarea.focus();
}

/* ── Renderização de resultado ──────────────────────────────── */
function renderResult(data) {
  const area = document.getElementById('result-area');
  area.classList.remove('hidden');

  const isMatched   = data.status === 'matched';
  const evidence    = data.evidence ?? {};
  const scoreLabel  = data.score > 0 ? `Similaridade: ${(data.score * 100).toFixed(1)}%` : '';

  const verdictIcon  = isMatched ? '✅' : '🔕';
  const verdictLabel = isMatched ? 'Checagem Encontrada' : 'Abstinência';
  const badgeClass   = isMatched ? 'matched' : 'abstained';

  const sources = (data.sources && data.sources.length)
    ? data.sources.map(s => ({
        titulo: s.title, dominio: s.domain, url: s.url, data_publicacao: s.published_at,
        cluster_label: s.cluster_label, cluster_review_status: s.cluster_review_status,
        cluster_confidence: s.cluster_confidence, score: s.score, rerank_score: s.rerank_score,
      }))
    : ((evidence.titulo || evidence.url) ? [evidence] : []);

  const evidenceBlock = sources.map((src, i) => `
      <div class="evidence-card">
        <p class="section-title">📋 Checagem de referência${sources.length > 1 ? ` ${i + 1}` : ''}</p>
        ${src.titulo
          ? `<p class="evidence-title">${escHtml(src.titulo)}</p>`
          : ''}
        <div class="evidence-meta">
          ${src.dominio
            ? `<span>🏢 ${escHtml(src.dominio)}</span>`
            : ''}
          ${src.data_publicacao
            ? `<span>📅 ${escHtml(src.data_publicacao)}</span>`
            : ''}
          ${src.cluster_label
            ? `<span class="cluster-chip" title="Cluster usado no re-ranking">${escHtml(src.cluster_label)}</span>`
            : ''}
          ${src.cluster_review_status
            ? `<span class="cluster-chip ${src.cluster_review_status === 'approved' ? 'approved' : 'pending'}">${escHtml(src.cluster_review_status)}</span>`
            : ''}
        </div>
        ${src.url
          ? `<a class="source-link"
                href="${escHtml(src.url)}"
                target="_blank"
                rel="noopener noreferrer">
               🔗 Leia a checagem completa
             </a>`
          : ''}
      </div>`).join('');

  area.innerHTML = `
    <div class="result-card">
      <div class="result-header">
        <span class="verdict-badge ${badgeClass}">
          ${verdictIcon} ${verdictLabel}
        </span>
        ${scoreLabel
          ? `<span class="score-badge">${scoreLabel}</span>`
          : ''}
      </div>
      <div class="result-body">
        <p class="section-title">💬 Contranarrativa gerada</p>
        <p class="counter-narrative">${escHtml(data.counter_narrative)}</p>
        ${evidenceBlock}
      </div>
    </div>`;
}

function renderError(msg) {
  const area = document.getElementById('result-area');
  area.classList.remove('hidden');
  area.innerHTML = `
    <div class="error-card">
      ❌ <strong>Erro:</strong> ${escHtml(msg)}
    </div>`;
}

function renderLoading() {
  const area = document.getElementById('result-area');
  area.classList.remove('hidden');
  area.innerHTML = `
    <div class="loading-card">
      <div class="loading-text">
        <div class="spinner"></div>
        Buscando checagens e gerando contranarrativa…
      </div>
    </div>`;
}

/* ── Escape HTML para evitar XSS ────────────────────────────── */
function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── Chamada principal ──────────────────────────────────────── */
async function verifyDebunk() {
  const message = textarea.value.trim();

  if (message.length < 10) {
    textarea.focus();
    textarea.style.borderColor = 'var(--accent-red)';
    setTimeout(() => { textarea.style.borderColor = ''; }, 1500);
    return;
  }

  const btn = document.getElementById('verify-btn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Verificando…';

  renderLoading();

  try {
    const res = await fetch('/api/debunk', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      renderError(err.detail ?? 'Erro desconhecido na API.');
      return;
    }

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    renderError(`Não foi possível conectar ao servidor. Verifique se o app está rodando.\n${err.message}`);
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Verificar mensagem';
  }
}

/* ── Submete com Enter (Ctrl+Enter / Cmd+Enter) ─────────────── */
textarea.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    verifyDebunk();
  }
});
