/* Local UI only: zero third-party runtime scripts; XSS-safe DOM rendering via textContent. */
(() => {
  const el = (id) => document.getElementById(id);
  const make = (tag, text = '', cls = '') => {
    const n = document.createElement(tag);
    if (text !== '') n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  const fmtNum = (value, maxFrac = 2) =>
    new Intl.NumberFormat('vi-VN', { maximumFractionDigits: maxFrac }).format(value);

  const scopes = { separate: 'Báo cáo riêng', consolidated: 'Hợp nhất', unknown: 'Chưa rõ phạm vi' };
  const qualities = {
    consistent: ['Khớp công thức', 'ok'],
    verified: ['Đã xác minh tay', 'ok'],
    needs_review: ['Cần kiểm tra', 'err'],
    ocr_unverified: ['OCR chưa xác minh', 'warn'],
    unverified: ['Chưa xác minh', 'neutral'],
  };

  let docs = [];
  let facts = [];
  let currentAnalysis = null;
  let colabReady = false;
  let viewerState = { docId: null, page: 1, maxPages: 1, chunkId: null, zoom: 100 };
  let graphState = { nodes: [], edges: [], positions: {}, panX: 0, panY: 0, scale: 1, dragging: false, lastX: 0, lastY: 0, selected: null };
  const chatHistoryByDoc = new Map();

  async function request(url, options) {
    const r = await fetch(url, options);
    const d = await r.json();
    if (!r.ok) {
      throw Error(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail));
    }
    return d;
  }

  // ---- Tabs ----
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      el(`tab-${btn.dataset.tab}`).classList.add('active');
      if (btn.dataset.tab === 'graph' && el('docs').value && !graphState.nodes.length) {
        loadGraph();
      }
    });
  });

  // ---- PDF Split-Pane Viewer ----
  function openSourcePage(docId, page, chunkId = null, bbox = null) {
    const doc = docs.find((d) => d.id === docId);
    const total = currentAnalysis?.metadata?.pages || 999;
    viewerState.docId = docId;
    viewerState.page = Math.max(1, Number(page) || 1);
    viewerState.maxPages = total;
    viewerState.chunkId = chunkId;

    const img = el('sourceImage');
    const isPdf = !doc || /\.pdf$/i.test(doc.name || '');
    if (!isPdf && doc) {
      img.hidden = true;
      el('viewerTitle').textContent = 'Tài liệu dạng văn bản (không phải PDF). Xem đoạn trích ở khung giữa.';
      el('pageIndicator').textContent = `Trang ${viewerState.page}`;
      return;
    }

    const query = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : '';
    el('viewerTitle').textContent = `Trang PDF ${viewerState.page} · ${
      bbox ? 'đang tô cam vùng bằng chứng' : 'xem trang gốc'
    }`;
    el('pageIndicator').textContent = currentAnalysis?.metadata?.pages
      ? `Trang ${viewerState.page}/${currentAnalysis.metadata.pages}`
      : `Trang ${viewerState.page}`;
    img.hidden = false;
    img.style.width = `${viewerState.zoom}%`;
    img.onerror = () => {
      img.hidden = true;
      el('viewerTitle').textContent = 'Không tải được ảnh trang PDF (tài liệu văn bản hoặc số trang ngoài phạm vi).';
    };
    img.src = `/documents/${encodeURIComponent(docId)}/pages/${viewerState.page}${query}`;
  }

  el('prevPage').onclick = () => {
    if (viewerState.docId && viewerState.page > 1) {
      openSourcePage(viewerState.docId, viewerState.page - 1, null, null);
    }
  };
  el('nextPage').onclick = () => {
    if (viewerState.docId && viewerState.page < viewerState.maxPages) {
      openSourcePage(viewerState.docId, viewerState.page + 1, null, null);
    }
  };
  el('zoomIn').onclick = () => {
    viewerState.zoom = Math.min(220, viewerState.zoom + 20);
    el('zoomLabel').textContent = `${viewerState.zoom}%`;
    el('sourceImage').style.width = `${viewerState.zoom}%`;
  };
  el('zoomOut').onclick = () => {
    viewerState.zoom = Math.max(60, viewerState.zoom - 20);
    el('zoomLabel').textContent = `${viewerState.zoom}%`;
    el('sourceImage').style.width = `${viewerState.zoom}%`;
  };

  function sourceButton(docId, page, chunkId, bbox, label = 'Xem nguồn') {
    const b = make('button', label, 'secondary sm');
    b.type = 'button';
    b.onclick = () => openSourcePage(docId, page, chunkId, bbox);
    return b;
  }

  // ---- Health & Jobs ----
  async function refreshHealth() {
    try {
      const h = await request('/health');
      el('engine').textContent = `SQLite FTS5 · ${h.chunks} chunk · ${h.nodes} node`;
      el('engine').className = 'pill ok';
    } catch (e) {
      el('engine').textContent = e.message;
      el('engine').className = 'pill warn';
    }

    try {
      const c = await request('/health/colab');
      colabReady = Boolean(c.ready);
      const pill = el('colabPill');
      if (c.ready) {
        pill.textContent = `Colab GPU: ${c.gpu?.name || 'Sẵn sàng'}`;
        pill.className = 'pill ok';
      } else if (c.configured) {
        pill.textContent = 'Colab GPU: Mất kết nối';
        pill.className = 'pill warn';
      } else {
        pill.textContent = 'Colab GPU: Chưa cấu hình (Chế độ cục bộ)';
        pill.className = 'pill';
      }
      el('colabPill').title = c.message || '';
    } catch {
      el('colabPill').textContent = 'Colab GPU: Không kiểm tra được';
      el('colabPill').className = 'pill warn';
    }

    try {
      const jobs = await request('/jobs');
      const box = el('jobsList');
      box.replaceChildren();
      const labels = { complete: 'Hoàn tất', running: 'Đang đọc', queued: 'Đang chờ', failed: 'Thất bại', interrupted: 'Gián đoạn' };
      if (!jobs.length) box.append(make('small', 'Chưa có lịch sử nạp nền.'));
      for (const job of jobs.slice(0, 6)) {
        const det = make('details');
        det.append(make('summary', `${labels[job.status] || job.status} · ${job.name}`));
        det.append(make('small', new Date(job.updated_at).toLocaleString('vi-VN')));
        det.append(make('p', job.error || `${job.completed_pages || 0}/${job.total_pages || '?'} trang`));
        box.append(det);
      }
    } catch {
      // Ignore background jobs error on init
    }
  }
  el('refreshHealth').onclick = refreshHealth;

  // ---- Document list & selection ----
  function renderDocList(selectedId) {
    const list = el('docList');
    list.replaceChildren();
    el('docs').replaceChildren();
    el('cmpLeft').replaceChildren();
    el('cmpRight').replaceChildren();

    if (!docs.length) {
      list.append(make('small', 'Chưa có báo cáo. Hãy upload PDF/TXT ở trên.'));
      return;
    }

    for (const d of docs) {
      const label = `${d.ticker || 'BCTC'} ${d.year || ''} · ${d.name}`;
      el('docs').add(new Option(label, d.id));
      el('cmpLeft').add(new Option(label, d.id));
      el('cmpRight').add(new Option(label, d.id));

      const card = make('div', '', `doc-item ${d.id === selectedId ? 'active' : ''}`);
      card.onclick = () => selectDocument(d.id);
      card.append(make('div', d.name || d.id, 'doc-item-title'));

      const meta = make('div', '', 'doc-item-meta');
      meta.append(make('span', `${d.ticker || '-'} · ${d.year || '-'} · ${scopes[d.scope] || d.scope}`));
      const delBtn = make('button', 'Xóa', 'danger sm');
      delBtn.type = 'button';
      delBtn.onclick = async (ev) => {
        ev.stopPropagation();
        await request(`/documents/${encodeURIComponent(d.id)}`, { method: 'DELETE' });
        chatHistoryByDoc.delete(d.id);
        await refreshDocs();
        await refreshHealth();
      };
      meta.append(delBtn);
      card.append(meta);
      list.append(card);
    }

    if (selectedId) el('docs').value = selectedId;
    if (docs.length >= 2) {
      el('cmpLeft').value = docs[0].id;
      el('cmpRight').value = docs[1].id;
    }
  }

  async function selectDocument(id) {
    if (!id) return;
    el('docs').value = id;
    renderDocList(id);
    graphState = { ...graphState, nodes: [], edges: [], selected: null };
    el('answer').textContent = '';
    el('mode').textContent = '';
    el('warnings').replaceChildren();
    el('trace').hidden = true;
    el('evidence').replaceChildren();
    renderChatHistory(id);
    await showReport();
  }

  async function refreshDocs(preferredId = null) {
    docs = await request('/documents');
    const selected = preferredId && docs.some((d) => d.id === preferredId)
      ? preferredId
      : docs.length
      ? docs.at(-1).id
      : '';
    renderDocList(selected);
    if (selected) {
      await showReport();
    } else {
      facts = [];
      currentAnalysis = null;
      el('reportInfo').textContent = '';
      el('facts').replaceChildren();
      el('kpiGrid').replaceChildren();
    }
  }

  // ---- Report Analysis, Overview, Charts & Facts Table ----
  function renderSvgChart(factList) {
    const svg = el('kpiChart');
    svg.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg';
    const picked = factList
      .filter((f) => ['10', '60', '280', '400'].includes(f.metric_code) || f.statement === 'chart')
      .slice(0, 8);
    if (!picked.length) {
      const txt = document.createElementNS(ns, 'text');
      txt.setAttribute('x', '20');
      txt.setAttribute('y', '110');
      txt.setAttribute('fill', '#64748b');
      txt.textContent = 'Chưa đủ số liệu chuẩn hoặc biểu đồ để vẽ.';
      svg.append(txt);
      return;
    }
    const maxVal = Math.max(...picked.map((f) => Math.abs(f.value)), 1);
    const barW = Math.min(54, Math.floor(560 / picked.length) - 12);
    picked.forEach((f, idx) => {
      const x = 44 + idx * (barW + 14);
      const h = Math.max(8, (Math.abs(f.value) / maxVal) * 145);
      const y = 175 - h;
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(barW));
      rect.setAttribute('height', String(h));
      rect.setAttribute('rx', '4');
      rect.setAttribute('fill', f.value < 0 ? '#dc2626' : '#0b6e75');
      const title = document.createElementNS(ns, 'title');
      title.textContent = `${f.metric} (${f.period_label || f.period}): ${fmtNum(f.value)} ${f.unit || ''}`;
      rect.append(title);
      svg.append(rect);

      const lbl = document.createElementNS(ns, 'text');
      lbl.setAttribute('x', String(x + barW / 2));
      lbl.setAttribute('y', '194');
      lbl.setAttribute('text-anchor', 'middle');
      lbl.setAttribute('font-size', '10');
      lbl.setAttribute('fill', '#334155');
      lbl.textContent = `${(f.metric_code || f.metric).slice(0, 8)} ${String(f.period_label || f.period || '').slice(-4)}`;
      svg.append(lbl);
    });
  }

  function renderFacts() {
    const q = el('filter').value.toLocaleLowerCase();
    const stmt = el('stmtFilter').value;
    const qual = el('qualityFilter').value;
    const tbody = el('facts');
    tbody.replaceChildren();
    const docId = el('docs').value;

    for (const f of facts) {
      if (stmt && f.statement !== stmt) continue;
      if (qual && f.quality !== qual) continue;
      if (q && !`${f.metric} ${f.metric_code || ''} ${f.period_label || f.period || ''}`.toLocaleLowerCase().includes(q)) {
        continue;
      }
      const tr = make('tr');
      tr.append(
        make('td', f.metric_code ? `${f.metric} [${f.metric_code}]` : f.metric),
        make('td', f.period_label || String(f.period || '?')),
        make('td', `${fmtNum(f.value, 4)} ${f.unit || ''}`),
      );
      const qTd = make('td');
      const [qLabel, qCls] = qualities[f.quality] || [f.quality, 'neutral'];
      qTd.append(make('span', qLabel, `badge ${qCls}`));
      tr.append(qTd);

      const actTd = make('td');
      const actRow = make('div', '', 'row');
      actRow.style.gap = '4px';
      actRow.append(sourceButton(docId, f.page, f.chunk_id, f.bbox, `Tr.${f.page}`));

      const verifyBtn = make('button', 'Xác minh', 'secondary sm');
      verifyBtn.type = 'button';
      verifyBtn.onclick = async () => {
        await request(`/facts/${encodeURIComponent(f.id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ quality: 'verified' }),
        });
        await showReport();
      };

      const editBtn = make('button', 'Sửa', 'secondary sm');
      editBtn.type = 'button';
      editBtn.onclick = async () => {
        const nextStr = window.prompt(`Nhập giá trị số mới cho "${f.metric}" (${f.period_label || f.period}):`, String(f.value));
        if (nextStr === null) return;
        const parsed = Number(nextStr.replace(/\s+/g, '').replace(',', '.'));
        if (!Number.isFinite(parsed)) {
          window.alert('Giá trị không hợp lệ.');
          return;
        }
        await request(`/facts/${encodeURIComponent(f.id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            value: parsed,
            raw_value: nextStr.trim(),
            quality: 'verified',
            recognition_note: 'Người dùng đã đối chiếu và chỉnh sửa thủ công.',
          }),
        });
        await showReport();
      };
      actRow.append(verifyBtn, editBtn);
      actTd.append(actRow);
      tr.append(actTd);
      tbody.append(tr);
    }
  }
  el('filter').oninput = renderFacts;
  el('stmtFilter').onchange = renderFacts;
  el('qualityFilter').onchange = renderFacts;

  async function showReport() {
    const id = el('docs').value;
    if (!id) return;
    const d = await request(`/documents/${encodeURIComponent(id)}/analysis`);
    if (el('docs').value !== id) return;
    currentAnalysis = d;
    facts = d.facts || [];
    const m = d.metadata || {};

    el('reportInfo').textContent = `${scopes[m.scope] || 'Báo cáo'} · ${m.pages || '?'} trang · ${m.ocr_pages || 0} OCR · ${facts.length} fact`;
    el('original').href = `/documents/${encodeURIComponent(id)}/file`;
    el('exportLink').href = `/documents/${encodeURIComponent(id)}/export`;
    el('summaryWarning').textContent = d.summary?.warning || '';

    const grid = el('kpiGrid');
    grid.replaceChildren();
    for (const item of d.summary?.cards || []) {
      const card = make('div', '', 'kpi-card');
      card.append(make('div', item.metric, 'doc-item-title'), make('small', item.period || ''));
      card.append(
        make('div', item.value === null ? 'Cần kiểm tra' : `${fmtNum(item.value)} ${item.unit || ''}`, 'kpi-val')
      );
      if (item.comparison) {
        card.append(make('small', `${fmtNum(Number(item.comparison.value))}% so với ${item.comparison.period}`));
      }
      if (item.warning) card.append(make('div', item.warning, 'warn-box'));
      const f = facts.find((v) => v.id === item.fact_ids[0]);
      if (f) {
        const wrap = make('div', '', '');
        wrap.style.marginTop = '6px';
        wrap.append(sourceButton(id, f.page, f.chunk_id, f.bbox, `Nguồn tr.${f.page}`));
        card.append(wrap);
      }
      grid.append(card);
    }
    if (!(d.summary?.cards || []).length) {
      grid.append(make('small', 'Chưa có chỉ tiêu chuẩn đủ dữ liệu cho thẻ KPI.'));
    }

    const rBox = el('ratiosList');
    rBox.replaceChildren();
    for (const ratio of d.summary?.ratios || []) {
      const det = make('details');
      det.append(make('summary', `${ratio.metric} (${ratio.period}): ${fmtNum(Number(ratio.value))}%`));
      det.append(make('pre', ratio.formula, 'status-pre'));
      rBox.append(det);
    }

    const c = m.checks || [];
    const labels = { pass: 'KHỚP', fail: 'LỆCH', incomplete: 'THIẾU' };
    el('quality').textContent =
      `${c.filter((v) => v.status === 'pass').length} khớp · ` +
      `${c.filter((v) => v.status === 'fail').length} lệch · ` +
      `${c.filter((v) => v.status === 'incomplete').length} thiếu.\n\n` +
      c
        .map(
          (v) =>
            `${labels[v.status] || v.status} · ${v.statement} · ${v.period} → ${v.formula}` +
            (v.difference !== undefined ? ` (lệch ${v.difference})` : '') +
            (v.missing_codes?.length ? ` [thiếu/trùng: ${v.missing_codes.join(', ')}]` : '')
        )
        .join('\n') +
      ((m.warnings || []).length ? `\n\nCảnh báo:\n${m.warnings.join('\n')}` : '');

    for (const side of ['left', 'right']) {
      el(side).replaceChildren();
      for (const f of facts) {
        el(side).add(new Option(`${f.metric} | ${f.period_label || f.period} | ${fmtNum(f.value)} ${f.unit || ''}`, f.id));
      }
    }

    renderSvgChart(facts);
    renderFacts();
    if (m.pages && (!viewerState.docId || viewerState.docId !== id)) {
      openSourcePage(id, 1, null, null);
    }
  }

  // ---- Upload & Polling ----
  async function pollJob(jobId) {
    try {
      const j = await request(`/jobs/${encodeURIComponent(jobId)}`);
      if (j.status === 'complete') {
        el('progress').hidden = true;
        el('uploadButton').disabled = false;
        const d = j.result;
        el('status').textContent = `Đã đọc ${d.pages} trang, ${d.statement_rows || 0} dòng bảng, ${d.facts} fact.`;
        await refreshDocs(d.document_id);
        await refreshHealth();
        return;
      }
      if (['failed', 'interrupted'].includes(j.status)) throw Error(j.error);
      el('status').textContent = j.total_pages ? `Đang đọc trang ${j.completed_pages}/${j.total_pages}…` : 'Đang xử lý…';
      if (j.total_pages) {
        el('progress').max = j.total_pages;
        el('progress').value = j.completed_pages;
      }
      setTimeout(() => pollJob(jobId), 1200);
    } catch (e) {
      el('status').textContent = e.message;
      el('uploadButton').disabled = false;
      el('progress').hidden = true;
      refreshHealth();
    }
  }

  el('upload').onsubmit = async (e) => {
    e.preventDefault();
    el('uploadButton').disabled = true;
    el('status').textContent = 'Đang tải lên…';
    el('progress').hidden = false;
    el('progress').removeAttribute('value');
    try {
      const f = new FormData(e.target);
      for (const k of ['ticker', 'year']) if (!f.get(k)) f.delete(k);
      f.set('background', 'true');
      const d = await request('/ingest', { method: 'POST', body: f });
      pollJob(d.job_id);
    } catch (err) {
      el('status').textContent = err.message;
      el('uploadButton').disabled = false;
      el('progress').hidden = true;
    }
  };

  // ---- QA Ask & Clickable Citations ----
  document.querySelectorAll('[data-q]').forEach((b) => {
    b.onclick = () => {
      el('question').value = b.dataset.q;
    };
  });

  function renderRichAnswer(container, text, evidence, docId) {
    container.replaceChildren();
    const parts = text.split(/(\[\d+\])/g);
    for (const part of parts) {
      const m = part.match(/^\[(\d+)\]$/);
      if (m) {
        const idx = Number(m[1]) - 1;
        const ev = evidence[idx];
        if (ev) {
          const btn = make('button', part, 'cite-btn');
          btn.type = 'button';
          btn.title = `Mở trang ${ev.page}`;
          btn.onclick = () => openSourcePage(ev.document_id || docId, ev.page, ev.chunk_id, ev.bbox);
          container.append(btn);
          continue;
        }
      }
      container.append(document.createTextNode(part));
    }
  }

  function renderChatHistory(docId) {
    const feed = el('chatHistory');
    feed.replaceChildren();
    const items = chatHistoryByDoc.get(docId) || [];
    for (const item of items.slice().reverse()) {
      const card = make('div', '', 'chat-msg');
      card.append(make('div', `Hỏi: ${item.question}`, 'chat-q'));
      const ans = make('div', '', 'answer-text');
      renderRichAnswer(ans, item.response.answer, item.response.evidence || [], docId);
      card.append(ans);
      feed.append(card);
    }
  }

  el('ask').onclick = async () => {
    const reportId = el('docs').value;
    if (!reportId) {
      el('answer').textContent = 'Hãy nạp hoặc chọn một báo cáo.';
      return;
    }
    const qText = el('question').value.trim();
    if (!qText) return;

    el('ask').disabled = true;
    el('answer').textContent = 'Đang đối chiếu dữ kiện và bằng chứng…';
    el('mode').textContent = '';
    el('trace').hidden = true;
    el('evidence').replaceChildren();
    el('warnings').replaceChildren();

    try {
      const d = await request('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: qText,
          document_id: reportId,
          use_llm: Boolean(el('useColab').checked),
        }),
      });
      if (el('docs').value !== reportId) return;

      renderRichAnswer(el('answer'), d.answer, d.evidence || [], reportId);
      const statusMap = {
        answered: d.mode === 'llm' ? 'Colab GPU có trích dẫn' : 'Đã trả lời từ dữ kiện',
        evidence_only: 'Chỉ có đoạn liên quan',
        needs_clarification: 'Cần làm rõ câu hỏi',
        insufficient_evidence: 'Chưa đủ bằng chứng',
      };
      el('mode').textContent = statusMap[d.answer_status] || d.mode;
      el('mode').className = `badge ${d.answer_status === 'answered' ? 'ok' : 'warn'}`;

      el('trace').hidden = !(d.steps || []).length;
      el('traceText').textContent = `${(d.steps || []).map((s, i) => `${i + 1}. ${s}`).join('\n')}\nMã xử lý: ${d.reason_code || d.mode}`;

      for (const w of d.warnings || []) {
        el('warnings').append(make('div', w, 'warn-box'));
      }

      (d.evidence || []).forEach((v, i) => {
        const box = make('div', '', 'evidence-item');
        const head = make('div', '', 'row');
        head.style.justifyContent = 'space-between';
        head.append(make('strong', `[${i + 1}] Trang ${v.page} · ${v.extraction_method || 'text'}`));
        if (v.document_id) {
          head.append(sourceButton(v.document_id, v.page, v.chunk_id, v.bbox, 'Xem trên PDF'));
        }
        box.append(head, make('pre', v.text));
        el('evidence').append(box);
      });

      if ((d.evidence || []).length && d.evidence[0].document_id) {
        openSourcePage(d.evidence[0].document_id, d.evidence[0].page, d.evidence[0].chunk_id, d.evidence[0].bbox);
      }

      const hist = chatHistoryByDoc.get(reportId) || [];
      hist.push({ question: qText, response: d });
      chatHistoryByDoc.set(reportId, hist.slice(-15));
      renderChatHistory(reportId);
    } catch (err) {
      el('answer').textContent = err.message;
    } finally {
      el('ask').disabled = false;
    }
  };

  // ---- Calculator & Report Comparison ----
  el('calc').onclick = async () => {
    try {
      const d = await request('/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operation: el('operation').value,
          left_fact_id: el('left').value,
          right_fact_id: el('right').value,
        }),
      });
      el('result').textContent = `${d.formula} = ${d.value} ${d.unit}`;
    } catch (err) {
      el('result').textContent = err.message;
    }
  };

  el('cmpBtn').onclick = async () => {
    const [statement, metricCode] = el('cmpMetric').value.split(':');
    try {
      const d = await request('/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          left_document_id: el('cmpLeft').value,
          right_document_id: el('cmpRight').value,
          statement,
          metric_code: metricCode,
          operation: el('cmpOp').value,
        }),
      });
      el('cmpResult').textContent = `${d.formula} = ${fmtNum(Number(d.value))} ${d.unit}`;
    } catch (err) {
      el('cmpResult').textContent = err.message;
    }
  };

  // ---- Interactive Canvas Knowledge Graph ----
  function drawGraph() {
    const canvas = el('graphCanvas');
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(400, rect.width || 600);
    canvas.height = 420;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!graphState.nodes.length) return;

    ctx.save();
    ctx.translate(graphState.panX, graphState.panY);
    ctx.scale(graphState.scale, graphState.scale);

    const colors = {
      Report: '#0b6e75',
      Company: '#e66a15',
      Period: '#0284c7',
      Metric: '#7c3aed',
      Fact: '#16a34a',
      Chunk: '#64748b',
    };

    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    for (const e of graphState.edges) {
      const s = graphState.positions[e.source];
      const t = graphState.positions[e.target];
      if (!s || !t) continue;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }

    for (const n of graphState.nodes) {
      const p = graphState.positions[n.id];
      if (!p) continue;
      const r = n.kind === 'Report' ? 13 : n.kind === 'Metric' ? 9 : 6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = colors[n.kind] || '#475569';
      ctx.fill();
      if (graphState.selected === n.id) {
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
      if (n.kind === 'Report' || n.kind === 'Metric' || n.kind === 'Company') {
        ctx.fillStyle = '#0f2537';
        ctx.font = '11px system-ui';
        ctx.fillText((n.name || n.ticker || n.id).slice(0, 22), p.x + r + 4, p.y + 4);
      }
    }
    ctx.restore();
  }

  async function loadGraph() {
    const id = el('docs').value;
    if (!id) return;
    const g = await request(`/documents/${encodeURIComponent(id)}/graph?limit=180`);
    const canvas = el('graphCanvas');
    const w = Math.max(520, canvas.getBoundingClientRect().width || 620);
    const h = 420;
    const layers = { Company: 0.1, Period: 0.1, Report: 0.28, Metric: 0.52, Fact: 0.72, Chunk: 0.9 };
    const buckets = {};
    for (const n of g.nodes) {
      const k = n.kind || 'Chunk';
      buckets[k] = buckets[k] || [];
      buckets[k].push(n);
    }
    const positions = {};
    for (const [kind, items] of Object.entries(buckets)) {
      const x = (layers[kind] || 0.5) * w;
      items.forEach((n, idx) => {
        const y = ((idx + 1) / (items.length + 1)) * (h - 40) + 20;
        positions[n.id] = { x, y };
      });
    }
    graphState = { ...graphState, nodes: g.nodes, edges: g.edges, positions, panX: 0, panY: 0, scale: 1, selected: null };
    el('graphInfo').textContent = `${g.nodes.length}/${g.total_nodes} node · ${g.edges.length} cạnh (kéo để di chuyển, bấm node để xem chi tiết).`;
    drawGraph();
  }

  el('loadGraphBtn').onclick = () => loadGraph().catch((e) => (el('graphInfo').textContent = e.message));

  el('graphCanvas').addEventListener('mousedown', (ev) => {
    graphState.dragging = true;
    graphState.lastX = ev.clientX;
    graphState.lastY = ev.clientY;
  });
  window.addEventListener('mouseup', () => {
    graphState.dragging = false;
  });
  window.addEventListener('mousemove', (ev) => {
    if (!graphState.dragging) return;
    graphState.panX += ev.clientX - graphState.lastX;
    graphState.panY += ev.clientY - graphState.lastY;
    graphState.lastX = ev.clientX;
    graphState.lastY = ev.clientY;
    drawGraph();
  });
  el('graphCanvas').addEventListener('click', (ev) => {
    const rect = el('graphCanvas').getBoundingClientRect();
    const cx = (ev.clientX - rect.left - graphState.panX) / graphState.scale;
    const cy = (ev.clientY - rect.top - graphState.panY) / graphState.scale;
    let best = null;
    let bestDist = 18;
    for (const n of graphState.nodes) {
      const p = graphState.positions[n.id];
      if (!p) continue;
      const dist = Math.hypot(p.x - cx, p.y - cy);
      if (dist < bestDist) {
        best = n;
        bestDist = dist;
      }
    }
    const detail = el('graphDetail');
    detail.replaceChildren();
    if (!best) return;
    graphState.selected = best.id;
    drawGraph();
    detail.append(make('strong', `${best.kind}: ${best.name || best.metric || best.ticker || best.id}`));
    if (best.raw_value) detail.append(make('p', `Giá trị: ${best.raw_value} ${best.unit || ''} (${best.period_label || ''})`));
    if (best.text) detail.append(make('pre', best.text.slice(0, 240), 'status-pre'));
    if (best.page && el('docs').value) {
      detail.append(sourceButton(el('docs').value, best.page, best.chunk_id || best.id, best.bbox, `Mở trang ${best.page}`));
    }
  });

  // Boot
  refreshHealth();
  refreshDocs().catch((e) => (el('status').textContent = e.message));
})();
