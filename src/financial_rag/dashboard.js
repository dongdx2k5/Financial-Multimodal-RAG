/* Local UI only: no third-party scripts or uploaded data sent to external services. */
(() => {
  const make = (tag, text='') => {const n=document.createElement(tag);n.textContent=text;return n;};
  const money = value => new Intl.NumberFormat('vi-VN',{maximumFractionDigits:2}).format(value);
  const panel=make('section');panel.id='overview';
  panel.append(make('h2','Tổng quan có bằng chứng'),make('small','Đang chọn báo cáo…'));
  document.querySelector('main > div').prepend(panel);
  const system=make('section');system.append(make('h2','Trạng thái xử lý'));
  const runtime=make('p','Đang kiểm tra Ollama…');
  const recent=make('div');const reload=make('button','Làm mới trạng thái');reload.className='secondary';
  system.append(runtime,reload,recent);document.querySelector('aside').append(system);
  const graphPanel=make('section');graphPanel.append(make('h2','Graph theo báo cáo'));
  const graphInfo=make('p','Chọn báo cáo để xem quan hệ chỉ tiêu → số liệu → trang nguồn.');
  const graphButton=make('button','Khám phá graph');graphButton.className='secondary';
  const graphView=make('div');graphPanel.append(graphInfo,graphButton,graphView);
  document.querySelector('main > div').append(graphPanel);
  let selected=null, current=null;
  function source(fact,id) {
    const button=make('button',`Nguồn · trang ${fact.page}`);button.className='secondary';
    button.onclick=()=>{
      document.getElementById('viewerTitle').textContent=`Trang PDF ${fact.page} · vùng bằng chứng`;
      const img=document.getElementById('sourceImage');
      img.onerror=()=>{document.getElementById('viewerTitle').textContent='Không tải được ảnh trang. Hãy mở tài liệu gốc.';};
      img.src=`/documents/${encodeURIComponent(id)}/pages/${fact.page}?chunk_id=${encodeURIComponent(fact.chunk_id)}`;
      document.getElementById('viewer').showModal();
    };return button;
  }
  async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`Không tải được dữ liệu (${r.status}).`);return r.json();}
  async function status(){
    reload.disabled=true;
    try{
      const health=await json('/health/ollama');
      runtime.textContent=`${health.model} · ${health.busy?'đang xử lý':health.ready?'sẵn sàng':'chưa sẵn sàng'} — ${health.message}`;
      const jobs=await json('/jobs');recent.replaceChildren();
      const labels={complete:'Hoàn tất',running:'Đang đọc',queued:'Đang chờ',failed:'Thất bại',interrupted:'Bị gián đoạn'};
      if(!jobs.length)recent.append(make('small','Chưa có lịch sử nạp nền.'));
      for(const job of jobs.slice(0,8)){
        const item=make('details');item.append(make('summary',`${labels[job.status]||job.status} · ${job.name}`));
        item.append(make('p',new Date(job.updated_at).toLocaleString('vi-VN')));
        item.append(make('p',job.error||`${job.completed_pages||0}/${job.total_pages||'?'} trang`));
        recent.append(item);
      }
    }catch(e){runtime.textContent=e.message;}finally{reload.disabled=false;}
  }
  reload.onclick=status;status();
  window.addEventListener('report-loaded',({detail:{data,id}})=>{
    selected=id;current=data;graphView.replaceChildren();graphInfo.textContent='Quan hệ từ kho dữ liệu thực, không phải graph minh họa.';
    panel.replaceChildren(make('h2','Tổng quan có bằng chứng'));
    const exportLink=make('a','Tải toàn bộ phân tích JSON');exportLink.href=`/documents/${encodeURIComponent(id)}/export`;
    panel.append(exportLink,make('p',data.summary.warning));
    const grid=make('div');grid.style.cssText='display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px';
    for(const item of data.summary.cards){
      const card=make('article');card.style.cssText='border:1px solid #dce5eb;border-radius:8px;padding:14px';
      card.append(make('strong',item.metric),make('p',item.period));
      const value=make('p',item.value===null?'Cần kiểm tra':`${money(item.value)} ${item.unit}`);value.style.fontWeight='700';card.append(value);
      if(item.comparison)card.append(make('p',`${money(Number(item.comparison.value))}% so với ${item.comparison.period}`));
      if(item.warning){const warning=make('small',item.warning);warning.style.display='block';card.append(warning);}
      const fact=data.facts.find(f=>f.id===item.fact_ids[0]);if(fact)card.append(source(fact,id));
      if(item.comparison){const details=make('details');details.append(make('summary','Công thức và đầu vào'),make('pre',item.comparison.formula));
        for(const key of item.comparison.fact_ids){const f=data.facts.find(v=>v.id===key);if(f){details.append(make('p',`${f.metric} · ${f.period_label}: ${money(f.value)} ${f.unit}`),source(f,id));}}card.append(details);}
      grid.append(card);
    }
    panel.append(grid);
    if(!data.summary.cards.length)panel.append(make('p','Chưa có chỉ tiêu chuẩn đủ dữ liệu. Vẫn có thể hỏi đáp từ văn bản.'));
    for(const ratio of data.summary.ratios){const details=make('details');details.append(make('summary',`${ratio.metric} · ${ratio.period}: ${money(Number(ratio.value))}%`),make('pre',ratio.formula));
      for(const key of ratio.fact_ids){const f=data.facts.find(v=>v.id===key);if(f)details.append(make('p',`${f.metric}: ${money(f.value)} ${f.unit}`),source(f,id));}panel.append(details);}
  });
  graphButton.onclick=async()=>{
    if(!selected)return;const id=selected;graphButton.disabled=true;
    try{
      const graph=await json(`/documents/${encodeURIComponent(id)}/graph?limit=1000`);
      if(selected!==id)return;
      graphInfo.textContent=`${graph.nodes.length}/${graph.total_nodes} node · ${graph.edges.length} cạnh${graph.truncated?' · đã giới hạn':''}`;
      graphView.replaceChildren();
      for(const metric of graph.nodes.filter(n=>n.kind==='Metric')){
        const detail=make('details');detail.append(make('summary',metric.name));
        const ids=graph.edges.filter(e=>e.target===metric.id&&e.relation==='MEASURES').map(e=>e.source);
        if(!ids.some(key=>current.facts.some(f=>f.id===key)))continue;
        for(const key of ids){const fact=current.facts.find(f=>f.id===key);if(!fact)continue;
          const row=make('article');row.append(make('p',`MEASURES ← ${fact.period_label}: ${money(fact.value)} ${fact.unit}`),make('small',`SUPPORTED_BY → ${fact.chunk_id} · ${fact.quality}`),source(fact,id));detail.append(row);}
        graphView.append(detail);
      }
    }catch(e){graphInfo.textContent=e.message;}finally{graphButton.disabled=false;}
  };
})();
