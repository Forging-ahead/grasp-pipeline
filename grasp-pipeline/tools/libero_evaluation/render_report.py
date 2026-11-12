"""Self-contained HTML with every new evaluation video embedded, no network dependencies."""
import base64, collections, html, json, sys
from pathlib import Path

r=Path(sys.argv[1]).resolve();d=r/'delivery'
rows=json.loads((d/'RESULTS.json').read_text());s=json.loads((d/'SUMMARY.json').read_text())
groups=json.loads((d/'CATEGORY_SUMMARY.json').read_text());p=json.loads((d/'PROTOCOL.json').read_text())
h=html.escape
notes={
'single_surface':'先按失败阶段分组：抓取未执行时检查接近候选；已经搬到目标附近时检查释放高度、支撑面接触和控制残差。',
'single_container':'对放置超时检查物体与开口的相对位姿、附着姿态、下降路线和实际跟踪误差。',
'single_compartment':'检查物体加夹爪的包络能否穿过狭窄开口，以及体积、接触模型与末端目标是否一致。',
'single_cabinet':'区分抓取不可达与放置进入柜格失败，检查柜门/柜架状态、入口位置、姿态和碰撞约束。',
'single_extract':'检查抽屉内的可见点云、接近方向、夹爪与抽屉边缘的几何关系，并验证抓后退出路线。',
'single_rack':'检查架体入口、瓶子朝向及接触后的支撑稳定性，不能只验证末端到达。',
'single_stack':'检查实际接触支撑、相对对齐和释放后的稳定性；保持单物体堆叠与整堆搬运任务分开。',
'multi_place':'检查多个阶段之间的障碍物更新、前一物体占位和抓放顺序；部分子目标通过不算整任务成功。',
'stack_transport':'需要显式先堆叠、再保持整堆搬运的技能。仅在托盘里最终叠好不能证明完成了语言要求的过程。',
'drawer':'检查把手候选、实际接触、滑动轴约束及释放后的关节状态，区分预检拒绝和实际拉动失败。',
'microwave':'检查门把手接近、铰链圆弧跟随和释放后的门角度；直线位移不足以覆盖旋转门操作。',
'stove':'检查旋钮实际接触、旋转轴及结束后的开关谓词，已在初态满足的条件不算一次旋钮操作。',
'drawer_place':'逐阶段检查打开、抓取、抽屉内放置和关闭；验证每次操作后状态变化及后续可达性。',
'microwave_place':'分别检查开门、杯子进腔和关门，定位是否卡在狭窄空间下降或控制跟踪。',
'stove_place':'检查旋钮与抓放间的动作衔接、障碍物刷新、目标支撑和所有最终谓词。',
'push':'需要持续接触的推移路径、接触方向和对象运动反馈；末端自由空间路径可行不等于推移可行。'}
table=[]
for g in groups:
    reasons='；'.join(f'{k} {v}' for k,v in g['failure_reasons'].items()) or '本初态未观察到失败'
    table.append(f'<tr data-category="{h(g["category"])}"><td>{h(g["name"])}</td><td>{g["tasks"]}</td><td>{g["full_success"]}</td><td>{g["failed"]}</td><td><b>{g["success_rate"]:.1%}</b><span class="bar"><i style="width:{100*g["success_rate"]:.2f}%"></i></span></td><td>{h(reasons)}'+(f'<br><span class="warn">其中成功记录有{g["success_with_order_issue"]}项已知顺序问题</span>' if g['success_with_order_issue'] else '')+'</td></tr>')
cards=[]
for i,x in enumerate(rows,1):
    result=x['result'];ok=x['full_success'];label='通过' if ok else '失败'
    reason=x['failure']['label'];video=base64.b64encode((d/x['video_path']).read_bytes()).decode()
    stages=[]
    for j,st in enumerate(result.get('stages',[]),1):
        skill=st.get('stage',{});status=st.get('status','unknown')
        stages.append(f'<tr><td>{j}</td><td>{h(skill.get("skill",""))}</td><td>{h(json.dumps(skill.get("predicate",[]),ensure_ascii=False))}</td><td>{h(status)}</td><td>{h(str(st.get("official_subgoal","—")))}</td><td>{st.get("steps","—")}</td></tr>')
    evidence=h(str(x['failure'].get('evidence','')))
    if len(evidence)>2200:evidence=evidence[:2200]+'…（完整记录见下）'
    details=json.dumps(dict(result=result,analysis={k:x[k] for k in ('failure','last_phase','phase_action_counts','order_note','queue_record')}),ensure_ascii=False,indent=2)
    cards.append(f'''<details class="case" data-category="{h(x['category'])}" data-status="{'pass' if ok else 'fail'}" data-search="{h((x['key']+' '+x['language']+' '+reason).lower(),quote=True)}">
<summary><span class="badge {'pass' if ok else 'fail'}">{label}</span><span><b>{i:03d} · {h(x['language'])}</b><small>{h(x['key'])}</small></span></summary>
<div class="body"><p class="category">{h(x['category_name'])} · {h(reason)}</p>
<video controls preload="none" playsinline src="data:video/mp4;base64,{video}"></video>
<p class="meta">{result['steps']}步 · 仿真动作{x['sim_action_seconds']:.2f}秒 · 实际耗时{x['wall_seconds']/60:.2f}分钟 · GPU{x['queue_record']['gpu']} · 初态0</p>
{('<p class="alert">'+h(x['order_note'])+'</p>') if x['order_note'] else ''}
<p>{evidence}</p><p class="meta">最后动作阶段：{h(x['last_phase'])}；{'已进入抓取/接触动作' if x['physical_skill_attempt'] else '未进入抓取/接触动作，可能只有准备动作'}。最终官方条件：{h(str(result.get('official_goal_at_end')))}；完整流水线：{h(str(result.get('pipeline_success')))}。</p>
<div class="scroll"><table class="stages"><thead><tr><th>阶段</th><th>技能</th><th>子目标</th><th>状态</th><th>子目标通过</th><th>步数</th></tr></thead><tbody>{''.join(stages)}</tbody></table></div>
<details><summary>完整结果与退出诊断</summary><pre>{h(details)}</pre></details>
</div></details>''')
diagnostics=''.join('<article><h3>'+h(g['name'])+'</h3><p>'+h('；'.join(f'{a}：{b}项' for a,b in g['failure_reasons'].items()) or '本初态全部通过。')+'</p><p class="meta">后续诊断方向：'+h(notes[g['category']])+'</p></article>' for g in groups)
failure_list=''.join('<li>'+h(k)+f'：{v}项</li>' for k,v in json.loads((d/'FAILURE_SUMMARY.json').read_text()).items())
options=''.join(f'<option value="{h(g["category"])}">{h(g["name"])}</option>' for g in groups)
protocol_view=h(json.dumps(p,ensure_ascii=False,indent=2))
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO 130任务 · libero分支2026-11-13复测</title><style>
:root{font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#172b3a;background:#f3f6f9;line-height:1.65}*{box-sizing:border-box}body{margin:0}main{max-width:1300px;margin:auto;padding:36px 24px 80px}h1{font-size:34px;line-height:1.25;margin:8px 0 20px}h2{font-size:24px;margin:38px 0 14px}h3{margin:4px 0 8px;font-size:17px}p{margin:12px 0}.eyebrow{color:#266178;font-weight:700;letter-spacing:.1em}.muted,.meta{color:#617284;font-size:14px}.banner,.alert{padding:16px 20px;border-left:4px solid #d99a23;background:#fff8e8;border-radius:5px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.metric{background:#fff;border:1px solid #dce5eb;border-radius:14px;padding:20px}.metric b{display:block;font-size:34px;line-height:1.3;color:#155d79}.metric span{color:#617284;font-size:14px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #e1e8ee;vertical-align:top}th{background:#e7f0f5;white-space:nowrap}td:nth-child(6){font-size:13px;min-width:230px}tbody tr[data-category]{cursor:pointer}tbody tr[data-category]:hover{background:#eef6fc}.bar{height:5px;background:#e4eaf0;display:block;min-width:85px;margin-top:6px;border-radius:5px}.bar i{display:block;background:#228273;height:100%;border-radius:5px}.warn{color:#9d6500}.analysis-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.analysis-grid article{background:#fff;border:1px solid #dce5eb;border-radius:10px;padding:16px}.filters{display:flex;flex-wrap:wrap;gap:10px;padding:14px;background:#e7f0f5;border-radius:10px;margin:16px 0}input,select,button{font:inherit;border:1px solid #c4d1dc;border-radius:6px;padding:9px;background:#fff}input{flex:1;min-width:220px}button{cursor:pointer}.case{background:white;border:1px solid #dce5eb;border-radius:12px;margin:12px 0;overflow:hidden}.case>summary{display:flex;align-items:flex-start;gap:12px;cursor:pointer;padding:16px;list-style:none}.case[open]>summary{border-bottom:1px solid #e1e8ee}.case summary small{display:block;color:#617284;overflow-wrap:anywhere;font-size:12px;margin-top:4px}.badge{padding:2px 9px;border-radius:5px;white-space:nowrap;font-size:13px}.pass{color:#14624d;background:#dff3e9}.fail{color:#9a3c34;background:#fae5e2}.body{padding:4px 20px 20px}.category{font-weight:600}video{display:block;width:min(100%,880px);height:auto;background:#152631;border-radius:7px;aspect-ratio:2/1}.stages{font-size:13px}.stages td{min-width:unset}pre{overflow:auto;max-height:450px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:15px;font-size:12px}.body details>summary{cursor:pointer;color:#225e79;margin-top:14px}.hidden{display:none!important}footer{margin-top:40px;color:#617284;border-top:1px solid #cbd7e1;padding-top:16px}@media(max-width:760px){main{padding:24px 12px}h1{font-size:27px}.metrics{grid-template-columns:repeat(2,1fr)}.metric{padding:14px}.metric b{font-size:28px}.analysis-grid{grid-template-columns:1fr}th,td{padding:9px}.body{padding:4px 12px 16px}.filters input,.filters select{width:100%}}@media print{video,.filters{display:none}main{max-width:none}.case{break-inside:avoid}.case:not([open])>.body{display:block}}
</style></head><body><main><div class="eyebrow">GRASPGEN × CUROBO × LIBERO</div>'''
page+=f'''<h1>130个任务，libero分支同版本复测（2026-11-13）</h1><p>本报告仅包含四卡服务器上本轮新运行的130条轨迹。每个任务固定初态0、随机种子0，成功和失败均保留视频；没有拼接历史版本的成功结果。</p>
<div class="metrics"><div class="metric"><b>{s['full_success']}/130</b><span>完整执行且最终条件通过</span></div><div class="metric"><b>{s['success_rate']:.1%}</b><span>本初态单次成功比例</span></div><div class="metric"><b>{s['failed']}</b><span>未完整完成任务</span></div><div class="metric"><b>130</b><span>本轮视频 · 16类任务</span></div></div>
<div class="banner"><b>统计口径：</b>成功要求实际执行、流水线完整结束且原始LIBERO最终目标全部通过。{'其中'+str(s['success_with_order_issue'])+'条通过记录存在已知“先堆叠再整体搬运”的顺序问题，剔除后为'+str(s['success_excluding_known_order_violations'])+'/130；这些轨迹不能直接作为完整语言任务的成功SFT样本。' if s['success_with_order_issue'] else '堆叠整体搬运任务另检查语言顺序，最终状态谓词本身不能证明过程正确。'} 这是一任务一初态的代码策略回归，不是官方多初态、多随机种子的基准成功率。</div>
<p class="meta">仅看结束时原始官方条件为True：{s['official_goal_at_end']}/130；完整执行口径：{s['full_success']}/130。若夹爪仍持物或后续技能未完成，即使某些状态谓词已满足，也不记为完整轨迹成功。</p>
<h2>这一版代码和测试范围</h2><p>本轮使用libero分支6d8cd7d，保留v10策略，其开发由先前的抓放演示、抓放任务回归以及机构和复合任务调试迭代得到。本轮冻结后覆盖全部130任务；其中部分任务参与过开发，因此不将这些结果称为未见任务的泛化评测。</p><p>手写code policy从BDDL任务目标与仿真真值几何计算阶段目标；预训练GraspGen生成、评分抓取位姿；cuRobo负责全局运动规划，局部接触阶段还使用MuJoCo雅可比IK及实际接触反馈，机器人动作通过OSC_POSE执行。尚未训练VLM决策模型，也不是只依赖RGB图像的感知方案。</p>
<details><summary>策略迭代样例与本轮统计的关系</summary><p>早期从Spatial、Object、Goal中的26个直接抓放任务开展调试，随后扩展抓放覆盖。之后使用63个尚未解决的任务及5个回归任务检验机构操作、复合调度和通用放置搜索，继续修正夹爪衔接、放置候选预检和机构同步检查。</p><p>最终v10的末期回归样例包括：牛奶放入篮子、杯子放入微波炉并关门、Study Scene3书本放入收纳盒右格、Study Scene4中间书本放到柜架。这一阶段调整的是接近目标时有限且有进展条件的控制延长；没有训练新的GraspGen权重。上述信息只说明开发来源，任何旧版本成功记录都没有纳入本页130项统计。</p></details>
<p class="meta">GraspGen用于刚体抓放的抓取候选；抽屉、门和旋钮的接触候选由几何规则生成。整套集成的结果不能代替生成器、判别器或规划器各自贡献的消融实验。并发负载会影响墙钟耗时和预算内的搜索深度，时间预算失败不直接证明模型能力不足。</p>
<p class="meta">四张RTX4090，起初每卡2项完成环境验证，之后根据本轮实际资源占用提高并发，最多16项并行，调度变化已留档；策略、初态和任务预算未改变。最大5000个控制步；96项现实时间预算1200秒、34项1800秒，含模型加载、规划和仿真。排队不计入单任务耗时。总轮次耗时约{s['wall_elapsed_seconds']/60:.1f}分钟；单任务实际耗时范围{s['wall_task_min_seconds']/60:.2f}—{s['wall_task_max_seconds']/60:.2f}分钟。仿真动作时间与实际耗时分开列出。</p>
<h2>16类结果</h2><p class="meta">点击任一类别行可筛选下方任务与视频。所有比例均只使用本轮对应类别的记录。</p><div class="scroll"><table id="categories"><thead><tr><th>任务类型</th><th>总数</th><th>通过</th><th>失败</th><th>成功比例</th><th>本轮失败退出原因</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>失败分析</h2><p>本轮{s['failed']}项未完整完成，其中{s['failure_only_preflight']}项未进入抓取/接触动作，{s['failure_after_physical_attempt']}项执行过实际技能动作。前者视频较短通常表示候选预检或搜索在机器人动作前退出；规划计算的等待时间不会出现在仿真动作视频里。</p><ul>{failure_list}</ul><p class="banner">退出原因与物理根因分开理解：没有找到路径，不等于真实环境中没有路径；未执行夹取，不能判为GraspGen抓取能力失败。这里只报告本轮日志可验证的阶段、检查和残差；碰撞近似、终点选择或控制参数是否是根因，需要从失败状态做对照验证。</p><div class="analysis-grid">{diagnostics}</div>
<h2 id="videos">全部130个任务的视频与记录</h2><p>展开任务即可播放。视频已经嵌入此HTML文件，转发整个文件后可离线观看，不依赖服务器、本地视频路径或外部网络。每段结尾额外停留2秒便于查看；零动作或只有准备动作的失败不会被填充为虚构动作。</p>
<div class="filters"><select id="category"><option value="">全部16类</option>{options}</select><select id="status"><option value="">成功与失败</option><option value="pass">只看通过</option><option value="fail">只看失败</option></select><input id="search" type="search" placeholder="搜索任务名、物体或失败原因"><button id="reset">清空筛选</button></div><p id="count" class="meta">显示130/130项</p><div id="cases">{''.join(cards)}</div>
<h2>复现实验与审计</h2><p>已逐项核对libero分支冻结源码及配置、130个初态编号、130份原始BDDL目标、连续动作/状态数量及视频完整解码。另核对模型权重、初态资产与外部代码指纹。数据包提供逐任务结果、阶段记录、连续状态、动作、关键点和单独视频；关键点标签来自手写code policy，不是VLM作出的工具选择。</p><details><summary>完整运行协议</summary><pre>{protocol_view}</pre></details><footer>仅包含本轮libero分支6d8cd7d四卡全量运行 · 初态0 · 每任务1次 · 失败视频全部保留 · 不混用历史结果</footer></main>
'''
page+='''<script>
const cards=[...document.querySelectorAll('.case')],cat=document.querySelector('#category'),state=document.querySelector('#status'),search=document.querySelector('#search');
function filter(){let n=0;const q=search.value.trim().toLowerCase();cards.forEach(c=>{const keep=(!cat.value||c.dataset.category===cat.value)&&(!state.value||c.dataset.status===state.value)&&(!q||c.dataset.search.includes(q));c.classList.toggle('hidden',!keep);if(keep)n++;else{c.open=false;c.querySelector('video').pause();}});document.querySelector('#count').textContent=`显示${n}/130项`;return n;}
[cat,state,search].forEach(e=>e.addEventListener('input',filter));document.querySelector('#reset').addEventListener('click',()=>{cat.value='';state.value='';search.value='';filter();});document.querySelectorAll('#categories tr[data-category]').forEach(tr=>tr.addEventListener('click',()=>{cat.value=tr.dataset.category;state.value='';search.value='';filter();document.querySelector('#videos').scrollIntoView({behavior:'smooth'});}));cards.forEach(c=>c.querySelector('video').addEventListener('play',e=>cards.forEach(o=>{const v=o.querySelector('video');if(v!==e.target)v.pause();})));
</script></body></html>'''
(d/'LIBERO130_libero分支复测_20261113_含全部视频.html').write_text(page)
print('Embedded all130 videos into self-contained HTML',flush=True)
