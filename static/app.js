const state = {
  meta: null,
  assessment: null,
  activeTab: "virtual",
};

const tabs = [
  ["virtual", "Virtual Qualification"],
  ["lifecycle", "Life-Cycle Profiles"],
  ["acceleration", "Accelerated Equivalence"],
  ["cycles", "Cycle Extractor"],
  ["physics", "Physics Models"],
  ["monte", "Monte Carlo"],
  ["map", "Failure-Site Map"],
  ["sensitivity", "Sensitivity"],
  ["iso", "Iso-Life Contours"],
  ["planner", "Test Planner"],
  ["whatif", "DfR What-If"],
  ["validation", "Validation"],
  ["report", "Qualification Report"],
  ["plugins", "Model Plugins"],
];

const $ = (id) => document.getElementById(id);

const fmt = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

const clean = (value) => String(value || "").replaceAll("_", " ");
const pct = (value) => `${fmt(Number(value) * 100, 0)}%`;
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

async function postJson(url, values) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

const materialById = (id) => state.meta.material_sets.find((m) => m.set_id === id) || state.meta.material_sets[0];
const couponById = (id) => state.meta.coupon_structures.find((c) => c.id === id) || state.meta.coupon_structures[0];
const couponInkById = (id) => state.meta.coupon_inks.find((i) => i.id === id) || state.meta.coupon_inks[0];
const couponsByZone = (zone) => state.meta.coupon_structures.filter((c) => c.zone === zone);

function populateControls() {
  $("materialSet").innerHTML = state.meta.material_sets.map((m) => `<option value="${m.set_id}">${m.stack}</option>`).join("");
  $("couponInk").innerHTML = state.meta.coupon_inks.map((i) => `<option value="${i.id}">${i.label}</option>`).join("");
  $("testMethod").innerHTML = state.meta.coupon_test_methods.map((m) => `<option value="${m.id}">${m.label}</option>`).join("");

  const zoneAOptions = couponsByZone("interface")
    .map((c) => `<option value="${c.id}">${c.label} - ${c.purpose}</option>`)
    .join("");
  const zoneBOptions = couponsByZone("bonding")
    .map((c) => `<option value="${c.id}">${c.label} - ${c.purpose}</option>`)
    .join("");
  $("zoneAStructure").innerHTML = zoneAOptions;
  $("zoneBStructure").innerHTML = zoneBOptions;

  const defaults = state.meta.defaults.coupon;
  $("materialSet").value = defaults.ink_family === "high_temp_500c" ? "ani_alumina" : "ag_np_alumina";
  $("couponInk").value = defaults.ink_family;
  $("analysisZone").value = defaults.coupon_zone;
  $("zoneAStructure").value = couponsByZone("interface")[0]?.id || "";
  $("zoneBStructure").value = defaults.coupon_structure;
  $("testMethod").value = defaults.test_method;
  $("agingTemp").value = defaults.aging_temp_c;
  $("agingHours").value = defaults.aging_hours;
  $("thermalCycles").value = defaults.thermal_cycles;
  $("strainPct").value = defaults.strain_pct;
  $("voidFraction").value = defaults.ct_void_fraction_pct;
  $("edgeRoughness").value = defaults.edge_roughness_um;
  $("sampleCount").value = 260;
  $("candidateCount").value = 360;

  renderMaterialStack();
  syncLabels();
}

function activeCouponId() {
  return $("analysisZone").value === "bonding" ? $("zoneBStructure").value : $("zoneAStructure").value;
}

function syncInkToMaterial() {
  const material = materialById($("materialSet").value);
  $("couponInk").value = material.set_id === "ani_alumina" ? "high_temp_500c" : "baseline_ag";
  renderMaterialStack();
}

function renderMaterialStack() {
  const material = materialById($("materialSet").value);
  $("materialStack").innerHTML = `
    <div class="stack-row"><span class="swatch ink"></span><strong>${material.ink}</strong><small>conductive ink</small></div>
    <div class="stack-row"><span class="swatch alumina"></span><strong>${material.substrate}</strong><small>shared coupon platform</small></div>
    <div class="stack-row"><span class="swatch conduct"></span><strong>${fmt(material.conductivity_s_m / 1e6, 2)} MS/m</strong><small>nominal conductivity</small></div>
  `;
}

function syncLabels() {
  $("agingTempOut").textContent = `${fmt($("agingTemp").value, 0)} C`;
  $("agingHoursOut").textContent = `${fmt($("agingHours").value, 0)} h`;
  $("thermalCyclesOut").textContent = fmt($("thermalCycles").value, 0);
  $("strainOut").textContent = `${fmt($("strainPct").value, 2)}%`;
  $("voidOut").textContent = `${fmt($("voidFraction").value, 1)}%`;
  $("roughOut").textContent = `${fmt($("edgeRoughness").value, 1)} um`;
  $("sampleOut").textContent = fmt($("sampleCount").value, 0);
  $("candidateOut").textContent = fmt($("candidateCount").value, 0);
}

function couponPayload() {
  const coupon = couponById(activeCouponId());
  const ink = couponInkById($("couponInk").value);
  const agingTemp = Number($("agingTemp").value);
  const lineWidth = coupon.nominal_width_um;
  const thickness = 3.2;
  const resistance = (coupon.path_length_mm * 1e-3) / Math.max((lineWidth * 1e-6) * (thickness * 1e-6) * ink.nominal_conductivity_s_m, 1e-12);
  return {
    coupon_structure: coupon.id,
    ink_family: ink.id,
    test_method: $("testMethod").value,
    nominal_width_um: coupon.nominal_width_um,
    path_length_mm: coupon.path_length_mm,
    overlap_area_mm2: coupon.overlap_area_mm2,
    bond_area_mm2: coupon.bond_area_mm2,
    print_speed_mm_s: 14.5,
    atomizer_voltage_v: 35.0,
    carrier_flow_sccm: 28.0,
    sheath_flow_sccm: 60.0,
    substrate_temp_c: 55.0,
    cure_peak_temp_c: ink.id === "high_temp_500c" ? 430.0 : 210.0,
    cure_time_min: ink.id === "high_temp_500c" ? 55.0 : 35.0,
    aging_temp_c: agingTemp,
    aging_hours: Number($("agingHours").value),
    cycle_low_temp_c: -40.0,
    cycle_high_temp_c: agingTemp >= 500 ? 500.0 : 125.0,
    thermal_cycles: Number($("thermalCycles").value),
    bend_radius_mm: Math.max(3.0, 18.0 / Math.max(Number($("strainPct").value), 0.1)),
    strain_pct: Number($("strainPct").value),
    strain_cycles: Number($("thermalCycles").value) * 4,
    ct_void_fraction_pct: Number($("voidFraction").value),
    oxidation_index: clamp((agingTemp - 25) / 760 + Number($("agingHours").value) / 2800, 0.04, 0.95),
    edge_roughness_um: Number($("edgeRoughness").value),
    alignment_error_um: 18 + Number($("voidFraction").value) * 1.2,
    line_width_um: lineWidth,
    thickness_um: thickness,
    initial_resistance_ohm: resistance,
    candidates: Number($("candidateCount").value),
    samples: Number($("sampleCount").value),
  };
}

function renderTabs() {
  $("tabStrip").innerHTML = tabs
    .map(([id, label]) => `<button class="tab-button ${id === state.activeTab ? "active" : ""}" data-tab="${id}" type="button">${label}</button>`)
    .join("");
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = button.dataset.tab;
      renderTabs();
      renderTabPanel();
    });
  });
}

async function runAssessment() {
  $("runAssessment").disabled = true;
  $("runAssessment").textContent = "Running assessment...";
  $("heroNarrative").textContent = "Running physics-of-failure, uncertainty, and Digital Twin modules.";
  try {
    state.assessment = await postJson("/api/rap/assessment", couponPayload());
    renderAll();
  } finally {
    $("runAssessment").disabled = false;
    $("runAssessment").textContent = "Run RAP Assessment";
  }
}

function metricCard(label, value, sub = "") {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong>${sub ? `<small>${sub}</small>` : ""}</div>`;
}

function statusPill(decision) {
  const cls = decision === "PASS" ? "pass" : decision === "DEFER_TO_INSPECTION" ? "defer" : "marginal";
  return `<span class="status-pill ${cls}">${clean(decision)}</span>`;
}

function bar(label, value, max = 100, cls = "") {
  const width = clamp((Number(value) / max) * 100, 0, 100);
  return `
    <div class="bar-row">
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${width}%"></div></div>
      <strong>${fmt(value, 1)}</strong>
    </div>
  `;
}

function probabilityBars(items) {
  return Object.entries(items || {})
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => bar(clean(label), value * 100, 100, label === "pass" || label === "PASS" ? "good" : "warn"))
    .join("");
}

function renderHero() {
  if (!state.assessment) return;
  const base = state.assessment.baseline;
  const pred = base.prediction;
  const derived = base.derived;
  const report = state.assessment.qualification_report;
  $("heroTitle").textContent = `${clean(base.input.coupon_structure)} assessment`;
  $("heroNarrative").textContent = report.key_results.join(" | ");
  $("heroMetrics").innerHTML = [
    metricCard("Qualification", statusPill(base.qualification_decision), `${fmt(derived.decision_confidence_pct, 0)}% confidence`),
    metricCard("Reliability", fmt(pred.reliability_score, 1), "score / 100"),
    metricCard("RUL estimate", `${fmt(derived.remaining_useful_life_hours, 0)} h`, `state index ${fmt(derived.degradation_state_index, 1)}`),
    metricCard("Dominant mode", clean(base.failure_mode), `${fmt(derived.failure_risk_index, 1)} risk index`),
  ].join("");
  renderCouponSvg();
}

function renderCouponSvg() {
  const svg = $("couponSvg");
  const sites = state.assessment?.failure_site_map?.sites || [];
  const active = activeCouponId();
  const dotMarkup = sites
    .map((site) => {
      const risk = site.risk_index;
      const fill = risk > 70 ? "#b85032" : risk > 45 ? "#c28a21" : "#2d7d46";
      const r = site.id === active ? 12 : 8;
      return `<g>
        <circle cx="${site.coordinates.x * 7.1}" cy="${site.coordinates.y * 3.4}" r="${r}" fill="${fill}" stroke="#fff" stroke-width="3"/>
        <text x="${site.coordinates.x * 7.1 + 13}" y="${site.coordinates.y * 3.4 + 4}" font-size="11" fill="#20333a" font-weight="800">${site.label}</text>
      </g>`;
    })
    .join("");
  const rel = state.assessment?.baseline?.prediction?.reliability_score;
  svg.innerHTML = `
    <defs>
      <linearGradient id="couponBg" x1="0" x2="1">
        <stop offset="0" stop-color="#fbfcfa"/>
        <stop offset="1" stop-color="#eef7f3"/>
      </linearGradient>
    </defs>
    <rect x="28" y="38" width="704" height="330" rx="8" fill="url(#couponBg)" stroke="#1c3438" stroke-width="2"/>
    <line x1="380" y1="56" x2="380" y2="350" stroke="#8da1a6" stroke-width="2" stroke-dasharray="8 8"/>
    <text x="72" y="78" fill="#1256a0" font-size="18" font-weight="900">Zone A - interface structures</text>
    <text x="440" y="78" fill="#267237" font-size="18" font-weight="900">Zone B - bonding structures</text>
    <path d="M82 126h160M82 154h160M82 182h160" stroke="#62686a" stroke-width="8"/>
    <path d="M82 224h160M82 252h160M82 280h160" stroke="#c8a13a" stroke-width="8"/>
    <path d="M274 122c68 0 68 32 0 32s-68 32 0 32s68 32 0 32s-68 32 0 32s68 32 0 32" fill="none" stroke="#c8a13a" stroke-width="10" stroke-linecap="round"/>
    <rect x="315" y="122" width="28" height="28" fill="#62686a"/><rect x="315" y="174" width="28" height="28" fill="#c8a13a"/>
    <path d="M420 132h134M420 190h134M420 248h134" stroke="#62686a" stroke-width="9"/>
    <rect x="404" y="118" width="28" height="28" fill="#62686a"/><rect x="552" y="118" width="28" height="28" fill="#62686a"/>
    <rect x="610" y="116" width="64" height="64" fill="#62686a"/><rect x="628" y="134" width="28" height="28" fill="#c8a13a"/>
    <path d="M606 250h54v-24h38v74h-38v-24h-54z" fill="#c8a13a"/>
    <rect x="52" y="58" width="24" height="24" fill="#6b6f70"/><rect x="684" y="58" width="24" height="24" fill="#6b6f70"/>
    <rect x="52" y="328" width="24" height="24" fill="#6b6f70"/><rect x="684" y="328" width="24" height="24" fill="#6b6f70"/>
    ${dotMarkup}
    <text x="54" y="396" fill="#20333a" font-size="18" font-weight="900">BOND-AI-RAP coupon reliability ${rel ? fmt(rel, 1) : "pending"}</text>
  `;
}

function renderAll() {
  renderHero();
  renderTabs();
  renderTabPanel();
}

function renderTabPanel() {
  if (!state.assessment) {
    $("tabPanel").innerHTML = `<div class="empty-state">Run an assessment to activate the RAP feature stack.</div>`;
    return;
  }
  const renderers = {
    virtual: renderVirtual,
    lifecycle: renderLifecycle,
    acceleration: renderAcceleration,
    cycles: renderCycles,
    physics: renderPhysics,
    monte: renderMonteCarlo,
    map: renderFailureMap,
    sensitivity: renderSensitivity,
    iso: renderIsoLife,
    planner: renderPlanner,
    whatif: renderWhatIf,
    validation: renderValidation,
    report: renderReport,
    plugins: renderPlugins,
  };
  $("tabPanel").innerHTML = renderers[state.activeTab]();
}

function renderVirtual() {
  const item = state.assessment.virtual_qualification;
  const margins = item.critical_margins.map((m) => bar(clean(m.metric), m.margin, 50, m.margin >= 0 ? "good" : "warn")).join("");
  return `
    <div class="tab-header"><p class="eyebrow">Virtual qualification</p><h2>Pre-build coupon disposition</h2><p>${item.reviewer_readout}</p></div>
    <div class="analysis-grid three">
      ${metricCard("Decision", statusPill(item.decision), `${fmt(item.confidence_pct, 0)}% model confidence`)}
      ${metricCard("Reliability", fmt(item.reliability_score, 1), `q90 ${fmt(item.q90_reliability.low, 1)} to ${fmt(item.q90_reliability.high, 1)}`)}
      ${metricCard("RUL estimate", `${fmt(item.remaining_useful_life_hours, 0)} h`, clean(item.failure_mode))}
    </div>
    <div class="panel-band"><h3>Critical margins</h3>${margins}</div>
  `;
}

function renderLifecycle() {
  const profile = state.assessment.life_cycle_profile;
  return `
    <div class="tab-header"><p class="eyebrow">Life-cycle profile manager</p><h2>${profile.mission_severity} severity mission profile</h2><p>Damage is normalized across thermal aging, cycling, strain, and inspection descriptors.</p></div>
    <div class="analysis-grid two">
      ${profile.segments
        .map(
          (s) => `<div class="feature-card"><h3>${s.name}</h3><p>${s.condition}</p>${bar(s.monitored_metric, s.damage_index * 100, 100, "accent")}</div>`
        )
        .join("")}
    </div>
    <div class="panel-band">${metricCard("Cumulative damage", fmt(profile.cumulative_damage_index * 100, 1), `${profile.dominant_segment.name} dominates`)}</div>
  `;
}

function renderAcceleration() {
  const a = state.assessment.accelerated_equivalence;
  return `
    <div class="tab-header"><p class="eyebrow">Accelerated test equivalence</p><h2>Field-use translation</h2><p>${a.assumption}</p></div>
    <div class="analysis-grid four">
      ${metricCard("Thermal AF", fmt(a.thermal_acceleration_factor, 1), `${fmt(a.use_temperature_c, 0)} C use to ${fmt(a.test_temperature_c, 0)} C test`)}
      ${metricCard("Equivalent aging", `${fmt(a.field_equivalent_hours, 0)} h`, "field-use basis")}
      ${metricCard("Cycle AF", fmt(a.cycle_acceleration_factor, 2), "delta-T acceleration")}
      ${metricCard("Equivalent cycles", fmt(a.field_equivalent_cycles, 0), "field-use cycles")}
    </div>
    <div class="notice">${a.warning}</div>
  `;
}

function renderCycles() {
  const c = state.assessment.cycle_extractor;
  return `
    <div class="tab-header"><p class="eyebrow">Temperature-cycle extractor</p><h2>${fmt(c.detected_cycles, 0)} detected cycles, ${fmt(c.temperature_delta_c, 0)} C swing</h2><p>CSV-ready schema: ${c.csv_ready_schema.join(", ")}</p></div>
    <div class="analysis-grid three">
      ${metricCard("Equivalent full cycles", fmt(c.equivalent_full_cycles, 1), "normalized to 540 C swing")}
      ${metricCard("Rainflow damage", fmt(c.rainflow_damage_index * 100, 1), "damage index")}
      ${metricCard("Cycle bins", c.bins.length, "thermal ranges")}
    </div>
    <div class="panel-band"><h3>Cycle bins</h3>${c.bins.map((b) => bar(b.range, b.damage_share * 100, 100, "accent")).join("")}</div>
  `;
}

function renderPhysics() {
  return `
    <div class="tab-header"><p class="eyebrow">Physics-of-failure library</p><h2>Selectable model priors</h2><p>Each model describes what BOND-AI-RAP can calibrate as real coupon data arrives.</p></div>
    <div class="analysis-grid three">
      ${state.assessment.physics_models.models
        .map((m) => `<div class="feature-card"><span class="model-family">${m.family}</span><h3>${m.name}</h3><p>${m.equation}</p><small>${m.bond_ai_use}</small></div>`)
        .join("")}
    </div>
  `;
}

function renderMonteCarlo() {
  const mc = state.assessment.monte_carlo;
  return `
    <div class="tab-header"><p class="eyebrow">Probabilistic reliability</p><h2>${fmt(mc.samples, 0)} sample Monte Carlo envelope</h2><p>Input uncertainty is propagated through the coupon reliability surrogate.</p></div>
    <div class="analysis-grid three">
      ${metricCard("Reliability P10/P50/P90", `${fmt(mc.reliability.p10, 1)} / ${fmt(mc.reliability.p50, 1)} / ${fmt(mc.reliability.p90, 1)}`, "score / 100")}
      ${metricCard("RUL P10/P50/P90", `${fmt(mc.rul_hours.p10, 0)} / ${fmt(mc.rul_hours.p50, 0)} / ${fmt(mc.rul_hours.p90, 0)} h`, "remaining useful life")}
      ${metricCard("Mean reliability", fmt(mc.reliability.mean, 1), "probabilistic center")}
    </div>
    <div class="analysis-grid two">
      <div class="panel-band"><h3>Decision probability</h3>${probabilityBars(mc.decision_probability)}</div>
      <div class="panel-band"><h3>Failure mode probability</h3>${probabilityBars(mc.failure_mode_probability)}</div>
    </div>
  `;
}

function renderFailureMap() {
  const fmap = state.assessment.failure_site_map;
  return `
    <div class="tab-header"><p class="eyebrow">Failure-site ranking map</p><h2>${fmap.highest_risk.label} is the current highest-risk site</h2><p>Zone A interface degradation and Zone B bond reliability are ranked on one coupon map.</p></div>
    <div class="site-list">
      ${fmap.sites
        .map(
          (s) => `<div class="site-row"><strong>${s.label}</strong><span>${s.zone} | ${s.purpose}</span>${bar(clean(s.failure_mode), s.risk_index, 100, s.risk_index > 65 ? "warn" : "accent")}</div>`
        )
        .join("")}
    </div>
  `;
}

function renderSensitivity() {
  const drivers = state.assessment.sensitivity.drivers;
  return `
    <div class="tab-header"><p class="eyebrow">Sensitivity analysis</p><h2>Top drivers of reliability movement</h2><p>Tornado ranking shows what reviewers can change or measure next.</p></div>
    <div class="panel-band">
      ${drivers.map((d) => bar(`${d.label} (${d.direction})`, d.impact, Math.max(1, drivers[0].impact), d.direction.includes("worse") ? "warn" : "good")).join("")}
    </div>
  `;
}

function heatCell(value) {
  const hue = value >= 75 ? "#2d7d46" : value >= 55 ? "#c28a21" : "#b85032";
  return `<span class="heat-cell" style="background:${hue}">${fmt(value, 0)}</span>`;
}

function renderIsoLife() {
  const iso = state.assessment.iso_life;
  const tempRows = iso.temp_hour_matrix
    .map((row) => `<div class="heat-row"><strong>${fmt(row.aging_temp_c, 0)} C</strong>${row.values.map((v) => heatCell(v.reliability_score)).join("")}</div>`)
    .join("");
  const morphRows = iso.morphology_matrix
    .map((row) => `<div class="heat-row"><strong>${fmt(row.ct_void_fraction_pct, 1)}% void</strong>${row.values.map((v) => heatCell(v.reliability_score)).join("")}</div>`)
    .join("");
  return `
    <div class="tab-header"><p class="eyebrow">Iso-life / iso-reliability contours</p><h2>Stress envelopes at a glance</h2><p>Reliability score is shown inside each contour cell.</p></div>
    <div class="analysis-grid two">
      <div class="heat-panel"><h3>Aging temperature vs hours</h3><div class="heat-axis"><span></span>${iso.hour_axis.map((h) => `<span>${fmt(h, 0)} h</span>`).join("")}</div>${tempRows}</div>
      <div class="heat-panel"><h3>Void fraction vs strain</h3><div class="heat-axis"><span></span>${iso.strain_axis.map((s) => `<span>${fmt(s, 1)}%</span>`).join("")}</div>${morphRows}</div>
    </div>
  `;
}

function renderPlanner() {
  const planner = state.assessment.test_planner;
  return `
    <div class="tab-header"><p class="eyebrow">Test planning wizard</p><h2>Uncertainty-guided next coupons</h2><p>Designed to reduce coupon count while closing the model uncertainty loop.</p></div>
    <div class="analysis-grid two">
      <div class="panel-band"><h3>Recommended experiments</h3>${planner.recommended_experiments
        .map((e) => `<div class="feature-card"><h3>#${e.rank} ${clean(e.coupon_structure)}</h3><p>${e.why}</p><small>${fmt(e.condition.aging_temp_c, 0)} C, ${fmt(e.condition.aging_hours, 0)} h, ${fmt(e.condition.thermal_cycles, 0)} cycles, ${fmt(e.condition.ct_void_fraction_pct, 1)}% voids</small></div>`)
        .join("")}</div>
      <div class="panel-band"><h3>Corrective actions</h3>${planner.actions.map((a) => `<p class="action-line">${a}</p>`).join("")}</div>
    </div>
  `;
}

function renderWhatIf() {
  const w = state.assessment.what_if;
  return `
    <div class="tab-header"><p class="eyebrow">Design-for-reliability what-if</p><h2>Best scenario: ${w.best.label}</h2><p>Instant recipe and stress-profile comparisons.</p></div>
    <div class="analysis-grid three">
      ${w.scenarios
        .map((s) => `<div class="feature-card"><h3>${s.label}</h3>${statusPill(s.decision)}${bar("Reliability delta", s.reliability_delta, 40, s.reliability_delta >= 0 ? "good" : "warn")}<small>RUL ${fmt(s.rul_hours, 0)} h (${fmt(s.rul_delta_hours, 0)} h delta)</small></div>`)
        .join("")}
    </div>
  `;
}

function renderValidation() {
  const v = state.assessment.validation;
  return `
    <div class="tab-header"><p class="eyebrow">Validation and calibration</p><h2>Prediction vs measured dashboard</h2><p>${v.calibration_message}</p></div>
    <div class="analysis-grid three">
      ${metricCard("Reliability MAE", fmt(v.holdout_metrics.mae.reliability_score, 2), "holdout synthetic")}
      ${metricCard("Reliability R2", fmt(v.holdout_metrics.r2.reliability_score, 3), "holdout synthetic")}
      ${metricCard("Q90 band", `+/- ${fmt(v.holdout_metrics.q90.reliability_score, 2)}`, `${fmt(v.coverage_target_pct, 0)}% target`)}
    </div>
    <div class="panel-band"><h3>Calibration points</h3>${v.calibration_points
      .map((p) => bar(p.structure, p.absolute_error, 8, "accent"))
      .join("")}</div>
  `;
}

function renderReport() {
  const r = state.assessment.qualification_report;
  return `
    <div class="tab-header"><p class="eyebrow">Qualification report export</p><h2>${r.title}</h2><p>Ready to become PDF/HTML evidence for design reviews.</p></div>
    <div class="report-sheet">
      <h3>Executive decision: ${statusPill(r.executive_decision)}</h3>
      <p><strong>Coupon:</strong> ${clean(r.coupon_configuration.structure)} | ${clean(r.coupon_configuration.ink)} | ${clean(r.coupon_configuration.test_method)}</p>
      <h3>Key results</h3>
      <ul>${r.key_results.map((line) => `<li>${line}</li>`).join("")}</ul>
      <h3>Decision reasons</h3>
      <ul>${r.decision_reasons.map((line) => `<li>${line}</li>`).join("")}</ul>
      <h3>Recommended next action</h3>
      <p>${r.recommended_next_action ? `${clean(r.recommended_next_action.coupon_structure)} at ${fmt(r.recommended_next_action.condition.aging_temp_c, 0)} C` : "No next action available."}</p>
    </div>
  `;
}

function renderPlugins() {
  const p = state.assessment.plugin_registry;
  return `
    <div class="tab-header"><p class="eyebrow">Failure model plugin interface</p><h2>Open architecture for lab-calibrated physics</h2><p>Contract: ${p.plugin_contract.inputs.join(" + ")} -> ${p.plugin_contract.outputs.join(" + ")}</p></div>
    <div class="analysis-grid two">
      <div class="panel-band"><h3>Candidate plugins</h3>${p.candidate_plugins.map((item) => `<p class="action-line">${item}</p>`).join("")}</div>
      <div class="panel-band"><h3>Registered model families</h3>${p.models.map((m) => `<p class="action-line"><strong>${m.family}</strong> - ${m.name}</p>`).join("")}</div>
    </div>
  `;
}

function bindEvents() {
  $("materialSet").addEventListener("change", syncInkToMaterial);
  document.querySelectorAll("input, select").forEach((input) => {
    input.addEventListener("input", () => {
      syncLabels();
      if (input.id === "analysisZone") renderCouponSvg();
    });
  });
  $("runAssessment").addEventListener("click", () => runAssessment().catch(showError));
}

function showError(error) {
  console.error(error);
  $("tabPanel").innerHTML = `<div class="empty-state error">${String(error.message || error).slice(0, 320)}</div>`;
}

async function boot() {
  const health = await fetch("/api/health").then((r) => r.json());
  $("healthDot").classList.toggle("ok", Boolean(health.ok));
  $("healthText").textContent = health.ok ? "Models loaded" : "Training artifacts missing";
  state.meta = await fetch("/api/metadata").then((r) => r.json());
  populateControls();
  bindEvents();
  renderTabs();
  await runAssessment();
}

boot().catch(showError);
