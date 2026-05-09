import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from simulator import PotatoSimulator
 
sim = PotatoSimulator()
 
# ─────────────────────────────────────────────────────────────────────
# SVG Plants — una por etapa de crecimiento
# ─────────────────────────────────────────────────────────────────────
 
_SOIL = """
  <rect x="0" y="172" width="200" height="58" fill="#7a5230"/>
  <ellipse cx="100" cy="172" rx="100" ry="7" fill="#9a6840"/>
"""
 
def _svg_seed():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  {_SOIL}
  <ellipse cx="100" cy="188" rx="10" ry="7" fill="#c8952a" opacity="0.75"/>
  <line x1="100" y1="172" x2="100" y2="163" stroke="#5a9c55" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
</svg>"""
 
def _svg_vegetative():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  {_SOIL}
  <path d="M97,172 Q93,155 85,135" stroke="#2e7d45" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <path d="M103,172 Q107,153 115,132" stroke="#2e7d45" stroke-width="3" fill="none" stroke-linecap="round"/>
  <ellipse cx="70"  cy="143" rx="20" ry="9"  fill="#4caf50" transform="rotate(-45 70 143)"/>
  <ellipse cx="78"  cy="127" rx="18" ry="8"  fill="#43a047" transform="rotate(-30 78 127)"/>
  <ellipse cx="128" cy="139" rx="20" ry="9"  fill="#4caf50" transform="rotate(45 128 139)"/>
  <ellipse cx="120" cy="124" rx="17" ry="8"  fill="#388e3c" transform="rotate(28 120 124)"/>
  <ellipse cx="85"  cy="130" rx="14" ry="7"  fill="#66bb6a" transform="rotate(-15 85 130)"/>
  <ellipse cx="115" cy="127" rx="13" ry="6"  fill="#57a85d" transform="rotate(15 115 127)"/>
</svg>"""
 
def _svg_flowering():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  {_SOIL}
  <path d="M95,172 Q88,150 78,118"  stroke="#27643a" stroke-width="4"   fill="none" stroke-linecap="round"/>
  <path d="M100,172 Q100,148 100,110" stroke="#27643a" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <path d="M105,172 Q112,150 122,118" stroke="#27643a" stroke-width="4"   fill="none" stroke-linecap="round"/>
  <ellipse cx="57"  cy="140" rx="22" ry="10" fill="#388e3c" transform="rotate(-45 57 140)"/>
  <ellipse cx="143" cy="140" rx="22" ry="10" fill="#2e7d32" transform="rotate(45 143 140)"/>
  <ellipse cx="65"  cy="122" rx="21" ry="9"  fill="#43a047" transform="rotate(-35 65 122)"/>
  <ellipse cx="135" cy="120" rx="21" ry="9"  fill="#388e3c" transform="rotate(35 135 120)"/>
  <ellipse cx="78"  cy="108" rx="18" ry="8"  fill="#4caf50" transform="rotate(-20 78 108)"/>
  <ellipse cx="122" cy="106" rx="18" ry="8"  fill="#43a047" transform="rotate(20 122 106)"/>
  <ellipse cx="100" cy="100" rx="18" ry="8"  fill="#66bb6a"/>
  <g transform="translate(78,112)">  <circle r="7" fill="white" stroke="#9c27b0" stroke-width="1.2"/> <circle r="3" fill="#ffd54f"/> </g>
  <g transform="translate(122,110)"> <circle r="7" fill="white" stroke="#9c27b0" stroke-width="1.2"/> <circle r="3" fill="#ffd54f"/> </g>
  <g transform="translate(100,94)">  <circle r="6" fill="white" stroke="#ab47bc" stroke-width="1"/>   <circle r="2.5" fill="#ffd54f"/> </g>
</svg>"""
 
def _svg_tuberization():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="172" width="200" height="58" fill="#6b4520"/>
  <path d="M97,175 Q88,185 80,198"   stroke="#4a3010" stroke-width="1.5" fill="none" opacity="0.7"/>
  <path d="M100,175 Q100,186 100,202" stroke="#4a3010" stroke-width="1.5" fill="none" opacity="0.7"/>
  <path d="M103,175 Q112,185 120,197" stroke="#4a3010" stroke-width="1.5" fill="none" opacity="0.7"/>
  <ellipse cx="78"  cy="200" rx="17" ry="11" fill="#c9922a"/>
  <ellipse cx="100" cy="206" rx="15" ry="10" fill="#d4a030"/>
  <ellipse cx="122" cy="199" rx="14" ry="10" fill="#c28525"/>
  <ellipse cx="58"  cy="209" rx="11" ry="7"  fill="#b87e20" opacity="0.8"/>
  <ellipse cx="100" cy="172" rx="100" ry="7" fill="#9a6840"/>
  <path d="M92,172  Q85,150 75,118"  stroke="#1b5e30" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M100,172 Q100,148 100,108" stroke="#1b5e30" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M108,172 Q115,150 125,118" stroke="#1b5e30" stroke-width="4" fill="none" stroke-linecap="round"/>
  <ellipse cx="52"  cy="138" rx="24" ry="11" fill="#2e7d32" transform="rotate(-45 52 138)"/>
  <ellipse cx="148" cy="138" rx="24" ry="11" fill="#27692c" transform="rotate(45 148 138)"/>
  <ellipse cx="60"  cy="118" rx="22" ry="10" fill="#388e3c" transform="rotate(-35 60 118)"/>
  <ellipse cx="140" cy="116" rx="22" ry="10" fill="#2e7d32" transform="rotate(35 140 116)"/>
  <ellipse cx="75"  cy="104" rx="20" ry="9"  fill="#43a047" transform="rotate(-20 75 104)"/>
  <ellipse cx="125" cy="102" rx="20" ry="9"  fill="#388e3c" transform="rotate(20 125 102)"/>
  <ellipse cx="100" cy="96"  rx="22" ry="10" fill="#4caf50"/>
  <g transform="translate(75,108)">  <circle r="6" fill="white" stroke="#7b1fa2" stroke-width="1"/> <circle r="2.5" fill="#ffd54f"/> </g>
  <g transform="translate(125,106)"> <circle r="6" fill="white" stroke="#7b1fa2" stroke-width="1"/> <circle r="2.5" fill="#ffd54f"/> </g>
</svg>"""
 
def _svg_senescence():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="172" width="200" height="58" fill="#6b4520"/>
  <path d="M97,175  Q88,185 80,198"   stroke="#4a3010" stroke-width="1.5" fill="none" opacity="0.7"/>
  <path d="M100,175 Q100,186 100,202" stroke="#4a3010" stroke-width="1.5" fill="none" opacity="0.7"/>
  <ellipse cx="80"  cy="200" rx="17" ry="11" fill="#c9922a"/>
  <ellipse cx="105" cy="206" rx="16" ry="11" fill="#d4a030"/>
  <ellipse cx="125" cy="199" rx="14" ry="10" fill="#c28525"/>
  <ellipse cx="100" cy="172" rx="100" ry="7" fill="#9a6840"/>
  <path d="M92,172  Q85,153 78,125"  stroke="#7d6020" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M100,172 Q100,150 100,118" stroke="#7d6020" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M108,172 Q115,153 122,125" stroke="#7d6020" stroke-width="4" fill="none" stroke-linecap="round"/>
  <ellipse cx="56"  cy="143" rx="23" ry="10" fill="#b8a020" transform="rotate(-40 56 143)"/>
  <ellipse cx="144" cy="143" rx="23" ry="10" fill="#a89018" transform="rotate(40 144 143)"/>
  <ellipse cx="64"  cy="123" rx="21" ry="9"  fill="#c8b025" transform="rotate(-30 64 123)"/>
  <ellipse cx="136" cy="121" rx="21" ry="9"  fill="#b8a020" transform="rotate(30 136 121)"/>
  <ellipse cx="78"  cy="110" rx="18" ry="8"  fill="#d4b82a" transform="rotate(-18 78 110)"/>
  <ellipse cx="122" cy="108" rx="18" ry="8"  fill="#c8a820" transform="rotate(18 122 108)"/>
  <ellipse cx="100" cy="104" rx="20" ry="9"  fill="#dcc030"/>
</svg>"""
 
def _svg_mature():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="172" width="200" height="58" fill="#6b4520"/>
  <path d="M95,175  Q84,186 74,201"   stroke="#4a3010" stroke-width="2" fill="none" opacity="0.8"/>
  <path d="M100,175 Q100,187 100,203" stroke="#4a3010" stroke-width="2" fill="none" opacity="0.8"/>
  <path d="M105,175 Q116,186 126,201" stroke="#4a3010" stroke-width="2" fill="none" opacity="0.8"/>
  <ellipse cx="72"  cy="203" rx="20" ry="13" fill="#d4a030"/>
  <ellipse cx="100" cy="208" rx="18" ry="12" fill="#e0b035"/>
  <ellipse cx="128" cy="203" rx="17" ry="12" fill="#d4a030"/>
  <ellipse cx="54"  cy="211" rx="12" ry="8"  fill="#c89025" opacity="0.9"/>
  <ellipse cx="146" cy="209" rx="12" ry="8"  fill="#c89025" opacity="0.9"/>
  <ellipse cx="100" cy="172" rx="100" ry="7" fill="#9a6840"/>
  <path d="M92,172  Q85,152 77,122"  stroke="#8b6914" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M100,172 Q100,150 100,115" stroke="#8b6914" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M108,172 Q115,152 123,122" stroke="#8b6914" stroke-width="4" fill="none" stroke-linecap="round"/>
  <ellipse cx="54"  cy="140" rx="23" ry="10" fill="#c8901a" transform="rotate(-42 54 140)"/>
  <ellipse cx="146" cy="140" rx="23" ry="10" fill="#b87e10" transform="rotate(42 146 140)"/>
  <ellipse cx="63"  cy="120" rx="21" ry="9"  fill="#d4a020" transform="rotate(-30 63 120)"/>
  <ellipse cx="137" cy="118" rx="21" ry="9"  fill="#c89018" transform="rotate(30 137 118)"/>
  <ellipse cx="77"  cy="107" rx="19" ry="8"  fill="#ddb02a" transform="rotate(-18 77 107)"/>
  <ellipse cx="123" cy="105" rx="19" ry="8"  fill="#d4a520" transform="rotate(18 123 105)"/>
  <ellipse cx="100" cy="100" rx="22" ry="10" fill="#e0bc30"/>
</svg>"""
 
def _svg_dead():
    return f"""<svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
  {_SOIL}
  <path d="M92,172  Q90,165 88,158" stroke="#6d4c28" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M100,172 Q100,163 100,155" stroke="#6d4c28" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M108,172 Q110,165 112,158" stroke="#6d4c28" stroke-width="3" fill="none" stroke-linecap="round"/>
  <ellipse cx="80"  cy="165" rx="15" ry="5" fill="#8d6835" transform="rotate(-50 80 165)"  opacity="0.8"/>
  <ellipse cx="100" cy="158" rx="12" ry="4" fill="#7d5828"                                  opacity="0.7"/>
  <ellipse cx="118" cy="162" rx="14" ry="5" fill="#8d6835" transform="rotate(45 118 162)"  opacity="0.8"/>
  <ellipse cx="72"  cy="170" rx="16" ry="4" fill="#6d4c28" transform="rotate(-10 72 170)"  opacity="0.6"/>
  <ellipse cx="128" cy="168" rx="15" ry="4" fill="#7d5828" transform="rotate(15 128 168)"  opacity="0.6"/>
</svg>"""
 
def _get_plant_svg(state: dict) -> str:
    if not state or not state.get("growing_season"):
        return _svg_seed()
    if state.get("crop_dead"):
        return _svg_dead()
    if state.get("crop_mature"):
        return _svg_mature()
    return {
        1: _svg_vegetative,
        2: _svg_flowering,
        3: _svg_tuberization,
        4: _svg_senescence,
    }.get(state.get("growth_stage", 0), _svg_seed)()
 
# ─────────────────────────────────────────────────────────────────────
# Panel HTML central
# ─────────────────────────────────────────────────────────────────────
 
def _card(icon, label, value, accent="#FF7C00"):
    return (
        f'<div style="background:#fff;border-radius:10px;padding:10px 14px;'
        f'border-left:3px solid {accent};box-shadow:0 1px 4px rgba(0,0,0,0.07);margin-bottom:8px;">'
        f'<div style="font-size:10px;color:#9ca3af;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">{icon} {label}</div>'
        f'<div style="font-size:16px;font-weight:700;color:#111827;margin-top:3px;">{value}</div>'
        f'</div>'
    )
 
def _estado_label(state: dict) -> str:
    if state.get("crop_dead"):   return "💀 Muerto"
    if state.get("crop_mature"): return "✅ Maduro"
    return {
        0: "🌑 Sin sembrar",
        1: "🌿 Vegetativo",
        2: "🌸 Floración",
        3: "🥔 Tuberización",
        4: "🍂 Senescencia",
    }.get(state.get("growth_stage", 0), "—")
 
def _build_panel(state: dict) -> str:
    v = "—"
 
    if not state:
        left  = (_card("📅","Fecha",v) + _card("🌱","DAP",v) + _card("🔄","Estado",v) +
                 _card("🍃","Cobertura dosel",v) + _card("⚖️","Biomasa",v) + _card("🥔","Rend. fresco",v))
        right = (_card("💧","Depleción",v,"#3b82f6") + _card("🏺","TAW",v,"#3b82f6") +
                 _card("☀️","ET₀",v,"#f59e0b") + _card("🚿","Riego hoy",v,"#10b981") +
                 _card("📊","Riego acumulado",v,"#10b981"))
        stage_label = "— Listo para iniciar —"
    else:
        left  = (_card("📅","Fecha",          state.get("date","—")) +
                 _card("🌱","DAP",            f"{state.get('dap','—')} días") +
                 _card("🔄","Estado",         _estado_label(state)) +
                 _card("🍃","Cobertura dosel",f"{state.get('canopy_cover',0)*100:.1f} %") +
                 _card("⚖️","Biomasa",        f"{state.get('biomass',0):.3f} t/ha") +
                 _card("🥔","Rend. fresco",   f"{state.get('fresh_yield',0):.3f} t/ha"))
        right = (_card("💧","Depleción",       f"{state.get('depletion',0):.1f} mm",       "#3b82f6") +
                 _card("🏺","TAW",             f"{state.get('taw',0):.1f} mm",             "#3b82f6") +
                 _card("☀️","ET₀",            f"{state.get('et0',0):.3f} mm",             "#f59e0b") +
                 _card("🚿","Riego hoy",       f"{state.get('irr_today',0):.1f} mm",       "#10b981") +
                 _card("📊","Riego acumulado", f"{state.get('irr_cumulative',0):.1f} mm",  "#10b981"))
        stage_label = _estado_label(state)
 
    plant = _get_plant_svg(state)
 
    return (
        '<div style="display:flex;gap:20px;padding:20px;background:#f9fafb;'
        'border-radius:16px;font-family:system-ui,sans-serif;align-items:flex-start;">'
 
        f'<div style="flex:1;display:flex;flex-direction:column;">{left}</div>'
 
        '<div style="flex:0 0 210px;display:flex;flex-direction:column;align-items:center;">'
        '<div style="background:white;border-radius:16px;padding:16px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.09);width:100%;text-align:center;">'
        f'{plant}'
        f'<div style="margin-top:8px;font-size:13px;font-weight:700;color:#374151;">{stage_label}</div>'
        '</div></div>'
 
        f'<div style="flex:1;display:flex;flex-direction:column;">{right}</div>'
 
        '</div>'
    )
 
# ─────────────────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────────────────
 
def _build_charts(log: list[dict]):
    if len(log) < 2:
        return None
 
    df  = pd.DataFrame(log)
    dap = df["dap"]
 
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), facecolor="#f9fafb")
    fig.suptitle("Evolución del cultivo", fontsize=13, fontweight="bold", color="#111827", y=1.01)
 
    plots = [
        (axes[0, 0], "canopy_cover",   lambda y: y * 100, "Cobertura del Dosel (%)",   "#4CAF50"),
        (axes[0, 1], "depletion",      None,              "Depleción del Suelo (mm)",  "#ef4444"),
        (axes[0, 2], "et0",            None,              "ET₀ diaria (mm)",           "#f97316"),
        (axes[1, 0], "biomass",        None,              "Biomasa (t/ha)",            "#8b5cf6"),
        (axes[1, 1], "fresh_yield",    None,              "Rendimiento Fresco (t/ha)", "#3b82f6"),
        (axes[1, 2], "irr_cumulative", None,              "Riego Acumulado (mm)",      "#10b981"),
    ]
 
    for ax, col, transform, title, color in plots:
        y = df[col].apply(transform) if transform else df[col]
        ax.fill_between(dap, y, alpha=0.12, color=color)
        ax.plot(dap, y, color=color, linewidth=2.2, solid_capstyle="round")
        ax.set_title(title, fontsize=9, fontweight="bold", color="#374151")
        ax.set_xlabel("DAP", fontsize=8, color="#6b7280")
        ax.tick_params(labelsize=7, colors="#6b7280")
        ax.set_facecolor("#ffffff")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#e5e7eb")
 
    # TAW como referencia en la gráfica de depleción
    axes[0, 1].plot(dap, df["taw"], color="#94a3b8", linewidth=1.5, linestyle="--", label="TAW")
    axes[0, 1].legend(fontsize=7)
 
    # Marcar días donde se regó
    irr = df[df["irr_today"] > 0]
    if not irr.empty:
        axes[1, 2].scatter(irr["dap"], irr["irr_cumulative"], color="#059669", s=25, zorder=5)
 
    plt.tight_layout(pad=2.0)
    return fig
 
# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────
 
def _to_aquacrop_date(s: str) -> str:
    return s.replace("-", "/")
 
def _refresh(info: str = ""):
    state = sim.get_current_state()
    return _build_panel(state), _build_charts(sim._daily_log), info
 
def cb_iniciar(fecha_inicio, fecha_fin, max_riego):
    try:
        sim.reset()
        sim.start(_to_aquacrop_date(fecha_inicio), _to_aquacrop_date(fecha_fin), int(max_riego))
        return _refresh("✅ Simulación iniciada — presiona **Avanzar día** para comenzar.")
    except Exception as e:
        return _refresh(f"❌ Error al iniciar: {e}")
 
def cb_avanzar(irrigation_mm):
    if sim._model is None:
        return _refresh("⚠️ Primero inicia la simulación.")
    if sim.is_finished():
        return _refresh("🏁 La simulación ya terminó. Usa **Reiniciar** para volver a empezar.")
    sim.step(float(irrigation_mm))
    msg = "🏁 ¡Cultivo finalizado! Revisa los resultados abajo." if sim.is_finished() else ""
    return _refresh(msg)
 
def cb_reset():
    sim.reset()
    return _refresh("🔄 Simulación reiniciada.")
 
# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────
 
with gr.Blocks(title="Simulador de Papa", theme=gr.themes.Default()) as app:
 
    gr.Markdown("# 🌱 Simulador de Riego — Papa")
 
    with gr.Row():
        inp_inicio   = gr.Textbox(label="Fecha inicio",          value="2017-06-14", scale=2)
        inp_fin      = gr.Textbox(label="Fecha fin",             value="2017-09-30", scale=2)
        inp_maxriego = gr.Number (label="Riego máximo (mm/día)", value=50, minimum=1, maximum=200, scale=1)
        btn_iniciar  = gr.Button ("▶ Iniciar",    variant="primary",   scale=1)
        btn_reset    = gr.Button ("🔄 Reiniciar", variant="secondary", scale=1)
 
    out_info  = gr.Markdown("")
    out_panel = gr.HTML(_build_panel({}))
 
    with gr.Row():
        inp_riego   = gr.Slider(minimum=0, maximum=50, step=1, value=0,
                                label="💧 Agua a aplicar hoy (mm)", scale=4)
        btn_avanzar = gr.Button("⏭ Avanzar un día", variant="primary", scale=1)
 
    out_chart = gr.Plot(label="Evolución del cultivo")
 
    outputs = [out_panel, out_chart, out_info]
 
    btn_iniciar.click(fn=cb_iniciar, inputs=[inp_inicio, inp_fin, inp_maxriego], outputs=outputs)
    btn_avanzar.click(fn=cb_avanzar, inputs=[inp_riego],                         outputs=outputs)
    btn_reset.click  (fn=cb_reset,   inputs=[],                                  outputs=outputs)
 
if __name__ == "__main__":
    app.launch()