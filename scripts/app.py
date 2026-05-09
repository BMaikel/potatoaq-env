"""
scripts/app.py
Interfaz Gradio para el Simulador de Riego de Papa (AquaCrop)
Compatible con Gradio 6+
"""

import gradio as gr
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator import PotatoSimulator

sim = PotatoSimulator()

# ── Etapas de crecimiento (growth_stage en AquaCrop-OSPy) ─────────
#   0: fuera de temporada / pre-emergencia
#   1: vegetativo         (emergencia → máx. cobertura de canopeo)
#   2: tuberización       (inicio formación de rendimiento / HIstart)
#   3: senescencia        (declive de canopeo → madurez)
#   4: maduración tardía  (post-senescencia / final de temporada)
STAGE_NAMES = {
    0: "Fuera de temporada",
    1: "Vegetativo",
    2: "Tuberización / Formación de rendimiento",
    3: "Senescencia",
    4: "Maduración",
}

# ── Imágenes por etapa ─────────────────────────────────────────────
# Coloca los archivos en:  <raíz del proyecto>/assets/stage_N.png
# donde N va de 0 a 4. El componente gr.Image mostrará None
# (espacio en blanco) si el archivo no existe todavía.
def _img_path(stage: int):
    base = os.path.join(os.path.dirname(__file__), '..', 'assets')
    p = os.path.join(base, f'stage_{stage}.png')
    return p if os.path.exists(p) else None

# ── Helpers ──────────────────────────────────────────────────────
def unpack(s: dict, msg: str = "", img=None) -> list:
    """Convierte el state dict en lista ordenada para los outputs de Gradio."""
    s  = s or {}
    cc = round(float(s.get("canopy_cover", 0)) * 100, 1)   # fracción → %
    return [
        # barra de tiempo
        int(s.get("step",  0)),
        s.get("date", ""),
        int(s.get("dap",   0)),
        STAGE_NAMES.get(s.get("growth_stage", 0), ""),
        # cultivo
        cc,
        round(float(s.get("biomass",        0)), 3),
        round(float(s.get("dry_yield",      0)), 3),
        round(float(s.get("fresh_yield",    0)), 3),
        round(float(s.get("harvest_index",  0)), 3),
        round(float(s.get("z_root",         0)), 3),
        # agua
        round(float(s.get("et0",            0)), 3),
        round(float(s.get("depletion",      0)), 2),
        round(float(s.get("taw",            0)), 2),
        round(float(s.get("irr_today",      0)), 2),
        round(float(s.get("irr_cumulative", 0)), 2),
        # imagen y mensaje
        img,
        msg,
    ]

# ── Callbacks ────────────────────────────────────────────────────
def start_sim(sim_start: str, sim_end: str, max_riego: float):
    try:
        sim.start(
            sim_start = sim_start.replace("-", "/"),
            sim_end   = sim_end.replace("-", "/"),
            MaxRiego  = float(max_riego),
        )
        img = _img_path(0)   # etapa 0: antes de la emergencia
        return unpack({}, "✅ Simulación iniciada — presiona ▶ Avanzar Día para correr el modelo.", img)
    except Exception as e:
        return unpack({}, f"❌ Error al iniciar: {e}")

def step_sim(irr: float):
    if sim._model is None:
        return unpack({}, "⚠️ Primero inicia la simulación con ▶ Iniciar.")
    if sim.is_finished():
        return unpack(sim.get_current_state(), "🏁 La simulación ha finalizado.")
    try:
        s     = sim.step(float(irr or 0))
        stage = s.get("growth_stage", 0)
        img   = _img_path(stage)

        flags = []
        if s.get("crop_mature"): flags.append("✅ Maduro")
        if s.get("crop_dead"):   flags.append("💀 Muerto")
        extra = "  |  " + "  ".join(flags) if flags else ""

        msg = (
            f"📅 {s['date']}  |  "
            f"DAP {s['dap']}  |  "
            f"{STAGE_NAMES.get(stage, stage)}"
            f"{extra}"
        )
        return unpack(s, msg, img)
    except Exception as e:
        return unpack({}, f"❌ Error: {e}")

def reset_sim():
    sim.reset()
    return unpack({}, "🔄 Simulación reiniciada.", None)

# ── CSS ──────────────────────────────────────────────────────────
# NOTA Gradio 6: css va en launch(), NO en gr.Blocks()
CSS = """
/* Ancho máximo y márgenes laterales */
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 28px !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 22px 12px 14px;
    border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
    margin-bottom: 18px;
}
.app-header h1 {
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.4;
    margin: 0 0 10px;
}
.app-header .authors {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued, #6b7280);
    line-height: 1.8;
}
.app-header .affiliation {
    font-style: italic;
    font-size: 0.72rem;
}

/* Títulos de sección */
.sec-title {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    margin: 0 0 4px !important;
}

/* Parámetros pequeños */
.tiny label > span {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
}
.tiny input[type=number],
.tiny input[type=text] {
    font-size: 0.82rem !important;
    padding: 3px 8px !important;
}

/* Barra de estado */
.status-bar textarea {
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
    resize: none !important;
}
"""

# ── HTML del encabezado ──────────────────────────────────────────
HEADER_HTML = """
<div class="app-header">
  <h1>🥔 Simulador de Riego para Cultivo de Papa basado en AquaCrop</h1>
  <div class="authors">
    Michel Bañares<sup>1</sup>&nbsp; · &nbsp;Rodrigo Céspedes<sup>1</sup>&nbsp; · &nbsp;
    Karen Huamani<sup>1</sup>&nbsp; · &nbsp;Dr. Carlos López de Castilla Vásquez<sup>1,*</sup>
    <br>
    <span class="affiliation">
      <sup>1</sup> Semillero de Investigación en Modelamiento de Datos Agropecuarios,
      Departamento de Estadística e Informática,<br>
      Facultad de Economía y Planificación — Universidad Nacional Agraria La Molina, Lima, Perú
      &nbsp;|&nbsp; <sup>*</sup> Asesor de investigación
    </span>
  </div>
</div>
"""

# ── UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="Simulador Papa — AquaCrop") as demo:

    gr.HTML(HEADER_HTML)

    # ── Fila 1: Configuración + Control de riego ─────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("**⚙️ Parámetros de simulación**")
            with gr.Row():
                inp_start    = gr.Textbox(value="2017-06-14", label="Inicio (YYYY-MM-DD)", scale=2)
                inp_end      = gr.Textbox(value="2017-10-15", label="Fin (YYYY-MM-DD)",    scale=2)
                inp_maxriego = gr.Number( value=50,           label="Riego máx. (mm/día)", scale=1, minimum=0)
            with gr.Row():
                btn_start = gr.Button("▶ Iniciar simulación", variant="primary", scale=3)
                btn_reset = gr.Button("↺ Reiniciar",          variant="stop",    scale=1)

        with gr.Column(scale=2):
            gr.Markdown("**💧 Control de riego diario**")
            inp_irr  = gr.Number(value=0.0, label="Riego a aplicar (mm)", minimum=0)
            btn_step = gr.Button("▶ Avanzar un día", variant="secondary")

    gr.HTML("<hr style='margin:14px 0 10px; border:none; border-top:1px solid var(--border-color-primary,#e5e7eb);'>")

    # ── Fila 2: Barra de tiempo ───────────────────────────────────
    with gr.Row():
        o_step  = gr.Number( label="Paso",             interactive=False, elem_classes="tiny", scale=1)
        o_date  = gr.Textbox(label="Fecha",            interactive=False, elem_classes="tiny", scale=2)
        o_dap   = gr.Number( label="DAP",              interactive=False, elem_classes="tiny", scale=1)
        o_stage = gr.Textbox(label="Etapa fenológica", interactive=False, elem_classes="tiny", scale=3)

    # ── Fila 3: Parámetros ← Imagen → Parámetros ─────────────────
    with gr.Row(equal_height=True):

        # ── Izquierda: Estado del cultivo ─────────────────────────
        with gr.Column(scale=2, min_width=190):
            gr.HTML("<p class='sec-title'>🌱 Estado del Cultivo</p>")
            with gr.Row():
                o_cc = gr.Number(label="Canopeo (%)",         interactive=False, elem_classes="tiny")
                o_bm = gr.Number(label="Biomasa (t/ha)",      interactive=False, elem_classes="tiny")
            with gr.Row():
                o_dy = gr.Number(label="Rdto. seco (t/ha)",   interactive=False, elem_classes="tiny")
                o_fy = gr.Number(label="Rdto. fresco (t/ha)", interactive=False, elem_classes="tiny")
            with gr.Row():
                o_hi = gr.Number(label="Índice cosecha",      interactive=False, elem_classes="tiny")
                o_zr = gr.Number(label="Prof. raíz (m)",      interactive=False, elem_classes="tiny")

        # ── Centro: Imagen del cultivo ────────────────────────────
        # Archivos esperados en assets/ (relativo a la raíz del proyecto):
        #   stage_0.png → Fuera de temporada / pre-emergencia
        #   stage_1.png → Vegetativo
        #   stage_2.png → Tuberización / Formación de rendimiento
        #   stage_3.png → Senescencia
        #   stage_4.png → Maduración
        with gr.Column(scale=3, min_width=260):
            o_img = gr.Image(
                label="Estado Fenológico del Cultivo",
                interactive=False,
                height=270,
            )

        # ── Derecha: Estado hídrico ───────────────────────────────
        with gr.Column(scale=2, min_width=190):
            gr.HTML("<p class='sec-title'>💧 Estado Hídrico</p>")
            with gr.Row():
                o_et0  = gr.Number(label="ET₀ (mm/día)",        interactive=False, elem_classes="tiny")
                o_dep  = gr.Number(label="Depleción (mm)",       interactive=False, elem_classes="tiny")
            with gr.Row():
                o_taw  = gr.Number(label="TAW (mm)",             interactive=False, elem_classes="tiny")
                o_irr  = gr.Number(label="Riego hoy (mm)",       interactive=False, elem_classes="tiny")
            with gr.Row():
                o_irrc = gr.Number(label="Riego acumulado (mm)", interactive=False, elem_classes="tiny")

    # ── Barra de estado ───────────────────────────────────────────
    o_msg = gr.Textbox(
        show_label=False,
        interactive=False,
        placeholder="El estado de la simulación aparecerá aquí…",
        elem_classes="status-bar",
        lines=1,
    )

    # ── Lista de outputs (orden igual al return de unpack()) ──────
    ALL_OUTPUTS = [
        o_step, o_date, o_dap, o_stage,          # tiempo
        o_cc, o_bm, o_dy, o_fy, o_hi, o_zr,      # cultivo
        o_et0, o_dep, o_taw, o_irr, o_irrc,       # agua
        o_img,                                     # imagen
        o_msg,                                     # mensaje
    ]

    # ── Conexiones ────────────────────────────────────────────────
    btn_start.click(fn=start_sim, inputs=[inp_start, inp_end, inp_maxriego], outputs=ALL_OUTPUTS)
    btn_step .click(fn=step_sim,  inputs=[inp_irr],                          outputs=ALL_OUTPUTS)
    btn_reset.click(fn=reset_sim, inputs=[],                                 outputs=ALL_OUTPUTS)


if __name__ == "__main__":
    demo.launch(css=CSS)   # ← Gradio 6: css va aquí, no en gr.Blocks()