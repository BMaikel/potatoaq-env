import pandas as pd
from datetime import datetime
from aquacrop import AquaCropModel, InitialWaterContent, IrrigationManagement


class PotatoSimulator:
    """
    Envuelve AquaCrop para simular el cultivo de papa paso a paso.
    El usuario aplica riego manualmente cada día llamando a step().
    """
    def __init__(self, weather_df, soil, crop, initial_wc=None):
        """
        Parámetros
        ----------
        weather_df  : DataFrame  preparado con prepare_weather()
        soil        : Soil       objeto AquaCrop calibrado
        crop        : Crop       objeto AquaCrop calibrado (papa)
        initial_wc  : InitialWaterContent  (por defecto: Capacidad de Campo)
        """
        self.weather_df = weather_df
        self.soil       = soil
        self.crop       = crop
        self.initial_wc = initial_wc or InitialWaterContent(wc_type='Pct', value=[50])

        # Estado interno
        self.current_step        = 0      # Dia actual
        self.start_date          = None   # Guarda el día de inicio

        self._model              = None   # Guarda el simulador
        self._daily_log          = []     # Guarda parámetros diarios del simulador
        self._irrigation_history = []     # Guarda el historial de riego

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def start(self, sim_start: str = "2017/06/14", sim_end: str = "2017/09/30"):
        """
        Inicializa la simulación. Debe llamarse antes de step().

        Parámetros
        ----------
        sim_start : str  "YYYY/MM/DD"
        sim_end   : str  "YYYY/MM/DD"
        """
        self.current_step = 0
        self.start_date   = datetime.strptime(sim_start, "%Y/%m/%d")

        self._daily_log.clear()
        self._irrigation_history.clear()

        irr_mgmt = IrrigationManagement(irrigation_method=5, MaxIrr = 50)  # Riego = min(depth, 50)

        self._model = AquaCropModel(
            sim_start_time        = sim_start,
            sim_end_time          = sim_end,
            weather_df            = self.weather_df,
            soil                  = self.soil,
            crop                  = self.crop,
            initial_water_content = self.initial_wc,
            irrigation_management = irr_mgmt,
        )

        self._model._initialize()

        print(f"Simulación iniciada → {sim_start} hasta {sim_end}")

    def step(self, irrigation_depth: float = 0.0) -> dict:
        """
        Avanza un día y aplica riego si se indica.

        Parámetros
        ----------
        irrigation_depth : float  agua a aplicar en mm (0 = no regar)

        Retorna
        -------
        dict con el estado del cultivo al final del día
        """
        if self._model is None:
            raise RuntimeError("Llama a start() antes de step().")
        if self.is_finished():
            return self.get_current_state()

        self._model._param_struct.IrrMngt.depth = float(irrigation_depth)

        self._model.run_model(num_steps=1, initialize_model=False)
        self.current_step += 1

        # Registrar riego si hubo
        if irrigation_depth > 0:
            self._irrigation_history.append({
                "step":     self.current_step,
                "date":     self.get_current_date(),
                "depth_mm": irrigation_depth,
            })

        state = self._save_daily_state()
        return state

    def stop(self):
        """Detiene la simulación antes de la cosecha."""
        # Forzamos la bandera interna del modelo
        self._model._clock_struct.model_is_finished = True
        print(f"Simulación detenida manualmente en el paso {self.current_step}.")

    def reset(self):
        """Reinicia todo. Llama a start() de nuevo para comenzar."""
        self._model     = None
        self.start_date = None
        self.current_step = 0
        self._daily_log.clear()
        self._irrigation_history.clear()

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def _save_daily_state(self) -> dict:
        """
        Lee _init_cond (fuente correcta para variables diarias)
        y guarda el estado en el historial.

        FIX: _outputs.water_flux es un numpy array de forma (N, 16),
        NO un diccionario. No tiene método .get().
        Toda la información diaria está en _init_cond.
        """
        cond = self._model._init_cond

        state = {
            # Tiempo
            "step":           self.current_step,         # Nro de paso         
            "date":           self.get_current_date(),   # Fecha
            "dap":            int(cond.dap),             # Dias después de siembra
            "growing_season": bool(cond.growing_season), 
            "growth_stage":   int(cond.growth_stage),    # Etapa de crecimiento

            # Cultivo
            "canopy_cover":   round(float(cond.canopy_cover), 4),      # Covertura del dosel
            "biomass":        round(float(cond.biomass), 4),
            "dry_yield":      round(float(cond.DryYield), 4),
            "fresh_yield":    round(float(cond.FreshYield), 4),
            "harvest_index":  round(float(cond.harvest_index), 4),     # Índice de Cosecha
            "z_root":         round(float(cond.z_root), 3),

            # Agua
            "depletion":      round(float(cond.depletion), 2),         # Agua que falta para volver a máxima capacidad de campo
            "taw":            round(float(cond.taw), 2),               # Cantidad Máxima de agua que el suelo puede soportar
            "irr_cumulative": round(float(cond.irr_cum), 2),           # Riego acumulado
            "irr_today":      round(float(self._model._param_struct.IrrMngt.depth), 2),  # Riego hoy
            "et0":            round(float(cond.et0), 3),    # Evapotranspiración de referencia

            # Estado
            "crop_mature":    bool(cond.crop_mature),       # Cultivo Maduro
            "crop_dead":      bool(cond.crop_dead),         # Cultivo está muerto
        }

        self._daily_log.append(state)
        return state

    def get_current_state(self) -> dict:
        """Estado del último día simulado."""
        if not self._daily_log:
            return {}
        return self._daily_log[-1]

    def get_current_date(self) -> str:
        """Fecha del día actual en formato YYYY-MM-DD."""
        if self.start_date is None:
            return ""
        return (self.start_date + pd.Timedelta(days=self.current_step)).strftime("%Y-%m-%d")

    def is_finished(self) -> bool:
        """
        True si la simulación terminó (cosecha o fin de período).
        """
        if self._model is None:
            return True
        return bool(self._model._clock_struct.model_is_finished)

    # ------------------------------------------------------------------
    # Resultados finales
    # ------------------------------------------------------------------

    def get_final_results(self) -> dict:
        """
        Resultados al terminar la simulación.
        """
        if not self._daily_log:
            return {"summary": {}, "history": pd.DataFrame()}

        if not self.is_finished():
            print("Advertencia: la simulación aún no ha terminado. Resultados parciales.")

        last  = self._daily_log[-1]
        total_irr = sum(e["depth_mm"] for e in self._irrigation_history)

        summary = {
            "total_steps":          self.current_step,
            "final_date":           last["date"],
            "dry_yield_t_ha":       last["dry_yield"],
            "fresh_yield_t_ha":     last["fresh_yield"],
            "total_irrigation_mm":  round(total_irr, 1),
            "irrigation_events":    len(self._irrigation_history),
            "water_use_efficiency": round((last["dry_yield"] * 1000) / total_irr, 3)
                                    if total_irr > 0 else 0.0,
            "crop_mature":          last["crop_mature"],
        }

        return {
            "summary":              summary,
            "history":              pd.DataFrame(self._daily_log),
            "irrigation_history":   pd.DataFrame(self._irrigation_history),
        }