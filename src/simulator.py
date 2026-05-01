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
        self.initial_wc = initial_wc or InitialWaterContent(value=["FC"])

        # Estado interno
        self._model              = None
        self.current_step        = 0
        self.start_date          = None
        self._daily_log          = []
        self._irrigation_history = []

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
        self._daily_log.clear()
        self._irrigation_history.clear()
        self.start_date = datetime.strptime(sim_start, "%Y/%m/%d")

        # FIX: irrigation_method=5 → el único método que respeta
        # el valor de .depth inyectado manualmente cada paso.
        # method=0 es solo lluvia (ignora depth completamente).
        irr_mgmt = IrrigationManagement(irrigation_method=5, MaxIrr = 50)

        self._model = AquaCropModel(
            sim_start_time=sim_start,
            sim_end_time=sim_end,
            weather_df=self.weather_df,
            soil=self.soil,
            crop=self.crop,
            initial_water_content=self.initial_wc,
            irrigation_management=irr_mgmt,
        )

        self._model._initialize()

        # FIX: No llamar _save_daily_state() aquí.
        # Antes del primer step(), _init_cond no tiene valores reales aún.
        # El historial empieza después del primer step().

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

        # FIX: method=5 lee IrrMngt.depth directamente.
        # Siempre asignamos el valor (incluso 0) para evitar
        # que quede el valor del día anterior.
        self._model._param_struct.IrrMngt.depth = float(irrigation_depth)

        # Avanzar un día — initialize_model=False es obligatorio
        # para continuar desde donde estamos (no reiniciar).
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
            "step":           self.current_step,
            "date":           self.get_current_date(),
            "dap":            int(cond.dap),
            "growing_season": bool(cond.growing_season),
            "growth_stage":   int(cond.growth_stage),

            # Cultivo
            "canopy_cover":   round(float(cond.canopy_cover), 4),
            "biomass":        round(float(cond.biomass), 4),
            "dry_yield":      round(float(cond.DryYield), 4),
            "fresh_yield":    round(float(cond.FreshYield), 4),
            "harvest_index":  round(float(cond.harvest_index), 4),
            "z_root":         round(float(cond.z_root), 3),

            # Agua
            # FIX: taw viene de _init_cond.taw, no de _param_struct.Soil.taw
            "depletion":      round(float(cond.depletion), 2),
            "taw":            round(float(cond.taw), 2),
            "irr_cumulative": round(float(cond.irr_cum), 2),
            "irr_today":      round(float(self._model._param_struct.IrrMngt.depth), 2),
            "et0":            round(float(cond.et0), 3),

            # Estado
            "crop_mature":    bool(cond.crop_mature),
            "crop_dead":      bool(cond.crop_dead),
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

        FIX: la lógica anterior comparaba current_step con len(weather_df),
        lo cual no considera la fecha de fin real ni la cosecha anticipada.
        La fuente correcta es model._clock_struct.model_is_finished.
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

        FIX: la versión original tenía la condición invertida:
        'if not self.is_finished()' mostraba advertencia cuando SÍ
        había terminado y retornaba None cuando NO había terminado.
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