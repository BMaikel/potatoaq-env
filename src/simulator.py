import os
import pandas as pd
from datetime import datetime
from aquacrop import AquaCropModel, InitialWaterContent, IrrigationManagement, Crop, Soil

class PotatoSimulator:
    """
    Envuelve AquaCrop para simular el cultivo de papa paso a paso.
    El usuario aplica riego manualmente cada día llamando a step().
    """
    def __init__(self, weather_df=None, soil=None, crop=None, initial_wc=None):
        self.weather_df = weather_df if weather_df is not None else PotatoSimulator.BuildWeather()
        self.soil       = soil       if soil       is not None else PotatoSimulator.BuildSoil()
        self.crop       = crop       if crop       is not None else PotatoSimulator.BuildPapa()
        self.initial_wc = initial_wc if initial_wc is not None else InitialWaterContent(value=['FC'])

        # Estado interno
        self.current_step        = 0      # Dia actual
        self.start_date          = None   # Guarda el día de inicio

        self._model              = None   # Guarda el simulador
        self._daily_log          = []     # Guarda parámetros diarios del simulador
        self._irrigation_history = []     # Guarda el historial de riego
    
    @staticmethod
    def BuildWeather():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(BASE_DIR, "..", "data", "datos_clima.csv")
        clima = pd.read_csv(ruta)
        clima["Date"] = pd.to_datetime(clima["Date"])
        return clima
    
    @staticmethod
    def BuildPapa(fecha_siembra = '06/14'):
        potato = Crop('Potato', planting_date=fecha_siembra)
        # Aqui modificar los parámetros del cultivo
        return potato
    
    @staticmethod
    def BuildSoil():
        suelo = Soil(soil_type='SandyLoam')
        # Aqui modificar los parámetros del suelo
        suelo.profile.loc[:, 'th_fc'] = 0.207   # CC = 20.7%
        suelo.profile.loc[:, 'th_wp'] = 0.087   # PMP = 8.7%
        suelo.profile.loc[:, 'th_s']  = 0.453   # Saturación = 45.3%
        suelo.profile.loc[:, 'Ksat']  = 949.4   # mm/día
        return suelo
    
    ## INICIAMOS CON EL MODELO

    def start(self, sim_start: str = "2017/06/14", sim_end: str = "2017/10/15", MaxRiego=50): # Riego = min(irrigation_depth, 50), 2017/09/30
        """
        Inicializa la simulación. Debe llamarse antes de step().

        Parámetros
        ----------
        sim_start : str   "YYYY/MM/DD"
        sim_end   : str   "YYYY/MM/DD"
        MaxRiego  : float "Cantidad de riego máxima que aceptará esta simulación"
        """
        VERSION = "0.1.0-dev"

        self.current_step = 0
        self.start_date   = datetime.strptime(sim_start, "%Y/%m/%d")

        end_date          = datetime.strptime(sim_end,   "%Y/%m/%d")
        total_days        = (end_date - self.start_date).days

        self._daily_log.clear()
        self._irrigation_history.clear()

        irr_mgmt = IrrigationManagement(irrigation_method=5, MaxIrr = MaxRiego)

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

        # Condiciones Iniciales del suelo:
        wc_mode  = getattr(self.initial_wc, 'wc_type',  'N/A')
        wc_value = getattr(self.initial_wc, 'value',    'N/A')
        
        soil_name  = getattr(self.soil, 'soil_type', 'N/A')

        print(
            f"\n{'─' * 46}\n"
            f"  SIMULADOR DE PAPA  |  v{VERSION}\n"
            f"{'─' * 46}\n"
            f"  Cultivo          : Papa (Solanum tuberosum)\n"
            f"  Inicio           : {self.start_date.strftime('%d/%m/%Y')}\n"
            f"  Fin              : {end_date.strftime('%d/%m/%Y')}\n"
            f"  Total de días    : {total_days}\n"
            f"  Tipo de suelo    : {soil_name}\n"
            f"  Humedad inicial  : {wc_mode} → {wc_value}\n"
            f"  Riego máximo     : {MaxRiego} mm/día\n"
            f"{'─' * 46}\n"
            f"  ⚠️  En desarrollo — resultados no validados\n"
            f"{'─' * 46}\n")

    def step(self, irrigation_depth: float = 0.0) -> dict:
        """
        Avanza un día y aplica riego si se indica.

        Parámetros
        ----------
        irrigation_depth : float  agua a aplicar en mm (0 = no regar)
        """
        if irrigation_depth < 0:
            raise ValueError(f"irrigation_depth no puede ser negativo (recibido: {irrigation_depth})")
        if self._model is None:
            raise RuntimeError("Llama a start() antes de step().")
        if self.is_finished():
            print("Advertencia: la simulación ya terminó. Llama a reset() para reiniciar.")
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
        if self._model is not None and not self.is_finished():
            self.stop()
        self._model       = None
        self.start_date   = None
        self.current_step = 0
        self._daily_log.clear()
        self._irrigation_history.clear()

    def _save_daily_state(self) -> dict:
        """Obtiene los estados diarios que devuelve el simulador y los guarda"""
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
            "et0":            round(float(cond.et0), 3),
            "depletion":      round(float(cond.depletion), 2),
            "taw":            round(float(cond.taw), 2),
            "irr_today":      round(float(self._model._param_struct.IrrMngt.depth), 2),
            "irr_cumulative": round(float(cond.irr_cum), 2),

            # Flags
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
        """
        if self._model is None:
            return True
        return bool(self._model._clock_struct.model_is_finished)

    def get_final_results(self) -> dict:
        """Devuelve los resultados finales de la simulación"""
        if not self._daily_log:
            return {"summary": {}, "history": pd.DataFrame(), "irrigation_history": pd.DataFrame()}

        if not self.is_finished():
            print("Advertencia: la simulación aún no ha terminado. Resultados parciales.")

        last      = self._daily_log[-1]
        total_irr = sum(e["depth_mm"] for e in self._irrigation_history)

        summary = {
            "total_steps":          self.current_step,
            "final_date":           last["date"],
            "dry_yield_t_ha":       last["dry_yield"],
            "fresh_yield_t_ha":     last["fresh_yield"],
            "total_irrigation_mm":  round(total_irr, 1),
            "irrigation_events":    len(self._irrigation_history),
            "water_use_efficiency": round(last["dry_yield"] * 1000 / total_irr, 3) if total_irr > 0 else 0.0,
            "crop_mature":          last["crop_mature"],
        }

        return {
            "summary":            summary,
            "history":            pd.DataFrame(self._daily_log),
            "irrigation_history": pd.DataFrame(self._irrigation_history),
        }