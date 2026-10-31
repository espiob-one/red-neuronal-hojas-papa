"""Background telemetry. PMIC rail sum is partial power, not input power."""
import csv
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil


def parse_pmic(raw):
    values = {name: float(value) for name, value in re.findall(r'(\w+)\s+(?:current|volt)\(\d+\)=([0-9.]+)[AV]', raw)}
    rails = {name[:-2]: current * values[name[:-2] + '_V'] for name, current in values.items() if name.endswith('_A') and name[:-2] + '_V' in values}
    return (sum(rails.values()) if rails else None), values.get('EXT5V_V'), rails


class Telemetry:
    def __init__(self, state, path):
        self.state = state
        self.path = Path(path)
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.previous = None
        self.energy_wh = 0.0
        self.valid_seconds = 0.0
        self.peak_w = 0.0
        self.samples = 0
        self.latest = {}
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return {**self.latest, 'sample_age_seconds': time.monotonic() - self.latest['monotonic'] if self.latest else None}

    def run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['utc','phase','cpu_percent','process_cpu_percent','ram_used_mib','process_rss_mib','temperature_c','pmic_partial_w','input_voltage_v','pmic_partial_wh','power_error']
        psutil.cpu_percent()
        self.process.cpu_percent()
        with self.path.open('a', newline='', buffering=1) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0:
                writer.writeheader()
            while not self.state.stop.is_set():
                now = time.monotonic()
                power = voltage = None
                error = None
                try:
                    raw = subprocess.run(['vcgencmd','pmic_read_adc'],capture_output=True,text=True,timeout=2,check=True).stdout
                    power, voltage, rails = parse_pmic(raw)
                    if power is None:
                        raise ValueError('No paired PMIC readings')
                except Exception as exc:
                    error = str(exc)
                if power is not None:
                    if self.previous is not None:
                        dt = now - self.previous[0]
                        if dt <= 6:
                            self.energy_wh += (power + self.previous[1]) * 0.5 * dt / 3600
                            self.valid_seconds += dt
                    self.previous = (now, power)
                    self.peak_w = max(self.peak_w, power)
                    self.samples += 1
                else:
                    self.previous = None
                try:
                    temp = float(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
                except (OSError, ValueError):
                    temp = None
                memory = psutil.virtual_memory()
                row = dict(utc=datetime.now(timezone.utc).isoformat(),phase=self.state.phase,cpu_percent=psutil.cpu_percent(),process_cpu_percent=self.process.cpu_percent(),ram_used_mib=round((memory.total-memory.available)/2**20,1),process_rss_mib=round(self.process.memory_info().rss/2**20,1),temperature_c=temp,pmic_partial_w=round(power,3) if power is not None else None,input_voltage_v=voltage,pmic_partial_wh=round(self.energy_wh,6),power_error=error)
                writer.writerow(row)
                with self.lock:
                    self.latest = {**row,'monotonic':now,'elapsed_seconds':round(now-self.started,1),'power_coverage_seconds':round(self.valid_seconds,1),'pmic_mean_w':round(self.energy_wh*3600/self.valid_seconds,3) if self.valid_seconds else None,'pmic_peak_w':round(self.peak_w,3),'power_scope':'partial_pmic_rails','csv_file':str(self.path)}
                self.state.stop.wait(2)
