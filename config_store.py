from dataclasses import dataclass, field
from typing import Dict, Any
import json
import os

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

@dataclass
class ConfigStore:
    filepath: str
    config: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.load(self.filepath)

    def load(self, filepath: str) -> None:
        self.filepath = filepath

        # 1) Wenn Datei nicht existiert -> Default erstellen + schreiben
        if not os.path.exists(self.filepath):
            print(f"[ConfigStore] '{self.filepath}' does not exist -> creating default config.")
            self.config = self.default_configuration()
            self.ensure_dicts()
            self.save()
            return

        # 2) Datei existiert -> JSON laden
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Falls JSON nicht das erwartete Format hat (muss dict sein)
            if not isinstance(data, dict):
                raise ValueError("JSON root must be an object/dict.")

            self.config = data
            self.ensure_dicts()

        except json.JSONDecodeError as e:
            # Datei existiert, aber ist kein gültiges JSON -> Default + schreiben
            print(f"[ConfigStore] Invalid JSON in '{self.filepath}' ({e}). Recreating default config.")
            self.config = self.default_configuration()
            self.ensure_dicts()
            self.save()

        except Exception as e:
            # Sonstiger Fehler (z.B. Berechtigung) -> Default im Speicher, aber nicht blind schreiben
            print(f"[ConfigStore] Failed to load '{self.filepath}': {e}")
            self.config = self.default_configuration()
            self.ensure_dicts()

    def save(self) -> None:
        # Ordner anlegen falls nötig (z.B. configs/config.json)
        parent = os.path.dirname(self.filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False, sort_keys=False)

    def clear(self) -> None:
        # Reset auf Default und speichern
        self.config = self.default_configuration()
        self.ensure_dicts()
        self.save()

    def ensure_dicts(self) -> None:
        """
        Stellt sicher, dass die Config zur erwarteten Struktur passt:
        - Fehlende Keys werden ergänzt
        - Falsche Typen werden auf Default zurückgesetzt
        """
        default = self.default_configuration()

        if not isinstance(self.config, dict):
            self.config = {}

        # --- helper: typ-sicher setdefault ---
        def ensure_key(key: str, expected_type, default_value):
            if key not in self.config or not isinstance(self.config[key], expected_type):
                self.config[key] = default_value

        # Top-level simple keys
        ensure_key("include_state", bool, default["include_state"])
        ensure_key("include_bias", bool, default["include_bias"])
        ensure_key("trajectory_horizon", int, default["trajectory_horizon"])
        ensure_key("lift_raw_input", bool, default["lift_raw_input"])
        ensure_key("bilinear_zxu", bool, default["bilinear_zxu"])

        # Dict-Sections
        ensure_key("time_delay", dict, default["time_delay"].copy())
        ensure_key("rbf", dict, default["rbf"].copy())
        ensure_key("gates", dict, default["gates"].copy())
        ensure_key("edmdc_params", dict, default["edmdc_params"].copy())

        # List sections
        ensure_key("custom_features", list, list(default["custom_features"]))
        ensure_key("custom_features_signatures", list, list(default["custom_features_signatures"]))

        # --- time_delay subkeys ---
        td = self.config["time_delay"]
        if not isinstance(td.get("enabled"), bool): td["enabled"] = default["time_delay"]["enabled"]
        if not isinstance(td.get("apply_on_x"), bool): td["apply_on_x"] = default["time_delay"]["apply_on_x"]
        if not isinstance(td.get("n_delays"), int): td["n_delays"] = default["time_delay"]["n_delays"]
        if not isinstance(td.get("tau"), int): td["tau"] = default["time_delay"]["tau"]

        # --- rbf subkeys ---
        rbf = self.config["rbf"]
        if not isinstance(rbf.get("enabled"), bool): rbf["enabled"] = default["rbf"]["enabled"]
        if not isinstance(rbf.get("type"), str): rbf["type"] = default["rbf"]["type"]
        if not isinstance(rbf.get("n_centers"), int): rbf["n_centers"] = default["rbf"]["n_centers"]
        if not _is_number(rbf.get("kernel_width")): rbf["kernel_width"] = default["rbf"]["kernel_width"]
        if not _is_number(rbf.get("polyharmonic_coeff")): rbf["polyharmonic_coeff"] = default["rbf"]["polyharmonic_coeff"]
        if not isinstance(rbf.get("centers"), list): rbf["centers"] = list(default["rbf"]["centers"])

        # --- gates subkeys ---
        gates = self.config["gates"]
        if not isinstance(gates.get("enabled"), bool): gates["enabled"] = default["gates"]["enabled"]
        if not isinstance(gates.get("gates_as_features"), bool): gates["gates_as_features"] = default["gates"]["gates_as_features"]
        if not isinstance(gates.get("items"), list): gates["items"] = default["gates"]["items"]

        # --- edmdc_params subkeys ---
        ep = self.config["edmdc_params"]
        if not _is_number(ep.get("lambda_dyn")): ep["lambda_dyn"] = default["edmdc_params"]["lambda_dyn"]
        if not _is_number(ep.get("lambda_dec")): ep["lambda_dec"] = default["edmdc_params"]["lambda_dec"]

    @staticmethod
    def default_configuration() -> Dict[str, Any]:
        """
        Create default JSON Configuration and return it
        """
        return {
            "include_state": True,
            "include_bias": False,
            "time_delay": {
                "enabled": False,
                "apply_on_x": True,
                "n_delays": 0,
                "tau": 1
            },
            "rbf": {
                "enabled": False,
                "type": "gauss",
                "n_centers": 0,
                "kernel_width": 1.0,
                "polyharmonic_coeff": 1.0,
                "centers": []
            },
            "custom_features": [],
            "custom_features_signatures": [],
            "gates": {
                "enabled": False,
                "gates_as_features": False,
                "items": []
            },
            "trajectory_horizon": 100,
            "edmdc_params": {
                "lambda_dyn": 1e-06,
                "lambda_dec": 1e-06
            },
            "lift_raw_input": True,
            "bilinear_zxu": False
        }

if __name__ == "__main__":
    # Mini-Selftest
    configStore = ConfigStore("config.json")
    print(configStore.config)
