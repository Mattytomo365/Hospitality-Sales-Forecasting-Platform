from pathlib import Path
import json, tempfile
'''
Model manifest orchestration - saving, reading, validation
'''
# custom exceptions for domain-specific signals
class RegistryError(Exception): pass # base exception for registry module
class ModelNotFound(RegistryError): pass # inherits from RegistryError
class ManifestError(RegistryError): pass

registry_dir = Path("models")

# validation for model manifest components
def validate(manifest):
    if manifest.kind not in {"ridge","rf","xgb"}:
        raise ValueError(f"Unsupported kind: {manifest.kind}")
    if manifest.target not in {"sales","covers"}:
        raise ValueError(f"Unsupported target: {manifest.target}")
    if not manifest.features:
        raise ValueError("features must be non-empty")
    # ensure metrics contain MAE/MAPE
    for k in ("MAE","MAPE"):
        if k not in manifest.metrics:
            raise KeyError(f"metrics missing '{k}'")

# Writes model manifest JSON (Human-readable model metadata for frontend use)
def save_manifest(model_id, target, features, best_params, metrics, t, out): 
    manifest = {
        "model_id": model_id,
        "kind": metrics["kind"],
        "target": target,
        "features": features,
        "params": best_params,
        "metrics": {metrics["MAE"], metrics["MAPE"], metrics["RMSE"], metrics["R2"]},
        "created_at": t
    }
    validate(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return str(out)

# Reads manifest JSON file - used for displaying model information to user
def read_manifest(model_id):
    path = registry_dir / model_id / "manifest.json" # joining path segments

    if not path.exists():
        raise ModelNotFound(f"Manifest not found for model_id={model_id}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e: # if deserialised data is invalid
        raise ManifestError(f"Invalid JSON in {path}: {e}") from e
    return data

def list_models():
    return
