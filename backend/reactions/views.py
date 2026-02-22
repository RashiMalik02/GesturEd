# reactions/views.py
# Place in: backend/reactions/views.py — REPLACE existing file entirely.

import json
import time

from django.core.cache import cache
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import stream_state
from .opencv_handler import get_latest_frame, start_lab, stop_lab

CHEMICALS = {
    "HCl":        {"label": "Hydrochloric Acid",   "type": "acid",    "formula": "HCl"},
    "H2SO4":      {"label": "Sulfuric Acid",        "type": "acid",    "formula": "H₂SO₄"},
    "HNO3":       {"label": "Nitric Acid",          "type": "acid",    "formula": "HNO₃"},
    "CitricAcid": {"label": "Citric Acid",          "type": "acid",    "formula": "C₆H₈O₇"},
    "AceticAcid": {"label": "Acetic Acid",          "type": "acid",    "formula": "CH₃COOH"},
    "NaOH":       {"label": "Sodium Hydroxide",     "type": "base",    "formula": "NaOH"},
    "KOH":        {"label": "Potassium Hydroxide",  "type": "base",    "formula": "KOH"},
    "NH3":        {"label": "Ammonia Solution",     "type": "base",    "formula": "NH₃"},
    "CaOH2":      {"label": "Calcium Hydroxide",    "type": "base",    "formula": "Ca(OH)₂"},
    "NaHCO3":     {"label": "Baking Soda",          "type": "base",    "formula": "NaHCO₃"},
    "Water":      {"label": "Distilled Water",      "type": "neutral", "formula": "H₂O"},
    "NaClSol":    {"label": "Saline Solution",      "type": "neutral", "formula": "NaCl(aq)"},
    "SugarSol":   {"label": "Sugar Solution",       "type": "neutral", "formula": "C₁₂H₂₂O₁₁(aq)"},
}

CACHE_KEY_CHEMICAL      = "active_chemical_type"
CACHE_KEY_CHEMICAL_META = "active_chemical_meta"
CACHE_KEY_REACTION_DONE = "reaction_complete_flag"
CACHE_TIMEOUT           = 3600


def _mjpeg_generator():
    while True:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)


# ── Reaction endpoints ────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def start_reaction_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    try:
        data          = json.loads(request.body)
        reaction_type = data.get("reaction_type", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    if reaction_type not in {"red_litmus", "blue_litmus"}:
        return JsonResponse({"error": "Invalid reaction_type."}, status=400)

    request.session["active_reaction"] = reaction_type
    stream_state.set_reaction(reaction_type)
    stream_state.state["chemical_id"]   = None
    stream_state.state["chemical_type"] = "neutral"

    cache.delete(CACHE_KEY_CHEMICAL)
    cache.delete(CACHE_KEY_CHEMICAL_META)
    cache.set(CACHE_KEY_REACTION_DONE, False, timeout=CACHE_TIMEOUT)

    start_lab()   # ← opens camera only now

    return JsonResponse({"message": "Reaction started.", "active_reaction": reaction_type})


@csrf_exempt
@require_http_methods(["POST"])
def stop_reaction_view(request):
    cleared = "active_reaction" in request.session
    request.session.pop("active_reaction", None)

    stop_lab()    # ← releases camera

    stream_state.state["reaction_type"]  = None
    stream_state.state["chemical_id"]    = None
    stream_state.state["chemical_type"]  = "neutral"
    cache.delete(CACHE_KEY_CHEMICAL)
    cache.delete(CACHE_KEY_CHEMICAL_META)
    cache.set(CACHE_KEY_REACTION_DONE, False, timeout=CACHE_TIMEOUT)
    return JsonResponse({"message": "Reaction stopped.", "cleared": cleared})


@require_http_methods(["GET"])
def current_reaction_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    active = request.session.get("active_reaction")
    return JsonResponse({"active_reaction": active, "is_running": active is not None})


@require_http_methods(["GET"])
def video_feed_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    if not request.session.get("active_reaction"):
        return JsonResponse({"error": "No active reaction. Call /start/ first."}, status=400)
    return StreamingHttpResponse(
        _mjpeg_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Chemical endpoints ────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def chemicals_view(request):
    payload = [{"id": cid, "label": m["label"]} for cid, m in CHEMICALS.items()]
    return JsonResponse({"chemicals": payload})


@csrf_exempt
@require_http_methods(["POST"])
def set_chemical_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    try:
        data        = json.loads(request.body)
        chemical_id = data.get("chemical_id", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    if chemical_id not in CHEMICALS:
        return JsonResponse({"error": "Unknown chemical."}, status=400)

    meta = CHEMICALS[chemical_id]
    stream_state.set_chemical(chemical_id)
    cache.set(CACHE_KEY_CHEMICAL,      meta["type"], timeout=CACHE_TIMEOUT)
    cache.set(CACHE_KEY_CHEMICAL_META, {
        "id": chemical_id, "label": meta["label"],
        "type": meta["type"], "formula": meta["formula"],
    }, timeout=CACHE_TIMEOUT)
    cache.set(CACHE_KEY_REACTION_DONE, False, timeout=CACHE_TIMEOUT)

    return JsonResponse({
        "message": f"Chemical set to {chemical_id}.",
        "chemical": {
            "id": chemical_id, "label": meta["label"],
            "type": meta["type"], "formula": meta["formula"],
        },
    })


# ── Status endpoint ───────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def status_view(request):
    return JsonResponse({
        "complete":      cache.get(CACHE_KEY_REACTION_DONE, False),
        "chemical":      cache.get(CACHE_KEY_CHEMICAL_META),
        "reaction_type": request.session.get("active_reaction"),
    })