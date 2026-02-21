import json

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .opencv_handler import generate_frames

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


@csrf_exempt
@require_http_methods(["POST"])
def start_reaction_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)

    try:
        data = json.loads(request.body)
        reaction_type = data.get("reaction_type", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    if not reaction_type:
        return JsonResponse({"error": "reaction_type is required."}, status=400)

    VALID_REACTIONS = {"red_litmus", "blue_litmus"}
    if reaction_type not in VALID_REACTIONS:
        return JsonResponse(
            {"error": f"Invalid reaction_type. Choose from: {', '.join(VALID_REACTIONS)}."},
            status=400,
        )

    request.session["active_reaction"] = reaction_type
    return JsonResponse({"message": "Reaction started.", "active_reaction": reaction_type})


@csrf_exempt
@require_http_methods(["POST"])
def stop_reaction_view(request):
    cleared = False
    if "active_reaction" in request.session:
        del request.session["active_reaction"]
        cleared = True

    return JsonResponse({"message": "Reaction stopped.", "cleared": cleared})


@require_http_methods(["GET"])
def current_reaction_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)

    active_reaction = request.session.get("active_reaction", None)
    return JsonResponse({
        "active_reaction": active_reaction,
        "is_running": active_reaction is not None,
<<<<<<< HEAD
    })


@require_http_methods(["GET"])
def video_feed_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)

    active_reaction = request.session.get("active_reaction")
    if not active_reaction:
        return JsonResponse(
            {"error": "No active reaction selected. Please select a reaction first."},
            status=400,
        )

    return StreamingHttpResponse(
        generate_frames(active_reaction),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
=======
    })
>>>>>>> cbd2dd0 (reaction views)
