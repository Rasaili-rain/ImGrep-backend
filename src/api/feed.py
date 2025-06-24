from flask import Blueprint, request, jsonify, current_app

bp = Blueprint("feed", __name__)

@bp.route("/feed", methods=["POST"])
def feed():
  current_app.logger.info(request.files)

  # TODO(slok): Feed the image to the model and store in the db

  return jsonify({"msg": len(request.files)})
