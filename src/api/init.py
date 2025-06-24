from flask import Blueprint, request, jsonify
from pydantic import BaseModel, ValidationError

bp = Blueprint("init", __name__)

class InitPayload(BaseModel):
  id: str

@bp.route("/init", methods=["POST"])
def init():
  try:
    payload = InitPayload(**request.get_json())
  except ValidationError as e:
    return jsonify({"Error": e.errors()}), 400

  # TODO(slok): Initialize the user with device id

  return jsonify({"Msg": "Sucess"}), 200
