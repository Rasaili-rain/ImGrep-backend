from flask import Blueprint, request, jsonify
from src.models import db, User

bp = Blueprint('apitest', __name__)

@bp.route('/users', methods=['POST'])
def create_user():
  data = request.get_json()
  username = data.get('username')

  if not username:
    return jsonify({'error': 'Username is required'}), 400

  if User.query.filter_by(username=username).first():
    return jsonify({'error': 'Username already exists'}), 409

  new_user = User(username=username)
  db.session.add(new_user)
  db.session.commit()

  return jsonify({'message': f'User {username} created'}), 201

@bp.route('/users', methods=['GET'])
def get_users():
  users = User.query.all()
  usernames = [user.username for user in users]
  return jsonify(usernames)
