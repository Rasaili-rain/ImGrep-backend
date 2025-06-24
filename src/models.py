import uuid
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

db = SQLAlchemy()

"""
  NOTE(slok): We're storing the embeddings as 512d vector in postgress.
  Here we're utilizing postgress vector searching extension
  If there occurs some sort of bottleneck with huge file search,
  then its better to switch to faiss database

 +--------------------+
 |      Images        |
 +---------+----------+
 | image_id | user_id |
 +---------+----------+

 +----------------------+
 |   Image Embeddings   |
 +----------+-----------+
 | image_id | embedding |
 +----------+-----------+

 +-----------------------+
 |     OCR Embeddings    |
 +----------+------------+
 | image_id | embeddings |
 +----------+------------+

 +-----------------------+
 |  Metadata Embeddings  |
 +----------+------------+
 | image_id | embeddings |
 +----------+------------+
"""

class User(db.Model):
  __tablename__ = "Users"

  id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

  # This can be mac address or android id. But should be unique to identify other devices
  device_id = db.Column(db.String(32), unique=True, nullable=False)

  # TODO(slok): Add other fields like email and stuff when login is introduced

  def __repr__(self):
    return f"<User {self.id}>"


class Image(db.Model):
  __tablename__ = "Images"

  image_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  user_id = db.Column(UUID(as_uuid=True), db.ForeignKey("Users.id"), nullable=False)

  def __repr__(self):
    return f"<Image {self.image_id}>"


class ImageEmbedding(db.Model):
  __tablename__ = "ImageEmbeddings"

  image_id = db.Column(UUID(as_uuid=True), db.ForeignKey("Images.image_id"), primary_key=True)
  embedding = db.Column(Vector(512), nullable=False)

  def __repr__(self):
    return f"<ImgEmb {self.image_id}>"


class OCREmbedding(db.Model):
  __tablename__ = "OCREmbeddings"

  image_id = db.Column(UUID(as_uuid=True), db.ForeignKey("Images.image_id"), primary_key=True)
  embedding = db.Column(Vector(512), nullable=False)

  def __repr__(self):
    return f"<OCREmb {self.image_id}>"


class MetaEmbedding(db.Model):
  __tablename__ = "MetaEmbeddings"

  image_id = db.Column(UUID(as_uuid=True), db.ForeignKey("Images.image_id"), primary_key=True)
  embedding = db.Column(Vector(512), nullable=False)

  def __repr__(self):
    return f"<MetaEmb {self.image_id}>"
