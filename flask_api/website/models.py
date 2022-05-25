from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from . import db
from pathlib import Path
import sys


# db_path = f'{Path(sys.path[0]).parents[0]}/sql/model_params.sqlite'
# engine = create_engine(f'sqlite:///{db_path}', echo=True)
#
# Base = declarative_base(engine)

class Model_params(db.Model):
    __tablename__ = 'Model'
    __table_args__ = {'autoload':True,'autoload_with':db.engine}



