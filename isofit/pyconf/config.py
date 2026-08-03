from pydantic import BaseModel, Field

from isofit.pyconf.atmosphere import Atmosphere


class Config(BaseModel):
    atmosphere: Atmosphere


# Testing code, ignore
import yaml

data = """\
atmosphere:
  engine:
    name: srtmnet
    base_dir: /Users/jamesmo/projects/isofit/dev/testo
"""
data = yaml.safe_load(data)
cfg = Config.model_validate(data)
cfg.model_dump()

# %%

from isofit.pyconf.atmosphere.engines.sixs import SixS

SixS(**data["atmosphere"]["engine"])
