import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import hashlib

# We use the internal docker network address for Postgres
# since this will run inside the Airflow container.
