import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import defaultdict as dd
import multiprocessing as mp
from fpdf import FPDF
from copy import deepcopy
from datetime import timedelta
from datetime import datetime as dt
from zoneinfo import ZoneInfo
from tqdm import tqdm
import copy

import yfinance as yf

import warnings

warnings.filterwarnings("ignore")
