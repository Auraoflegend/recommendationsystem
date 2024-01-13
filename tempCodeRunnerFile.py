import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

def convert_views_to_numeric(views):
    if 'K' in views:
        return int(float(views.replace('K', '')) * 1000)