"""Standalone Face Morph Studio with automatic startup dependency preparation."""
from .bootstrap import ensure_dependencies

ensure_dependencies()

from .nodes import FaceMorphStudio
from . import api
NODE_CLASS_MAPPINGS = {'DKST_FaceMorphStudio': FaceMorphStudio}
NODE_DISPLAY_NAME_MAPPINGS = {'DKST_FaceMorphStudio': 'DKST Face Morph Studio'}
WEB_DIRECTORY = './web'
__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']
