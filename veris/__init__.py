from veris.model import model
from veris.set_inits import set_inits
from veris.variables import VARIABLES
from veris.settings import SETTINGS


__VEROS_INTERFACE__ = dict(
    name = 'veris',
    setup_entrypoint = set_inits,
    run_entrypoint = model,
    settings = SETTINGS,
    variables = VARIABLES,
)