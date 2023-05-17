def model(state):
    """
    All of the calculations are directly done in the setup file. In Veros, a plug-in
    is called at the end of the time step but a sea ice model needs to be called
    before the ocean model as it modifies the surface forcing (heat, salt and
    momentum flux). veris is still installed as a plug-in so its variables and
    settings are imported and can be used.
    """
    pass
