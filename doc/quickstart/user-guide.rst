Basic usage
===========

Install the current checkout into an activated Python environment::

   python -m pip install -e .

Run this example in a fresh Python process. Initialize before importing transport
or dynamics modules so the example can select its serial periodic halo backend::

   import jax
   jax.config.update("jax_enable_x64", True)

   from veris.setup.artificial import initialize, step

   state, settings = initialize(nx=8, ny=12, wind=5.0)
   for _ in range(3):
       state = step(state, settings, cooling=100.0)
   jax.block_until_ready(state)
   print(float(state.hIceMean[2:-2, 2:-2].mean()))

The example uses a periodic Cartesian grid with a central island. Each time step
updates ice velocities and stresses, transports ice and snow, then computes
thermodynamic growth and ocean heat/salt exchange. Cooling is prescribed in
W/m², positive upward. Ocean fields remain prescribed.

State and settings are immutable named tuples. Update them with ``_replace``;
keep timestep reciprocals consistent when changing a timestep::

   settings = settings._replace(
       deltatTherm=300, recip_deltatTherm=1 / 300,
       deltatDyn=300, recip_deltatDyn=1 / 300,
   )

The default five EVP subcycles demonstrate the integration sequence; they do not
establish a converged dynamics solution. See :doc:`/reference/setups/artificial`
for the example API and :doc:`/reference/settings` for kernel defaults.

Development checks
------------------

From the repository root, run the full correctness suite or a deterministic
collection-time sample::

   pytest tests/ -q
   pytest tests/ -q --fast

``VERIS_TEST_SEED`` changes the stable approximately 10% sample. CPU execution is
the default test target; selecting a GPU requires a compatible JAX installation.
