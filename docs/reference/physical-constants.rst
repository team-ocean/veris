Physical constants
==================

``veris.model.physical_constants.PHYSICALCONSTANTS`` defines the values used by
the frozen ``PhysicalConstants`` dataclass, a collection whose fields cannot be
reassigned. These include physical coefficients (numbers used in equations), coefficients based on
observations, physical limits, scales that keep equations well behaved, and
reference heights for atmospheric forcing, the supplied air conditions. Each model initialization creates
its own constants object. Physical constants stay outside the numerical State
PyTree, the collection of arrays used by JAX.

Change independent values with ``dataclasses.replace``. Values calculated from
exact relationships, such as density ratios, follow the changed base values::

   from dataclasses import replace
   from veris.model.physical_constants import PhysicalConstants
   constants = replace(PhysicalConstants(), rhoIce=920.0)
   assert constants.rhoIce2rhoSnow == 920.0 / constants.rhoSnow

Parameterizations are formulas that approximate physical processes. Their
independently rounded coefficients keep their numerical defaults. Similar
names do not mean that formulas are interchangeable. Ice-removal thresholds
such as ``hIce_min``, thresholds for drag against the seabed such as
``basalDragMinArea``, and minimum wind speeds in bulk-flux formulas (formulas for
transfer between surface and air) belong here because they define these
physical approximations. Supply ``constants={...}``
in an :doc:`Experiment </api/experiment>`, or replace fields on
``PhysicalConstants``. The low-level
:func:`veris.execution.initialization.initialize` instead uses
``physical_overrides``.

``pressReplFac`` weights the replacement pressure that depends on strain (ice
deformation) in the law relating ice stress to deformation. Its default remains
1.0. Fractional values change that weight continuously. Set it through the same
constants group or by creating an updated constants object.

``evpStressRelaxation`` and ``evpShearRelaxation`` also belong here. The first
scales the equilibrium stresses. Their ratio scales the equilibrium deviatoric
stresses, the part of stress that changes shape rather than volume. The two
values keep independent defaults. EVP means elastic-viscous-plastic, the method
used to solve for ice velocity and stress. Its numerical relaxation rates
``evpAlpha`` and ``evpBeta`` remain model settings.

Registry defaults
-----------------

.. exec::

   from veris.model.physical_constants import PHYSICALCONSTANTS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Constant")
   print("     - Default")
   print("     - Unit")
   print("     - Description")
   for name, metadata in PHYSICALCONSTANTS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       units = "``-``" if metadata.units in ("1", "-", "") else metadata.units
       print(f"     - {units}")
       print(f"     - {metadata.description}")
