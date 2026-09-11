Physical constants
==================

``veris.physical_constants.PHYSICALCONSTANTS`` defines physical and empirical
coefficients, physical bounds, regularization scales and forcing reference
heights for the frozen ``PhysicalConstants`` dataclass. Each model
initialization constructs its own constants object. Physical constants stay
outside the numerical State PyTree.

Override independent values with ``dataclasses.replace``. Exact dependencies,
such as density ratios, follow the changed base values::

   from dataclasses import replace
   from veris.physical_constants import PhysicalConstants
   constants = replace(PhysicalConstants(), rhoIce=920.0)
   assert constants.rhoIce2rhoSnow == 920.0 / constants.rhoSnow

Independently rounded coefficients in established parameterizations retain
their numerical defaults. Similar names do not imply interchangeable formulas.
Ice-removal thresholds such as ``hIce_min``, basal-drag thresholds such as
``basalDragMinArea``, and bulk-flux wind floors belong here because they define
the physical parameterizations. Override them through ``physical_overrides``
at initialization, or replace fields on ``PhysicalConstants``.

``pressReplFac`` weights the strain-dependent replacement pressure in the ice
constitutive law. Its default remains 1.0; fractional values retain their
continuous weighting behavior. Configure it through ``physical_overrides``
or immutable replacement of ``PhysicalConstants``.

``evpStressRelaxation`` and ``evpShearRelaxation`` also belong here: the first
scales the equilibrium stresses, while their ratio scales equilibrium
deviatoric stresses. They retain independent defaults. The numerical EVP
relaxation rates ``evpAlpha`` and ``evpBeta`` remain model settings.

Registry defaults
-----------------

.. exec::

   from veris.physical_constants import PHYSICALCONSTANTS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Constant")
   print("     - Default")
   print("     - Type")
   print("     - Description")
   for name, metadata in PHYSICALCONSTANTS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - ``{metadata.type.__name__}``")
       print(f"     - {metadata.description}")
