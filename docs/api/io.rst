Input and output
================

Import the public classes and record functions below from ``veris.io``.
They run in Python on the host that controls the calculation. They stay outside
the model calculations through which JAX computes derivatives. See
:doc:`/reference/io` for complete examples, calendar intervals, sampling rules,
defaults from the model's named tables and limits on reading selected fields.

Output configuration
--------------------

.. autoclass:: veris.io.Stream

.. autoclass:: veris.io.OutputSettings

.. py:data:: veris.io.STREAM_SETTINGS
   :type: dict[str, veris.io.configuration.OutputOption]

   Stream defaults and metadata; see :doc:`/reference/io`.

.. py:data:: veris.io.OUTPUT_SETTINGS
   :type: dict[str, veris.io.configuration.OutputOption]

   OutputSettings defaults and metadata; see :doc:`/reference/io`.

Calendar arithmetic
-------------------

.. autoclass:: veris.io.FixedDate

.. autoclass:: veris.io.Calendar
   :members:

Sampling and snapshots
----------------------

Use OutputManager as a context manager so it closes files and finishes averages
for complete averaging windows. Snapshot writing saves values at a single
instant, independently of the stream schedules. Existing output paths are
rejected.

.. autoclass:: veris.io.OutputManager
   :members:

.. autofunction:: veris.io.write_snapshot

Reading selected fields
-----------------------

``update_state`` inserts fields sampled at a single instant into an initialized
serial State. It preserves the stored halos, the extra cells holding boundary
or neighboring values. Refresh halos before advancing the model again.
Selected records do not reconstruct everything needed for a complete restart.

.. autoclass:: veris.io.Record
   :members:

.. autofunction:: veris.io.read_record

.. autofunction:: veris.io.update_state

Distributed collection
----------------------

The collector removes the halos from each partition, or grid piece, and gathers
the physical fields from its interior cells. All processes must enter collection
in the same order. Only the writing rank, the process assigned to write files,
receives fields for file output.

.. autofunction:: veris.io.distributed.distributed_collector
