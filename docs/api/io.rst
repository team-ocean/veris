Input and output
================

The public classes and record functions below are imported from ``veris.io``.
They run on the host, outside differentiated integration. See
:doc:`/reference/io` for complete examples, calendar windows, sampling rules,
registry defaults and selected-field input limitations.

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

Use OutputManager as a context manager to close files and finalize complete
averaging windows. Snapshot writing is independent of its stream schedules.
Existing output paths are rejected.

.. autoclass:: veris.io.OutputManager
   :members:

.. autofunction:: veris.io.write_snapshot

Reading selected fields
-----------------------

``update_state`` inserts instantaneous fields into an initialized serial
State, preserving the stored halos. Refresh halos before further integration.
Selected records do not reconstruct a complete restart.

.. autoclass:: veris.io.Record
   :members:

.. autofunction:: veris.io.read_record

.. autofunction:: veris.io.update_state

Distributed collection
----------------------

The collector removes every partition's halos and gathers physical fields.
All processes enter collection in the same order; only the writing rank
receives fields for file output.

.. autofunction:: veris.io.distributed.distributed_collector
