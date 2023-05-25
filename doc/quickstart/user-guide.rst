Basic usage
===========

Assuming that you have already `installed Veros <https://veros.readthedocs.io/en/latest/introduction/get-started.html>`_, the following command is required to install the sea ice plugin Veris:

::

   $ pip install veris

To get started with a new setup, you can use :obj:`seaice_global_4deg` as a template:

::

   $ veros copy-setup seaice_global_4deg


To enable Veris on a completely new setup, you will have to register it as a Veris plugin.
Add the following to your setup definition:

::

   import veris

   class MyVerosSetup(VerosSetup):
       __veros_plugins__ = (veris,)

This registers the plugin for use with Veros.
Then, you can use :doc:`the Veris settings </reference/settings>` to configure Veris.

.. seealso::

   All new :doc:`settings </reference/settings>` and :doc:`variables </reference/variables>` defined by Veris in their respective sections.
