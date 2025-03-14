"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from ..rendering import RenderedObject


import typeguard


@typeguard.typechecked
class Plugin(RenderedObject):
    """general interface for all plugins"""

    def __init__(self):
        raise NotImplementedError("abstract base class only")

    def get_generic_plugin_rendering_context(self) -> dict:
        """
        retrieve a context valid for "any plugin"

        Problem: Every plugin has its respective schema, and it is difficult
        in JSON (particularly in a mustache-compatible way) to get the type
        of the schema.

        Solution: The normal rendering of plugins get_rendering_context()
        provides **only their parameters**, i.e. there is **no meta
        information** on types etc.

        If a generic plugin is requested one can use the schema for
        "Plugin" (this class), for which this method returns the
        correct content, which includes metainformation and the data on the
        schema itself.

        E.g.:

        .. code::

            {
                "type": {
                    "phasespace": true,
                    "auto": false,
                    ...
                },
                "data": DATA
            }

        where DATA is the serialization as returned by get_rendering_context().

        There are *two* context serialization methods for plugins:

        - get_rendering_context()

            - provided by RenderedObject parent class, serialization ("context
              building") performed by _get_serialized()
            - _get_serialized() implemented in *every plugin*
            - checks against schema of respective plugin
            - returned context is a representation of *exactly this plugin*
            - (left empty == not implemented in parent Plugin)

        - get_generic_plugin_rendering_context()

            - implemented in parent class Plugin
            - returned representation is generic for *any plugin*
              (i.e. contains meta information which type is actually used)
            - passes information from get_rendering_context() through
            - returned representation is designed for easy use with templating
              engine mustache
        """
        # import here to avoid circular inclusion
        from .auto import Auto
        from .phase_space import PhaseSpace
        from .energy_histogram import EnergyHistogram

        template_name_by_type = {Auto: "auto", PhaseSpace: "phasespace", EnergyHistogram: "energyhistogram"}
        if self.__class__ not in template_name_by_type:
            raise RuntimeError("unkown type: {}".format(self.__class__))

        serialized_data = self.get_rendering_context()

        # create dict with all types set to false, except for the current one
        typeID = dict(map(lambda type_name: (type_name, False), template_name_by_type.values()))
        self_class_template_name = template_name_by_type[self.__class__]
        typeID[self_class_template_name] = True

        # final context to be returned: data + type info
        returned_context = {"typeID": typeID, "data": serialized_data}

        # make sure it passes schema checks
        RenderedObject.check_context_for_type(Plugin, returned_context)
        return returned_context
