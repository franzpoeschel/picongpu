"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from ...pypicongpu import laser

import typeguard


@typeguard.typechecked
class FromOpenPMDPulseLaser:
    """PICMI object for FromOpenPMDPulseLaser"""

    def __init__(
        self,
        propagation_direction,
        polarization_direction,
        time_offset_si,
        file_path,
        iteration,
        dataset_name,
        datatype,
        polarisationAxisOpenPMD,
        propagationAxisOpenPMD,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: list[list[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        self.propagation_direction = propagation_direction
        self.polarization_direction = polarization_direction
        self.file_path = file_path
        self.iteration = iteration
        self.dataset_name = dataset_name
        self.datatype = datatype
        self.time_offset_si = time_offset_si
        self.polarisationAxisOpenPMD = polarisationAxisOpenPMD
        self.propagationAxisOpenPMD = propagationAxisOpenPMD
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions

    def get_as_pypicongpu(self) -> laser.FromOpenPMDPulseLaser:
        pypicongpu_laser = laser.FromOpenPMDPulseLaser()
        pypicongpu_laser.propagation_direction = self.propagation_direction
        pypicongpu_laser.polarization_direction = self.polarization_direction
        pypicongpu_laser.file_path = self.file_path
        pypicongpu_laser.iteration = self.iteration
        pypicongpu_laser.dataset_name = self.dataset_name
        pypicongpu_laser.datatype = self.datatype
        pypicongpu_laser.time_offset_si = self.time_offset_si
        pypicongpu_laser.polarisationAxisOpenPMD = self.polarisationAxisOpenPMD
        pypicongpu_laser.propagationAxisOpenPMD = self.propagationAxisOpenPMD
        pypicongpu_laser.huygens_surface_positions = self.picongpu_huygens_surface_positions

        return pypicongpu_laser
