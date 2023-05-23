#!/usr/bin/env python

import os
import h5netcdf

import veris
import veris.heat_flux_MITgcm as flux_mitgcm
import veris.heat_flux_CESM as flux_cesm
from veris.area_mass import SeaIceMass, AreaWS
from veris.dynsolver import WindForcingXY, IceVelocities
from veris.dynamics_routines import SeaIceStrength
from veris.ocean_stress import OceanStressUV
from veris.advection import Advection
from veris.clean_up import clean_up_advection, ridging
from veris.growth import Growth
from veris.fill_overlap import fill_overlap, fill_overlap_uv

import veros.tools
from veros import VerosSetup, veros_routine, veros_kernel, KernelOutput, logger
from veros.variables import Variable
from veros.core.operators import numpy as npx, update, at

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets(
    "global_4deg", os.path.join(BASE_PATH, "assets.json")
)


class GlobalFourDegreeSetup(VerosSetup):
    """Global 4 degree model with 15 or 40 vertical levels
    and sea-ice model.

    This setup demonstrates:
     - setting up a realistic model
     - reading input data from external files
     - including Indonesian throughflow
     - implementing surface forcings
     - implementing a sea ice model

    `Adapted from global 4 degree model <https://veros.readthedocs.io/en/latest/reference/setups/4deg.html>`_.

    Reference:
        `Jan Philipp Gärtner. (2023). Sea Ice Modelling in Python. MSc. 55p
        <https://sid.erda.dk/share_redirect/CVvcrowL22/Thesis/Jan_Gaertner_MSc_thesis.pdf>`_.
    """
    __veros_plugins__ = (veris,)

    min_depth = 10.0
    max_depth = 5400.0
    fine_z = False

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = "4deg"

        if self.fine_z:
            settings.nx, settings.ny, settings.nz = 90, 40, 40
        else:
            settings.nx, settings.ny, settings.nz = 90, 40, 15

        settings.dt_mom = 1800.0
        settings.dt_tracer = 86400.0
        settings.runlen = settings.dt_tracer * 360

        settings.x_origin = 4.0
        settings.y_origin = -76.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 1000.0
        settings.iso_dslope = 4.0 / 1000.0
        settings.iso_slopec = 1.0 / 1000.0
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (4 * settings.degtom) ** 3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_implicit_vert_friction = True
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True

        settings.enable_idemix = False
        settings.enable_idemix_hor_diffusion = True
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True

        settings.eq_of_state_type = 5

        # custom variables
        hor_dim = ("xt", "yt")
        forc_dim = hor_dim + ("nmonths",)
        state.dimensions["nmonths"] = 12
        state.var_meta.update(
            sss_clim=Variable("sss_clim", forc_dim, "", "", time_dependent=False),
            sst_clim=Variable("sst_clim", forc_dim, "", "", time_dependent=False),
            qnec=Variable("qnec", forc_dim, "", "", time_dependent=False),
            qnet=Variable("qnet", forc_dim, "", "", time_dependent=False),
            taux=Variable("taux", forc_dim, "", "", time_dependent=False),
            tauy=Variable("tauy", forc_dim, "", "", time_dependent=False),
            #
            qnec_forc=Variable("qnec_forc", hor_dim, "", "", time_dependent=False),
            qnet_forc=Variable("qnet_forc", hor_dim, "", "", time_dependent=False),
            lwnet=Variable("lwnet", hor_dim, "", "", time_dependent=False),
            sen=Variable("sen", hor_dim, "", "", time_dependent=False),
            lat=Variable("lat", hor_dim, "", "", time_dependent=False),
            #
            spres=Variable("spres", forc_dim, "", "", time_dependent=False),
            zbot=Variable("zbot", forc_dim, "", "", time_dependent=False),
            ubot=Variable("ubot", forc_dim, "", "", time_dependent=False),
            vbot=Variable("vbot", forc_dim, "", "", time_dependent=False),
            tcc=Variable("tcc", forc_dim, "", "", time_dependent=False),
            qbot=Variable("qbot", forc_dim, "", "", time_dependent=False),
            rbot=Variable("rbot", forc_dim, "", "", time_dependent=False),
            tbot=Variable("tbot", forc_dim, "", "", time_dependent=False),
            thbot=Variable("thbot", forc_dim, "", "", time_dependent=False),
            swr_net=Variable("swr_net", forc_dim, "", "", time_dependent=False),
            lwr_dw=Variable("lwr_dw", forc_dim, "", "", time_dependent=False),
            u10m=Variable("u10m", forc_dim, "", "", time_dependent=False),
            v10m=Variable("v10m", forc_dim, "", "", time_dependent=False),
            q10m=Variable("q10m", forc_dim, "", "", time_dependent=False),
            t2m=Variable("t2m", forc_dim, "", "", time_dependent=False),
            # veris
            uWind_f=Variable("Zonal wind velocity", forc_dim, "m/s"),
            vWind_f=Variable("Meridional wind velocity", forc_dim, "m/s"),
            SWdown_f=Variable("Downward shortwave radiation", forc_dim, "W/m2"),
            LWdown_f=Variable("Downward longwave radiation", forc_dim, "W/m2"),
            ATemp_f=Variable("Atmospheric temperature", forc_dim, "K"),
            aqh_f=Variable("Atmospheric specific humidity", forc_dim, "g/kg"),
            precip_f=Variable("Precipitation rate", forc_dim, "m/s"),
            snowfall_f=Variable("Snowfall rate", forc_dim, "m/s"),
            evap_f=Variable("Evaporation", forc_dim, "m"),
            surfPress_f=Variable("Surface pressure", forc_dim, "P"),
            #
            qnet_=Variable("", hor_dim, ""),
            forc_salt_surface_ice=Variable("", hor_dim, ""),
            forc_salt_surface_res=Variable("", hor_dim, ""),
            sss=Variable("", hor_dim, ""),
        )

    def _read_forcing(self, var, forcing="forcing", flip_y=False):
        with h5netcdf.File(DATA_FILES[forcing], "r") as infile:
            var_obj = npx.array(infile.variables[var]).T
            if flip_y:
                return npx.flip(var_obj, axis=1)
            else:
                return var_obj

    def _read_slice_interp_upd(
        self,
        veros_var,
        forcing_var,
        veros_grid,
        forcing_grid,
        forcing="era5_ml",
        slice=(slice(None, None, None),
               slice(None, None, None),
               slice(None, None, None)),
        flip_y=True,
    ):
        """Read forcing data, interpolate it to veros grid and update the variable

        Arguments:
            veros_var (:obj:`ndarray`): veros variable to be updated by
                the interpolated forcing variable
            forcing_var (str): the name of forcing variable to be read
            veros_grid (tuple): veros grid
            forcing_grid (tuple): forcing grid
            forcing (str): type of forcing (file) to be read from
            slice (tuple): tuple of indexes to slice the forcing variable
            flip_y (bool): on/off flipping of second axis of the forcing variable

        Returns:
            :obj:`ndarray`
        """

        input_var = self._read_forcing(forcing_var, forcing=forcing, flip_y=flip_y)[
            slice
        ]
        interp_var = veros.tools.interpolate(forcing_grid, input_var, veros_grid)
        output_var = update(veros_var, at[2:-2, 2:-2, :], interp_var)

        return output_var

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        if self.fine_z:
            settings = state.settings
            vs.dzt = veros.tools.get_vinokur_grid_steps(
                settings.nz, self.max_depth, self.min_depth, refine_towards="lower"
            )
        else:
            ddz = npx.array(
                [
                    50.0,
                    70.0,
                    100.0,
                    140.0,
                    190.0,
                    240.0,
                    290.0,
                    340.0,
                    390.0,
                    440.0,
                    490.0,
                    540.0,
                    590.0,
                    640.0,
                    690.0,
                ]
            )
            vs.dzt = ddz[::-1]
        vs.dxt = 4.0 * npx.ones_like(vs.dxt)
        vs.dyt = 4.0 * npx.ones_like(vs.dyt)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t,
            at[...],
            2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi),
        )

    @veros_routine(dist_safe=False, local_variables=["kbot", "xt", "yt", "zt"])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        bathymetry_data = self._read_forcing("bathymetry")
        salt_data = self._read_forcing("salinity")[:, :, ::-1]
        zt_forc = self._read_forcing("zt")[::-1]

        salt_interp = veros.tools.interpolate(
            (vs.xt[2:-2], vs.yt[2:-2], zt_forc),
            salt_data,
            (vs.xt[2:-2], vs.yt[2:-2], vs.zt),
            kind="nearest",
        )

        if self.fine_z:
            salt = salt_interp
        else:
            salt = salt_data

        land_mask = (
            vs.zt[npx.newaxis, npx.newaxis, :] <= bathymetry_data[..., npx.newaxis]
        ) | (salt == 0.0)
        vs.kbot = update(
            vs.kbot, at[2:-2, 2:-2], 1 + npx.sum(land_mask.astype("int"), axis=2)
        )

        # set all-land cells
        all_land_mask = (bathymetry_data == 0) | (vs.kbot[2:-2, 2:-2] == settings.nz)
        vs.kbot = update(
            vs.kbot, at[2:-2, 2:-2], npx.where(all_land_mask, 0, vs.kbot[2:-2, 2:-2])
        )

    @veros_routine(
        dist_safe=False,
        local_variables=[
            "taux",
            "tauy",
            "qnec",
            "qnet",
            "sss_clim",
            "sst_clim",
            "temp",
            "salt",
            "area_t",
            "maskT",
            "forc_iw_bottom",
            "forc_iw_surface",
            "xt",
            "yt",
            "zt",
            "spres",
            "zbot",
            "ubot",
            "vbot",
            "tcc",
            "qbot",
            "rbot",
            "tbot",
            "thbot",
            "swr_net",
            "lwr_dw",
            "u10m",
            "v10m",
            "q10m",
            "t2m",
            # veris forcing
            "uWind_f",
            "vWind_f",
            "SWdown_f",
            "LWdown_f",
            "ATemp_f",
            "aqh_f",
            "precip_f",
            "snowfall_f",
            "evap_f",
            "surfPress_f",
            # veris masks and grid (and veros ones to be copied to veris)
            "maskT",
            "maskU",
            "maskV",
            "coriolis_t",
            "ht",
            "dxt",
            "dyt",
            "dxu",
            "dyu",
            "area_t",
            "area_u",
            "area_v",
            "Fu",
            "Fv",
        ],
    )
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        zt_forc = self._read_forcing("zt")[::-1]

        # initial conditions for T and S
        temp_data = self._read_forcing("temperature")[:, :, ::-1]
        temp_interp = veros.tools.interpolate(
            (vs.xt[2:-2], vs.yt[2:-2], zt_forc),
            temp_data,
            (vs.xt[2:-2], vs.yt[2:-2], vs.zt),
            kind="nearest",
        )
        if self.fine_z:
            temp = temp_interp
        else:
            temp = temp_data
        vs.temp = update(
            vs.temp,
            at[2:-2, 2:-2, :, :2],
            temp[:, :, :, npx.newaxis] * vs.maskT[2:-2, 2:-2, :, npx.newaxis],
        )

        salt_data = self._read_forcing("salinity")[:, :, ::-1]
        salt_interp = veros.tools.interpolate(
            (vs.xt[2:-2], vs.yt[2:-2], zt_forc),
            salt_data,
            (vs.xt[2:-2], vs.yt[2:-2], vs.zt),
            kind="nearest",
        )
        if self.fine_z:
            salt = salt_interp
        else:
            salt = salt_data
        vs.salt = update(
            vs.salt,
            at[2:-2, 2:-2, :, :2],
            salt[..., npx.newaxis] * vs.maskT[2:-2, 2:-2, :, npx.newaxis],
        )

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], self._read_forcing("tau_x"))
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], self._read_forcing("tau_y"))

        #
        xt_forc = self._read_forcing("longitude", forcing="era5_ml")
        yt_forc = self._read_forcing("latitude", forcing="era5_ml")[::-1]

        time_xygrid = (vs.xt[2:-2], vs.yt[2:-2], npx.arange(12))
        forc_time_xygrid = (xt_forc, yt_forc, npx.arange(12))
        time_xyzgrid = (
            vs.xt[2:-2],
            vs.yt[2:-2],
            npx.arange(2),
            npx.arange(12),
        )  # 3rd-dim - 2 vertical model levs
        forc_time_xyzgrid = (
            xt_forc,
            yt_forc,
            npx.arange(2),
            npx.arange(12),
        )  # 3rd-dim - 2 vertical model levs

        hyai = self._read_forcing("hyai", forcing="era5_ml")[-3:]
        hybi = self._read_forcing("hybi", forcing="era5_ml")[-3:]
        hyam = self._read_forcing("hyam", forcing="era5_ml")[-2:]  # L136-L137
        hybm = self._read_forcing("hybm", forcing="era5_ml")[-2:]  # L136-L137

        # -------------------
        lnsp = veros.tools.interpolate(
            forc_time_xygrid,
            self._read_forcing("lnsp", forcing="era5_ml", flip_y=True)[
                ..., 0, :
            ],  # L136
            time_xygrid,
        )
        q = veros.tools.interpolate(
            forc_time_xyzgrid,
            self._read_forcing("q", forcing="era5_ml", flip_y=True)[
                ..., 1:, :
            ],  # L136-L137
            time_xyzgrid,
        )
        t = veros.tools.interpolate(
            forc_time_xyzgrid,
            self._read_forcing("t", forcing="era5_ml", flip_y=True)[
                ..., 1:, :
            ],  # L136-L137
            time_xyzgrid,
        )
        vs.ubot = self._read_slice_interp_upd(
            vs.ubot, "u", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        )  # L136
        vs.vbot = self._read_slice_interp_upd(
            vs.vbot, "v", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        )  # L136
        vs.tcc = self._read_slice_interp_upd(
            vs.tcc, "tcc", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.swr_net = self._read_slice_interp_upd(
            vs.swr_net, "msnswrf", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.lwr_dw = self._read_slice_interp_upd(
            vs.lwr_dw, "msdwlwrf", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.u10m = self._read_slice_interp_upd(
            vs.u10m, "u10", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.v10m = self._read_slice_interp_upd(
            vs.v10m, "v10", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.t2m = self._read_slice_interp_upd(
            vs.t2m, "t2m", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )

        vs.q10m = update(vs.q10m, at[2:-2, 2:-2, :], q[..., -1, :])
        vs.qbot = update(vs.qbot, at[2:-2, 2:-2, :], q[..., 0, :])  # L136  # aqh_f
        vs.tbot = update(vs.tbot, at[2:-2, 2:-2, :], t[..., 0, :])  # L136  # ATemp_f
        vs.spres = update(vs.spres, at[2:-2, 2:-2, :], npx.exp(lnsp))

        for m in range(12):
            ph = flux_cesm.get_press_levs(vs.spres[..., m], hyai, hybi)
            pf = flux_cesm.get_press_levs(vs.spres[..., m], hyam, hybm)
            vs.zbot = update(
                vs.zbot,
                at[2:-2, 2:-2, m],
                flux_cesm.compute_z_level(settings, t[..., m], q[..., m], ph[2:-2, 2:-2]),
            )  # L136

            # air density
            vs.rbot = update(
                vs.rbot,
                at[2:-2, 2:-2, m],
                settings.mwdair / settings.rgas * pf[2:-2, 2:-2, 0] / vs.tbot[2:-2, 2:-2, m],
            )  # L136

            # potential temperature
            vs.thbot = update(
                vs.thbot,
                at[2:-2, 2:-2, m],
                vs.tbot[2:-2, 2:-2, m] * (settings.p0 / pf[2:-2, 2:-2, 0]) ** settings.cappa,
            )  # L136

        # -----------------------------------------------------------------
        # heat flux
        with h5netcdf.File(DATA_FILES["ecmwf"], "r") as ecmwf_data:
            qnec_var = ecmwf_data.variables["Q3"]
            vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], npx.array(qnec_var).T)
            vs.qnec = npx.where(vs.qnec <= -1e10, 0.0, vs.qnec)

        q = self._read_forcing("q_net")
        vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], -q)
        vs.qnet = npx.where(vs.qnet <= -1e10, 0.0, vs.qnet)

        mean_flux = (
            npx.sum(vs.qnet[2:-2, 2:-2, :] * vs.area_t[2:-2, 2:-2, npx.newaxis])
            / 12
            / npx.sum(vs.area_t[2:-2, 2:-2])
        )
        logger.info(
            " removing an annual mean heat flux imbalance of %e W/m^2" % mean_flux
        )
        vs.qnet = (vs.qnet - mean_flux) * vs.maskT[:, :, -1, npx.newaxis]

        # SST and SSS
        vs.sst_clim = update(vs.sst_clim, at[2:-2, 2:-2, :], self._read_forcing("sst"))
        vs.sss_clim = update(vs.sss_clim, at[2:-2, 2:-2, :], self._read_forcing("sss"))

        if settings.enable_idemix:
            vs.forc_iw_bottom = update(
                vs.forc_iw_bottom,
                at[2:-2, 2:-2],
                self._read_forcing("tidal_energy") / settings.rho_0,
            )
            vs.forc_iw_surface = update(
                vs.forc_iw_surface,
                at[2:-2, 2:-2],
                self._read_forcing("wind_energy") / settings.rho_0 * 0.2,
            )

        # ERA5 forcing fields
        vs.uWind_f = self._read_slice_interp_upd(
            vs.uWind_f, "u", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        ) # ubot
        vs.vWind_f = self._read_slice_interp_upd(
            vs.vWind_f, "v", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        ) # vbot
        vs.SWdown_f = self._read_slice_interp_upd(
            vs.SWdown_f, "msdwswrf", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )
        vs.LWdown_f = self._read_slice_interp_upd(
            vs.LWdown_f, "msdwlwrf", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        ) # lwr_dw
        vs.ATemp_f = self._read_slice_interp_upd(
            vs.ATemp_f, "t", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        )
        vs.aqh_f = self._read_slice_interp_upd(
            vs.aqh_f, "q", time_xygrid, forc_time_xygrid, slice=npx.s_[:, :, 1, :]
        )
        vs.precip_f = (
            self._read_slice_interp_upd(
                vs.precip_f, "crr", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
            )
            + self._read_slice_interp_upd(
                vs.precip_f, "lsrr", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
            )
        ) / settings.rhoFresh
        vs.snowfall_f = (
            self._read_slice_interp_upd(
                vs.snowfall_f, "csfr", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
            )
            + self._read_slice_interp_upd(
                vs.snowfall_f,
                "lssfr",
                time_xygrid,
                forc_time_xygrid,
                forcing="era5_sfc",
            )
        ) / settings.rhoFresh
        vs.evap_f = (
            self._read_slice_interp_upd(
                vs.evap_f, "e", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
            )
            / 86400
        )
        vs.surfPress_f = self._read_slice_interp_upd(
            vs.surfPress_f, "sp", time_xygrid, forc_time_xygrid, forcing="era5_sfc"
        )

        vs.Fu = update(
            vs.Fu, at[2:-2, 2:-2], self._read_forcing("fu", forcing="formfact")
        )
        vs.Fv = update(
            vs.Fv, at[2:-2, 2:-2], self._read_forcing("fv", forcing="formfact")
        )

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        state.diagnostics["snapshot"].output_frequency = 86400.0
        state.diagnostics["snapshot"].output_variables += [
            "zbot",
            "spres",
            "ubot",
            "vbot",
            "tcc",
            "qbot",
            "rbot",
            "tbot",
            "thbot",
            "swr_net",
            "lwr_dw",
            "qnet_forc",
            "qnec_forc",
            "lwnet",
            "sen",
            "lat",
            "hIceMean",
            "hSnowMean",
            "Area",
            "uIce",
            "vIce",
            "TSurf",
            "uWind",
            "vWind",
            "uOcean",
            "vOcean",
            "ATemp",
            "LWdown",
            "SWdown",
            "Qnet",
            "qnet_",
            "forc_temp_surface",
            "forc_salt_surface",
            "sss",
            "maskT",
            "theta",
        ]
        state.diagnostics["overturning"].output_frequency = 360 * 86400.0
        state.diagnostics["overturning"].sampling_frequency = settings.dt_tracer
        state.diagnostics["energy"].output_frequency = 360 * 86400.0
        state.diagnostics["energy"].sampling_frequency = 86400
        average_vars = [
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "surface_taux",
            "surface_tauy",
            "psi",
            "kbot",
            "qnet_forc",
            "qnec_forc",
        ]
        state.diagnostics["averages"].output_variables = average_vars
        state.diagnostics["averages"].output_frequency = 30 * 86400.0
        state.diagnostics["averages"].sampling_frequency = 86400

    @veros_routine
    def after_timestep(self, state):
        pass


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    conditional_outputs = {}

    use_cesm_forcing = False
    use_mitgcm_forcing = True

    year_in_seconds = 360 * 86400.0
    (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
        vs.time, year_in_seconds, year_in_seconds / 12.0, 12
    )

    # interpolate the monthly mean data to the value at the current time step
    def current_value(field):
        return f1 * field[:, :, n1] + f2 * field[:, :, n2]

    swr_net = current_value(vs.swr_net)
    lwr_dw = current_value(vs.lwr_dw)

    ocn_mask = vs.maskT[:, :, -1]
    temp = vs.temp[:, :, -1, vs.tau] + 273.12

    if use_cesm_forcing and not use_mitgcm_forcing:
        spres = current_value(vs.spres)
        zbot = current_value(vs.zbot)
        ubot = current_value(vs.ubot)
        vbot = current_value(vs.vbot)
        tcc = current_value(vs.tcc)
        qbot = current_value(vs.qbot)
        rbot = current_value(vs.rbot)
        tbot = current_value(vs.tbot)
        thbot = current_value(vs.thbot)

        # ocean net surface heat flux
        (
            sen,
            lat,
            lwup,
            _,
            taux,
            tauy,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = flux_cesm.flux_atmOcn(
            state,
            ocn_mask,
            rbot,
            zbot,
            ubot,
            vbot,
            qbot,
            tbot,
            thbot,
            vs.u[:, :, -1, vs.tau],
            vs.v[:, :, -1, vs.tau],
            temp,
        )

        # Net LW radiation flux from sea surface
        vs.lwnet = flux_cesm.net_lw_ocn(
            state, ocn_mask, vs.yt[:], qbot, temp, tbot, tcc
        )

        # different/ simpler formula for the surface heat flux
        qir, qh, qe = flux_cesm.flux_atmOcn_simple(
            state,
            ocn_mask,
            spres,
            qbot,
            rbot,
            ubot,
            vbot,
            tbot,
            vs.u[:, :, -1, vs.tau],
            vs.v[:, :, -1, vs.tau],
            temp,
        )

        qnet_simple = swr_net + qir + lwr_dw + qh + qe

        dqir_dt, dqh_dt, dqe_dt = flux_cesm.dqnetdt(
            state,
            ocn_mask,
            spres,
            rbot,
            temp,
            ubot,
            vbot,
            vs.u[:, :, -1, vs.tau],
            vs.v[:, :, -1, vs.tau],
        )

    elif not use_cesm_forcing and use_mitgcm_forcing:
        u10m = current_value(vs.u10m)
        v10m = current_value(vs.v10m)
        q10m = current_value(vs.q10m)
        t2m = current_value(vs.t2m)

        lwup, lat, sen, qnec, taux, tauy, _, _, _ = flux_mitgcm.bulkf_formula_lanl(
            state, u10m, v10m, t2m, q10m, temp, ocn_mask
        )

    else:
        raise ValueError("Either cesm forcing or mitgcm forcing should be active")

    # wind stress
    # if use_cesm_forcing and not use_mitgcm_forcing:
    #    vs.surface_taux = ocn_taux
    #    vs.surface_tauy = ocn_tauy
    # elif not use_cesm_forcing and use_mitgcm_forcing:
    #    vs.surface_taux = taux
    #    vs.surface_tauy = tauy
    # else:
    vs.surface_taux = current_value(vs.taux)
    vs.surface_tauy = current_value(vs.tauy)

    # tke flux
    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (
                    0.5
                    * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])
                    / settings.rho_0
                )
                ** 2
                + (
                    0.5
                    * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])
                    / settings.rho_0
                )
                ** 2
            )
            ** 1.5,
        )

    # heat flux [W/m^2 K kg/J m^3/kg] = [K m/s]
    cp_0 = 3991.86795711963
    sst = current_value(vs.sst_clim)
    swr_net = swr_net * (1.0 - settings.ocean_albedo)

    if use_cesm_forcing and not use_mitgcm_forcing:
        qnec = -(dqir_dt + dqh_dt + dqe_dt)
        qnet = swr_net + vs.lwnet + sen + lat  # mean_flux = (
        #    npx.sum(qnet[2:-2, 2:-2] * vs.area_t[2:-2, 2:-2]) / npx.sum(vs.area_t[2:-2, 2:-2])
        # )
        # qnet = qnet + 50
        conditional_outputs.update(lwnet=vs.lwnet, sen=sen, lat=lat)
    elif not use_cesm_forcing and use_mitgcm_forcing:
        qnec = -qnec
        qnet = swr_net - lwup + lwr_dw + sen + lat
        # qnet = qnet + 50 energy conservation
        conditional_outputs.update(lwnet=lwr_dw - lwup, sen=sen, lat=lat)
    else:
        qnec = current_value(vs.qnec)
        qnet = current_value(vs.qnet)

    forc_temp_surface = (
        (qnet + qnec * (sst - vs.temp[:, :, -1, vs.tau]))
        * ocn_mask
        / cp_0
        / settings.rho_0
    )

    vs.forc_temp_surface = update(
        vs.forc_temp_surface, at[2:-2, 2:-2], forc_temp_surface[2:-2, 2:-2]
    )

    # salinity restoring
    t_rest = 30 * 86400.0
    sss = current_value(vs.sss_clim)
    forc_salt_surface_res = (
        1.0 / t_rest * (sss - vs.salt[:, :, -1, vs.tau]) * ocn_mask * vs.dzt[-1]
    )

    ###############################################
    #################### VERIS ####################
    ###############################################

    # interpolate the forcing data to the current time step
    vs.uWind = current_value(vs.uWind_f)
    vs.vWind = current_value(vs.vWind_f)
    vs.wSpeed = npx.sqrt(vs.uWind**2 + vs.vWind**2)
    vs.SWdown = current_value(vs.SWdown_f)
    vs.LWdown = current_value(vs.LWdown_f)
    vs.ATemp = current_value(vs.ATemp_f)
    vs.aqh = current_value(vs.aqh_f)
    vs.precip = current_value(vs.precip_f)
    vs.snowfall = current_value(vs.snowfall_f)
    vs.surfPress = current_value(vs.surfPress_f)

    # calculate evaporation from latent heat flux
    if use_cesm_forcing or use_mitgcm_forcing:
        vs.evap = lat / (settings.lhEvap * settings.rhoSea)
    else:
        vs.evap = current_value(vs.evap_f)

    # fill overlaps of the forcing fields used in the dynamics routines.
    # the other forcing fields are only used in the thermodynamic routines
    # which do not require an overlap
    vs.uWind, vs.vWind = fill_overlap_uv(state, vs.uWind, vs.vWind)
    vs.surfPress = fill_overlap(state, vs.surfPress)

    # copy ocean surface velocity, temperature and salinity
    vs.uOcean = vs.u[:, :, -1, vs.tau]
    vs.vOcean = vs.v[:, :, -1, vs.tau]
    vs.theta = vs.temp[:, :, -1, vs.tau] + settings.celsius2K
    vs.ocSalt = vs.salt[:, :, -1, vs.tau]

    # copy ocean surface net and shortwave heat flux
    vs.Qnet = -vs.forc_temp_surface * settings.cpWater * settings.rhoSea
    vs.Qsw = -vs.SWdown

    # calculate sea ice mass centered around c-, u-, and v-points
    vs.SeaIceMassC, vs.SeaIceMassU, vs.SeaIceMassV = SeaIceMass(state)

    # calculate sea ice cover fraction centered around u- and v-points
    vs.AreaW, vs.AreaS = AreaWS(state)

    # calculate surface forcing due to wind
    vs.WindForcingX, vs.WindForcingY = WindForcingXY(state)

    # calculate ice strength
    vs.SeaIceStrength = SeaIceStrength(state)

    # calculate ice velocities
    vs.uIce, vs.vIce = IceVelocities(state)

    # calculate stresses on ocean surface
    vs.OceanStressU, vs.OceanStressV = OceanStressUV(state)

    # calculate change in sea ice fields due to advection
    vs.hIceMean, vs.hSnowMean, vs.Area = Advection(state)

    # correct overshoots and other pathological cases after advection
    (
        vs.hIceMean,
        vs.hSnowMean,
        vs.Area,
        vs.TSurf,
        vs.os_hIceMean,
        vs.os_hSnowMean,
    ) = clean_up_advection(state)

    # cut off ice cover fraction at 1 after advection
    vs.Area = ridging(state)

    # calculate thermodynamic ice growth
    (
        vs.hIceMean,
        vs.hSnowMean,
        vs.Area,
        vs.TSurf,
        vs.EmPmR,
        vs.forc_salt_surface_ice,
        vs.Qsw,
        vs.Qnet,
        vs.SeaIceLoad,
        vs.IcePenetSW,
        vs.recip_hIceMean,
    ) = Growth(state)

    # fill overlaps
    vs.hIceMean = fill_overlap(state, vs.hIceMean)
    vs.hSnowMean = fill_overlap(state, vs.hSnowMean)
    vs.Area = fill_overlap(state, vs.Area)
    vs.forc_salt_surface_ice = fill_overlap(state, vs.forc_salt_surface_ice)
    vs.Qnet = fill_overlap(state, vs.Qnet)

    # update the stress on the ocean surface
    vs.surface_taux = vs.surface_taux * (1 - vs.AreaW) + vs.OceanStressU * vs.AreaW
    vs.surface_tauy = vs.surface_tauy * (1 - vs.AreaS) + vs.OceanStressV * vs.AreaS

    # update surface heat and salt flux (Qnet and forc_salt_surface_ice are already area weighted)
    vs.forc_temp_surface = -vs.Qnet / (settings.cpWater * settings.rhoSea)
    vs.forc_salt_surface = vs.forc_salt_surface_ice + forc_salt_surface_res

    # use this if you want to use the salt nudging in regions of open water only
    # vs.forc_salt_surface = vs.forc_salt_surface_ice + forc_salt_surface_res * ( 1 - vs.Area )

    return KernelOutput(
        hIceMean=vs.hIceMean,
        hSnowMean=vs.hSnowMean,
        Area=vs.Area,
        TSurf=vs.TSurf,
        EmPmR=vs.EmPmR,
        Qsw=vs.Qsw,
        Qnet=vs.Qnet,
        SeaIceLoad=vs.SeaIceLoad,
        IcePenetSW=vs.IcePenetSW,
        recip_hIceMean=vs.recip_hIceMean,
        forc_salt_surface_ice=vs.forc_salt_surface_ice,
        forc_salt_surface_res=forc_salt_surface_res,
        uIce=vs.uIce,
        vIce=vs.vIce,
        qnet_=qnet,
        qnet_forc=qnet,
        qnec_forc=qnec,
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
        uWind=vs.uWind,
        vWind=vs.vWind,
        wSpeed=vs.wSpeed,
        SWdown=vs.SWdown,
        LWdown=vs.LWdown,
        ATemp=vs.ATemp,
        aqh=vs.aqh,
        precip=vs.precip,
        snowfall=vs.snowfall,
        evap=vs.evap,
        surfPress=vs.surfPress,
        **conditional_outputs
    )