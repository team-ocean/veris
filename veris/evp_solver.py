"""Elastic-viscous-plastic subcycling of ice velocity and stress tensors."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from veris._solver_types import EVPCarry, EVPSettings, EVPState
from veris._typing import jit
from veris.averaging import c_point_to_z_point
from veris.dynamics_routines import (
    basal_drag_coeffs,
    ocean_drag_coeffs,
    side_drag,
    strainrates,
    stressdiv,
    viscosities,
)
from veris.fill_overlap import fill_overlap_uv
from veris.global_sum import global_sum

printEvpResidual = False
plotEvpResidual = False


@partial(jit, static_argnames=["sett", "axis_names"])
def evp_solver(
    vs: EVPState, sett: EVPSettings, *, axis_names: tuple[str, ...] = ()
) -> tuple[Array, Array, Array, Array, Array]:
    """solve the momentum equation and calculate u^n, sigma^n from u^(n-1), sigma^(n-1)
    using subcycling iterations of evp_solver_body.

    Inside shard_map, pass its mesh axis names to combine residual diagnostics
    over all devices/processes. The default reports the serial interior norm.
    """

    def evp_solver_body(iEVP: int | Array, arg_body: EVPCarry) -> EVPCarry:
        """loop body of the elastic-viscous-plastic solver
        the components of the strain rate tensor and stress tensor are calculated
        following Hibler (1979). the time stepping of the stress and velocity
        equations is done following Kimmritz (2016)
        """

        (
            vs,
            uIce,
            vIce,
            uIceNm1,
            vIceNm1,
            sigma1,
            sigma2,
            sigma12,
            denom1,
            denom2,
            EVPcFac,
            evpAlphaC,
            evpAlphaZ,
            evpBetaU,
            evpBetaV,
            resSig,
            resU,
        ) = arg_body

        if sett.computeEvpResidual:
            # save previous (p-1) iteration for residual computation
            sig11Pm1 = 0.5 * (sigma1 + sigma2)
            sig22Pm1 = 0.5 * (sigma1 - sigma2)
            sig12Pm1 = sigma12
            uIcePm1 = uIce
            vIcePm1 = vIce

        e11, e22, e12 = strainrates(vs, sett, uIce, vIce)
        zeta, _eta, press = viscosities(vs, sett, e11, e22, e12)
        # sig11, sig22, sig12 = stress(vs, sett, e11, e22, e12, zeta, eta, press)

        # calculate adaptive relaxation parameters
        if sett.useAdaptiveEVP:
            evpAlphaC = (
                jnp.sqrt(
                    zeta * EVPcFac / jnp.maximum(vs.SeaIceMassC, 1e-4) * vs.recip_rA
                )
                * vs.iceMask
            )

            evpAlphaC = jnp.maximum(evpAlphaC, sett.aEVPalphaMin)
            denom1 = 1.0 / evpAlphaC
            denom2 = denom1

        # copied from the MITgcm
        evpRevFac = 1
        recip_evpRevFac = 0.25

        # define principle strain rate components
        ep = e11 + e22
        em = e11 - e22

        # used to calculate the components of the stress tensor
        divergence = 2 * zeta * ep - press
        tension = 2 * zeta * em
        shear = 2 * c_point_to_z_point(vs, sett, zeta) * e12

        # step principal stress components
        sigma1 = (sigma1 * (evpAlphaC - evpRevFac) + divergence) * denom1 * vs.iceMask
        sigma2 = (
            (sigma2 * (evpAlphaC - evpRevFac) + tension * recip_evpRevFac)
            * denom2
            * vs.iceMask
        )

        # recover components of the stress tensor
        sig11 = 0.5 * (sigma1 + sigma2)
        sig22 = 0.5 * (sigma1 - sigma2)

        # calculate adaptive relaxation parameter on z-points
        if sett.useAdaptiveEVP:
            evpAlphaZ = 0.5 * (evpAlphaC + jnp.roll(evpAlphaC, 1, 1))
            evpAlphaZ = 0.5 * (evpAlphaZ + jnp.roll(evpAlphaZ, 1, 0))
            denom2 = 1.0 / evpAlphaZ

        # step sigma12
        sigma12 = (sigma12 * (evpAlphaZ - evpRevFac) + shear * recip_evpRevFac) * denom2

        # calculate divergence of stress tensor
        stressDivX, stressDivY = stressdiv(vs, sett, sig11, sig22, sigma12)

        # calculate drag coefficients
        cDrag = ocean_drag_coeffs(vs, sett, uIce, vIce)
        cBotC = basal_drag_coeffs(vs, sett, uIce, vIce)

        # Materialize shared stencil inputs on CUDA: this improves the measured
        # 256x256 P100 case, at a modest cost for small grids. Native CPU coupled
        # benchmarks did not support the same barrier, so keep CPU fusion intact.
        # JAX selects by the actual compilation target, not the host default;
        # this identity changes neither equations nor AD. See benchmarks/.
        cDrag, cBotC, stressDivX, stressDivY = jax.lax.platform_dependent(
            (cDrag, cBotC, stressDivX, stressDivY),
            cuda=jax.lax.optimization_barrier,
            default=lambda values: values,
        )

        # over open ocean..., see comments in MITgcm: pkg/seaice/seaice_evp.F
        locMaskU = vs.SeaIceMassU
        locMaskV = vs.SeaIceMassV
        locMaskU = jnp.where(locMaskU != 0, 1, locMaskU)
        locMaskV = jnp.where(locMaskV != 0, 1, locMaskV)

        # calculate total ocean and wind forcing
        ForcingX = (
            vs.WindForcingX
            + (
                0.5 * (cDrag + jnp.roll(cDrag, 1, 0)) * sett.cosWat * vs.uOcean
                - jnp.sign(vs.fCori)
                * sett.sinWat
                * 0.5
                * (
                    cDrag * 0.5 * (vs.uOcean - uIce + jnp.roll(vs.uOcean - uIce, -1, 1))
                    + jnp.roll(cDrag, 1, 0)
                    * 0.5
                    * (
                        jnp.roll(vs.uOcean - uIce, 1, 0)
                        + jnp.roll(jnp.roll(vs.uOcean - uIce, 1, 0), -1, 1)
                    )
                )
                * locMaskU
            )
            * vs.AreaW
        )

        ForcingY = (
            vs.WindForcingY
            + (
                0.5 * (cDrag + jnp.roll(cDrag, 1, 1)) * sett.cosWat * vs.vOcean
                + jnp.sign(vs.fCori)
                * sett.sinWat
                * 0.5
                * (
                    cDrag * 0.5 * (vs.vOcean - vIce + jnp.roll(vs.vOcean - vIce, -1, 0))
                    + jnp.roll(cDrag, 1, 1)
                    * 0.5
                    * (
                        jnp.roll(vs.vOcean - vIce, 1, 1)
                        + jnp.roll(jnp.roll(vs.vOcean - vIce, 1, 1), -1, 0)
                    )
                )
                * locMaskV
            )
            * vs.AreaS
        )

        # add coriolis term
        fvAtC = vs.SeaIceMassC * vs.fCori * 0.5 * (vIce + jnp.roll(vIce, -1, 1))
        fuAtC = vs.SeaIceMassC * vs.fCori * 0.5 * (uIce + jnp.roll(uIce, -1, 0))
        ForcingX = ForcingX + 0.5 * (fvAtC + jnp.roll(fvAtC, 1, 0))
        ForcingY = ForcingY - 0.5 * (fuAtC + jnp.roll(fuAtC, 1, 1))

        # interpolate relaxation parameters to velocity points
        if sett.useAdaptiveEVP:
            evpBetaU = 0.5 * (evpAlphaC + jnp.roll(evpAlphaC, 1, 0))
            evpBetaV = 0.5 * (evpAlphaC + jnp.roll(evpAlphaC, 1, 1))

        betaFacU = evpBetaU * sett.recip_deltatDyn
        betaFacV = evpBetaV * sett.recip_deltatDyn
        betaFacP1U = betaFacU + sett.recip_deltatDyn
        betaFacP1V = betaFacV + sett.recip_deltatDyn

        denomU = vs.SeaIceMassU * betaFacP1U + vs.AreaW * (
            0.5 * (cDrag + jnp.roll(cDrag, 1, 0)) * sett.cosWat
            + 0.5 * (cBotC + jnp.roll(cBotC, 1, 0))
        )
        denomV = vs.SeaIceMassV * betaFacP1V + vs.AreaS * (
            0.5 * (cDrag + jnp.roll(cDrag, 1, 1)) * sett.cosWat
            + 0.5 * (cBotC + jnp.roll(cBotC, 1, 1))
        )

        denomU = jnp.where(denomU == 0, 1, denomU)
        denomV = jnp.where(denomV == 0, 1, denomV)

        # add lateral drag
        if not sett.noSlip:
            SideDragU, SideDragV = side_drag(vs, sett, uIce, vIce)

            # the side drag coefficients are not multiplied by the area because they are calculated from
            # SeaIceMass which is calculated from hIceMean which includes the area
            denomU = denomU + SideDragU
            denomV = denomV + SideDragV

        uIce = (
            vs.iceMaskU
            * (
                betaFacU * vs.SeaIceMassU * uIce
                + vs.SeaIceMassU * sett.recip_deltatDyn * uIceNm1
                + ForcingX
                + stressDivX
            )
            / denomU
        )
        vIce = (
            vs.iceMaskV
            * (
                betaFacV * vs.SeaIceMassV * vIce
                + vs.SeaIceMassV * sett.recip_deltatDyn * vIceNm1
                + ForcingY
                + stressDivY
            )
            / denomV
        )

        # fill overlaps
        uIce, vIce = fill_overlap_uv(uIce, vIce)

        # residual computation
        if sett.computeEvpResidual:
            sig11Pm1 = (sig11 - sig11Pm1) * evpAlphaC * vs.iceMask
            sig22Pm1 = (sig22 - sig22Pm1) * evpAlphaC * vs.iceMask
            sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ  # * maskZ

            uIcePm1 = vs.iceMaskU * (uIce - uIcePm1) * evpBetaU
            vIcePm1 = vs.iceMaskV * (vIce - vIcePm1) * evpBetaV

            # if not explicitDrag:
            #     ForcingX = ForcingX - uIce * dragU
            #     ForcingY = ForcingY - vIce * dragV

            # uIcePm1 = ( SeaIceMassU * (uIce - uIceNm1)*recip_deltatDyn
            #             - (ForcingX + stressDivX)
            #            ) * iceMaskU
            # vIcePm1 = ( SeaIceMassV * (vIce - vIceNm1)*recip_deltatDyn
            #             - (ForcingY + stressDivY)
            #            ) * iceMaskV

            # Halo exchange uses two cells on each side. Residual norms count
            # each interior cell once, excluding periodic halo duplicates.
            stress_norm = (sig11Pm1**2 + sig22Pm1**2 + sig12Pm1**2)[2:-2, 2:-2].sum()
            velocity_norm = (uIcePm1**2 + vIcePm1**2)[2:-2, 2:-2].sum()
            resSig = resSig.at[iEVP].set(global_sum(stress_norm, axis_names))
            resU = resU.at[iEVP].set(global_sum(velocity_norm, axis_names))

            if printEvpResidual:
                jax.debug.print(
                    "evp resU, resSigma: {i} {u:.6e} {s:.6e}",
                    i=iEVP,
                    u=resU[iEVP],
                    s=resSig[iEVP],
                )

        return (
            vs,
            uIce,
            vIce,
            uIceNm1,
            vIceNm1,
            sigma1,
            sigma2,
            sigma12,
            denom1,
            denom2,
            EVPcFac,
            evpAlphaC,
            evpAlphaZ,
            evpBetaU,
            evpBetaV,
            resSig,
            resU,
        )

    # calculate parameter used for adaptive relaxation parameters
    if sett.useAdaptiveEVP:
        aEVPcStar = 4
        EVPcFac = sett.deltatDyn * aEVPcStar * (jnp.pi * sett.aEvpCoeff) ** 2
    else:
        EVPcFac = 0

    denom1 = jnp.full(vs.iceMask.shape, 1 / sett.evpAlpha)
    denom2 = denom1

    # copy previous time step (n-1) of ice velocities and stress tensor
    uIceNm1 = vs.uIce
    vIceNm1 = vs.vIce
    uIce = vs.uIce
    vIce = vs.vIce
    sigma1 = vs.sigma1
    sigma2 = vs.sigma2
    sigma12 = vs.sigma12

    # initialize adaptive EVP specific fields
    evpAlphaC = jnp.full(vs.iceMask.shape, sett.evpAlpha)
    evpAlphaZ = jnp.full(vs.iceMask.shape, sett.evpAlpha)
    evpBetaU = jnp.full(vs.iceMask.shape, sett.evpBeta)
    evpBetaV = jnp.full(vs.iceMask.shape, sett.evpBeta)

    resSig = jnp.zeros(sett.nEVPsteps)
    resU = jnp.zeros(sett.nEVPsteps)

    # set argument for the loop (the for_loop of jax can only take one argument)
    arg_body: EVPCarry = (
        vs,
        uIce,
        vIce,
        uIceNm1,
        vIceNm1,
        sigma1,
        sigma2,
        sigma12,
        denom1,
        denom2,
        EVPcFac,
        evpAlphaC,
        evpAlphaZ,
        evpBetaU,
        evpBetaV,
        resSig,
        resU,
    )

    # calculate u^n, sigma^n and residuals
    arg_body = jax.lax.fori_loop(0, sett.nEVPsteps, evp_solver_body, arg_body)

    # return uIce, vIce, sigma1, sigma2, sigma12
    return arg_body[1], arg_body[2], arg_body[5], arg_body[6], arg_body[7]
