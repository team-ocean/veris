"""The mean reducer retains spatial buffers, independent of trajectory length."""

from datetime import timedelta
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from veris.io.configuration import Stream


def test_reduction_scan_has_only_carry_and_constant_storage() -> None:
    """Inspect real JAX scan outputs, then check its arithmetic sample sum."""
    from veris.io.scan import _initial_carry, _reduce_chunk

    streams = (
        Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=1000)),
        Stream("instant", ("hIceMean",), timedelta(seconds=1)),
    )
    state = {"Area": jnp.zeros((6, 7)), "hIceMean": jnp.ones((6, 7))}

    def select(value: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return value

    with jax.enable_x64():
        carry = _initial_carry(state, streams, select)
        assert tuple(set(fields) for fields in carry[1]) == ({"Area"}, set())
        reduce = partial(
            _reduce_chunk,
            advance=lambda value: {name: array + 1 for name, array in value.items()},
            select=select,
            streams=streams,
            intervals=(1, 1),
            sample_initial=True,
            checkpoint=False,
            physics_x64=True,
        )
        shapes = []
        for steps in (10, 1000):
            result = jax.eval_shape(partial(reduce, steps=steps), carry)
            shapes.append([leaf.shape for leaf in jax.tree.leaves(result)])
            graph = jax.make_jaxpr(partial(reduce, steps=steps))(carry)
            scan = next(eq for eq in graph.jaxpr.eqns if eq.primitive.name == "scan")
            if "ft_out" in scan.params:
                output_carry, history = scan.params["ft_out"].unpack()
                constants, input_carry, samples = scan.params["ft_in"].unpack()
                assert len(history) == len(samples) == 0
                assert len(scan.outvars) == len(output_carry)
                assert len(scan.invars) == len(constants) + len(input_carry)
            else:
                assert len(scan.outvars) == scan.params["num_carry"]
                assert (
                    len(scan.invars)
                    == scan.params["num_carry"] + scan.params["num_consts"]
                )
        assert shapes[0] == shapes[1]
        final, sums, counts, iteration = jax.jit(partial(reduce, steps=10))(carry)
        np.testing.assert_array_equal(final["Area"], 10)
        np.testing.assert_array_equal(sums[0]["Area"], 45)
        assert int(counts[0]) == 10 and int(iteration) == 10
