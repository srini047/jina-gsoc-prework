## Please Note: This is not official and at the same time there could be a bugs in the code.
## Also, this is intended to be only for the issue: https://github.com/jina-ai/GSoC/issues/21
## Need to work on the AbstractJaxComputationalBackend which will located in `docarray/computation/`
## Sole purpose of this is to get familiarized with the issue and learn the Jax methods and attributes before GSOC coding period begins

import typing
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp

# import numpy as onp
# from docarray.computation import AbstractComputationalBackend
# from docarray.computation import AbstractJaxComputationalBackend (need to create if needed)

# JaxArray => Equivalent to tensorflow tensor ``from docarray.typing import TensorFlowTensor``


def _unsqueeze_if_single_axis(*matrices: jnp.array) -> List[jnp.Array]:
    """
    Unsqueezes arrays(arrays) that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    unsqueezed = []
    for m in matrices:
        if len(m.shape) == 1:
            unsqueezed.append(jax.lax.expand_dims(m, axis=0))
        else:
            unsqueezed.append(m)
    return unsqueezed


def _unsqueeze_if_scalar(j: jnp.array) -> jnp.array:
    """
    Unsqueezes array of a scalar, from shape () to shape (1,).

    :param t: array to unsqueeze.
    :return: unsqueezed jnp.Array
    """
    if len(j.shape) == 0:  # avoid scalar output
        j = jax.lax.expand_dims(j, 0)
    return j


def norm_left(j: jnp.array) -> JaxArray:
    return JaxArray(array=j)


def norm_right(j: jnp.array) -> jnp.array:
    return j.array


class JaxCompBackend():
    """
    Computational backend for Jax.
    """

    _module = jnp
    _cast_output: Callable = norm_left
    _get_array: Callable = norm_right

    @classmethod
    def to_numpy(cls, array: 'JaxArray') -> 'jnp.array':
        return cls._get_array(array).onp.array(array)

    @classmethod
    def none_value(cls) -> typing.Any:
        """Provide a compatible value that represents None in numpy."""
        return jax.nn.initializers.constant(float('nan'))

    @classmethod
    def to_device(cls, array: 'JaxArray', device: str) -> 'JaxArray':
        """Move the array to the specified device."""
        if cls.device(array) == device:
            return array
        else:
            with jax.devices():    # Need to figure out the right alternative
                return cls._cast_output(jnp.identity(cls._get_array(array)))

    @classmethod
    def device(cls, array: 'JaxArray') -> Optional[str]:
        """Return the device on which the array is allocated."""
        return cls._get_array(array).device

    @classmethod
    def detach(cls, array: 'JaxArray') -> 'JaxArray':
        """
        Returns the array detached from its current graph.

        :param array: array to be detached
        :return: a detached array with the same data.
        """
        return cls._cast_output(jax.lax.stop_gradient(cls._get_array(array)))

    @classmethod
    def dtype(cls, array: 'JaxArray') -> jax.dtypes:
        """Get the data type of the array."""
        d_type = cls._get_array(array).dtype
        return d_type.name

    @classmethod
    def minmax_normalize(
        cls,
        array: 'JaxArray',
        t_range: Tuple = (0.0, 1.0),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'JaxArray':
        a, b = t_range

        j = jax.lax.convert_element_type(cls._get_array(array), jnp.float32)
        min_d = x_range[0] if x_range else jnp.min(j, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else jnp.max(j, axis=-1, keepdims=True)

        i = (b - a) * (j - min_d) / (max_d - min_d +
                                     jax.nn.initializers.constant(eps) + a)

        normalized = jnp.clip(i, *((a, b) if a < b else (b, a)))
        # Need to research more about it for its equivalent => ``tensor.tensor.dtype``
        return cls._cast_output(jax.lax.convert_element_type(normalized, jax.array))

    class Retrieval(AbstractComputationalBackend.Retrieval[JaxArray]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'JaxArray',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['JaxArray', 'JaxArray']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: JaxArray of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: the computational device to use.
            :return: Tuple of JaxArray(s) containing the retrieved values, and
                their indices. Both are of shape (n_queries, k)
            """
            comp_be = JaxCompBackend

            if device is not None:
                values = comp_be.to_device(values, device)

            jax_values: jnp.numpy.array = comp_be._get_array(values)
            if len(jax_values.shape) <= 1:
                jax_values = jax.lax.expand_dims(jax_values, axis=0)

            len_jax_values = (
                jax_values.shape[-1] if len(
                    jax_values.shape) > 1 else len(jax_values)
            )

            k = min(k, len_jax_values)

            if not descending:
                jax_values = -jax_values

            result = jax.lax.top_k(input=jax_values, k=k, sorted=True)
            res_values = result.values
            res_indices = result.indices

            if not descending:
                res_values = -result.values

            return comp_be._cast_output(res_values), comp_be._cast_output(res_indices)

        class Metrics(AbstractComputationalBackend.Metrics[JaxArray]):
            """
            Abstract base class for metrics (distances and similarities).
            """

            @staticmethod
            def cosine_sim(
                x_mat: 'JaxArray',
                y_mat: 'JaxArray',
                eps: float = 1e-7,
                device: Optional[str] = None,
            ) -> 'JaxArray':
                """
                Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor  of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
                """
                comp_be = JaxCompBackend
                x_mat_jp: jnp.array = comp_be._get_array(x_mat)
                y_mat_jp: jnp.array = comp_be._get_array(y_mat)

                with jax.devices():
                    x_mat_jp = jnp.identity(x_mat_jp)
                    y_mat_jp = jnp.identity(y_mat_jp)

                x_mat_jp, y_mat_jp = _unsqueeze_if_single_axis(
                    x_mat_jp, y_mat_jp)

                a_n = jax.nn.standardize(x_mat_jp, axis=1)[1]
                b_n = jax.nn.standardize(y_mat_jp, axis=1)[1]
                # Need to find a way for Max as there is not attribute like in tensorflow
                a_norm = x_mat_jp / \
                    jnp.clip(
                        a_n, a_min=eps, a_max=jnp.finfo(jnp.float32).max)
                
                b_norm = y_mat_jp / \
                    jnp.clip(
                        b_n, b_min=eps, a_max=jnp.finxfo(jnp.float32).max)

                sims = jax.lax.squeeze(jnp.matmul(a_norm, jax.lax.transpose(b_norm)))    # Check which is better alternatives available
                sims = _unsqueeze_if_scalar(sims)

                return comp_be._cast_output(sims)
            
            @staticmethod
            def euclidean_distance(
                x_mat: 'JaxArray',
                y_mat: 'JaxArray',
                device: Optional[str] = None
            ) -> 'JaxArray':
                """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: array of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
                """

                comp_be = JaxCompBackend
                x_max_jp: jnp.array = comp_be._get_array(x_mat)
                y_max_jp: jnp.array = comp_be._get_array(y_mat)

                with jax.devices():
                    x_mat_jp = jnp.identity(x_mat_jp)
                    y_mat_jp = jnp.identity(y_mat_jp)

                x_mat_jp, y_mat_jp = _unsqueeze_if_single_axis(x_mat_jp, y_mat_jp)

                dists = jax.lax.squeeze(jnp.linalg.norm(jnp.subtract(x_mat_jp, y_mat_jp), axis=-1))
                dists - _unsqueeze_if_scalar(dists)

                return comp_be._cast_output(dists)

            @staticmethod
            def sqeuclidean_dist(
                x_mat: 'JaxArray',
                y_mat: 'JaxArray',
                device: Optional[str] = None
            ) -> 'JaxArray':
                """
                Pairwise Squared Euclidian distances between all vectors
                in x_mat and y_mat.

            :param x_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: array of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Array of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
                """

                dists = JaxCompBackend.Metrics.euclidean_dist(x_mat, y_mat)
                squared: jnp.array = jax.lax.square(
                    JaxCompBackend._get_array(dists)
                )

                return JaxCompBackend._cast_output(squared)

