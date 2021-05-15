#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
Adapted from: https://github.com/tensorflow/addons/blob/v0.9.1/tensorflow_addons/image/dense_image_warp.py#L189-L246
Modified original 2D image warp for 3D images
"""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


def _interpolate_trilinear(grid,
                          query_points,
                          name='interpolate_trilinear',
                          indexing='ijk'):
  """Similar to Matlab's interp2 function.

  Finds values for query points on a grid using trilinear interpolation.

  Args:
    grid: a 5-D float `Tensor` of shape `[batch, depth, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 3]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ijk),
      or Cartesian coordinates (xyz).

  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`

  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """
  if indexing != 'ijk' and indexing != 'xyz':
    raise ValueError('Indexing mode must be \'ijk\' or \'xyz\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 5:
      msg = 'Grid must be 5 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))
    
    ## depth = z axis, height = y axis, width = x axis
    batch_size, depth, height, width, channels = (array_ops.shape(grid)[0],
                                           array_ops.shape(grid)[1],
                                           array_ops.shape(grid)[2],
                                           array_ops.shape(grid)[3],
                                           array_ops.shape(grid)[4])
    shape = [batch_size, depth, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.'),
        check_ops.assert_equal(
            array_ops.shape(query_points)[2],
            3,
            message='Query points must be size 3 in dim 2.') #Dim of flow field = 3
    ]):
      num_queries = array_ops.shape(query_points)[1] #number of voxels 

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            depth, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            height, 2, message='Grid width must be at least 2.'), 
        check_ops.assert_greater_equal(
            width, 2, message='Grid depth must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      ## diplacement fields output by network are, Channel 0:1:2 = uz: uy: ux
      index_order = [0, 1, 2] if indexing == 'ijk' else [2, 1, 0] 
      unstacked_query_points = array_ops.unstack(query_points, axis=-1)

    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim] 

        size_in_indexing_dimension = shape[dim + 1] # depth, height, width

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = constant_op.constant(0.0, dtype=query_type)
        # above function is ensuring that index dont go out of bounds
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor, dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        # ensuring that alpha has value bw 0 and 1.
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)

    with ops.control_dependencies([
        check_ops.assert_less_equal(
            math_ops.cast(batch_size * depth * height * width , dtype=dtypes.float32),
            np.iinfo(np.int32).max / 8,
            message="""The image size or batch size is sufficiently large
                       that the linearized addresses used by array_ops.gather
                       may exceed the int32 limit.""")
    ]):
      flattened_grid = array_ops.reshape(
          grid, [batch_size * depth * height * width , channels])
      #Creates a sequence of batch offets
      batch_offsets = array_ops.reshape(
          math_ops.range(batch_size) * depth * height * width , [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(z_coords, y_coords, x_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + z_coords * height * width + y_coords * width + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 8 corners around each query point
    back_top_left = gather(floors[0], floors[1], floors[2], 'top_left_up')
    back_top_right = gather(floors[0], floors[1], ceils[2], 'top_left_down')
    back_bottom_left = gather(floors[0], ceils[1], floors[2], 'top_right_up')
    back_bottom_right = gather(floors[0], ceils[1], ceils[2], 'top_right_down')
    front_top_left = gather(ceils[0], floors[1], floors[2], 'bottom_left_up')
    front_top_right = gather(ceils[0], floors[1], ceils[2], 'bottom_left_down')
    front_bottom_left = gather(ceils[0], ceils[1], floors[2], 'bottom_right_up')
    front_bottom_right = gather(ceils[0], ceils[1], ceils[2], 'bottom_right_down')

    # now, do the actual interpolation
    with ops.name_scope('interpolate'):
      inter_back_top = alphas[2] * (back_top_right - back_top_left ) + back_top_left
      inter_back_bottom =alphas[2] * (back_bottom_right - back_bottom_left ) + back_bottom_left
      inter_front_top = alphas[2] * (front_top_right - front_top_left ) + front_top_left
      inter_front_bottom =alphas[2] * (front_bottom_right - front_bottom_left ) + front_bottom_left      
      inter_back = alphas[1] * (inter_back_bottom - inter_back_top) + inter_back_top
      inter_front = alphas[1] * (inter_front_bottom - inter_front_top) + inter_front_top
      inter = alphas[0] * (inter_front - inter_back) + inter_back

    return inter


def dense_image_warp(image, flow, name='dense_image_warp'):
  """Image warping using per-pixel flow vectors.

  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, k, j, i, c] is
  images[b, k-  flow[b, k, j, i, 2] ,  j - flow[b, k, j, i, 0], i - flow[b, k, j, i, 1], c].

  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by trilinear
  interpolation of the 6 nearest pixels around
  (b, k-  flow[b, k, j, i, 2] ,  j - flow[b, k, j, i, 0], i - flow[b, k, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.


  Args:
    image: 5-D float `Tensor` with shape `[batch, depth, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, depth, height, width, 3]`.
    name: A name for the operation (optional).

    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.

  Returns:
    A 5-D float `Tensor` with shape`[batch, depth, height, width, channels]`
      and same type as input image.

  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  """
  with ops.name_scope(name):
    batch_size, depth, height, width, channels = (array_ops.shape(image)[0],
                                           array_ops.shape(image)[1],
                                           array_ops.shape(image)[2],
                                           array_ops.shape(image)[3],
                                           array_ops.shape(image)[4])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_z, grid_y, grid_x = array_ops.meshgrid(
        math_ops.range(depth), math_ops.range(height), math_ops.range(width), indexing='ij')
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_z, grid_y, grid_x], axis=-1), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
#    query_points_on_grid = batched_grid - flow
    query_points_on_grid = batched_grid + flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, depth*height*width , 3])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_trilinear(image, query_points_flattened)
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, depth, height, width, channels])
    return interpolated
