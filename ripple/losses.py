import torch
from torch import nn

# from torchvision.transforms.v2.functional import elastic
import torchvision
from torchvision.transforms.functional import affine
from .util import tile, pile, cheap_half


# class ShearLoser(nn.Module):
#     def __init__(self, side, filter_count=24):
#         super().__init__()
#         self.side = side
#         self.filter_count = filter_count

#         self.register_buffer(
#             "filters",
#             torch.empty(
#                 (
#                     self.filter_count,
#                     self.side,
#                     self.side // 2 + 1,
#                 )  # , requires_grad=False
#             ),
#         )
#         self.draw_all_filters()

#     def forward(self, y, ŷ):
#         total = 0.0

#         yr = self.rfft_and_roll(y)
#         ŷr = self.rfft_and_roll(ŷ)

#         for f in range(self.filters.shape[0]):
#             y_match = yr * self.filters[f]
#             ŷ_match = ŷr * self.filters[f]
#             diff = (y_match - ŷ_match) ** 2
#             total += torch.mean(diff)

#         return total ** (1 / 2)

#     def rfft_and_roll(self, x):
#         x = torch.fft.rfft2(x).real
#         return torch.roll(x, x.shape[-2] // 2, dims=-2)

#     def unroll_and_unrfft(self, x):
#         x = torch.roll(x, x.shape[-2] // 2, dims=-2)
#         return torch.fft.irfft2(x)

#     def draw_all_filters(self):
#         for n in range(self.filter_count):
#             self.filters[n] = self.draw_rfft_mask(n, self.filter_count, self.side)
#         # return filters

#     def draw_rfft_mask(self, wedge_pair, out_of, edge_length):
#         """
#         If it's western, we draw it as eastern.
#         If it's southern, we draw it as northern.
#         Then we flip it into place.
#         """
#         n = wedge_pair
#         filters = out_of

#         half_plane = torch.zeros(edge_length, edge_length // 2 + 1)

#         n %= filters

#         northern = n < (filters // 2)
#         eastern = (n < (filters // 4)) or (n > (3 * (filters // 4)))

#         nm2 = n % (filters // 2)
#         n_in_octant = nm2 if nm2 < (filters // 4) else (filters // 2) - nm2 - 1

#         wedge = self._draw_wedge(n_in_octant, filters // 4, edge_length)

#         if not eastern:
#             wedge = wedge.rot90(2).transpose(-2, -1)

#         half_plane[: wedge.shape[0], 1:] = wedge
#         if not northern:
#             half_plane = half_plane.flip((0,))

#         return half_plane

#     def _draw_wedge(self, n, wedges_per_octant, edge_length):
#         assert n < wedges_per_octant

#         quadrant = torch.zeros((edge_length // 2, edge_length // 2))
#         xs = torch.linspace(start=0, end=1, steps=edge_length // 2)
#         ys = xs.flip((0,))

#         coords = torch.cartesian_prod(ys, xs).reshape(xs.shape[0], xs.shape[0], 2)
#         angle = torch.atan(coords[..., -2] / coords[..., -1])

#         wedge_outer_edge_length = (edge_length / 2) / wedges_per_octant

#         start_edge_position = wedge_outer_edge_length * n
#         end_edge_position = wedge_outer_edge_length * (n + 1)

#         start_ang = torch.atan(torch.tensor(start_edge_position) / (edge_length / 2))
#         end_ang = torch.atan(torch.tensor(end_edge_position) / (edge_length / 2))

#         quadrant[((angle > start_ang) & (angle < end_ang))] = 1
#         # quadrant[((coords[:, 0] < 1/2) | (coords[:, 1] < 1/2))] = 0
#         quadrant[torch.norm(coords, dim=2, p=torch.inf) < 1 / 2] = 0

#         binomial = torch.tensor([1, 2, 1]) / 4
#         binomial = torch.outer(binomial, binomial.t())
#         binomial = binomial.unsqueeze(0).unsqueeze(0)

#         for _ in range(4):
#             quadrant = quadrant.unsqueeze(0)
#             quadrant = torch.nn.functional.conv2d(quadrant, binomial, padding="same")
#             quadrant = quadrant.squeeze(0)

#         return quadrant


# # def quick_shear(x):
# #     h, w = x.shape[-2], x.shape[-1]
# #     sheared = torch.zeros((*x.shape[:-1], x.shape[-1]*2))

# #     return sheared


# # def shear_equivariance_loss(g, x):
# def rotation_equivariance_loss(gen, x):
#     pan = tile(x[:, :16], 4)
#     mul = x[:, 16:]
#     angle = float(torch.rand(1) * 360 - 180)

#     _, _, normal = gen(x)

#     pan_rotated = affine(
#         pan, angle, [0, 0], 1, 0, torchvision.transforms.InterpolationMode.BILINEAR
#     )
#     mul_rotated = affine(
#         mul, angle, [0, 0], 1, 0, torchvision.transforms.InterpolationMode.BILINEAR
#     )
#     x_rotated = torch.concat([pile(pan_rotated, 4), mul_rotated], dim=1)
#     _, _, rotated = gen(x_rotated)

#     normal_rotated = affine(
#         normal, angle, [0, 0], 1, 0, torchvision.transforms.InterpolationMode.BILINEAR
#     )

#     return torch.mean(torch.abs(normal_rotated - rotated))
