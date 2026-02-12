import torch
import numpy as np

class SurfaceShape:
    def __init__(self, device='cpu'):
        self.device = device

    def sample_surface(self, num_points, num_envs, params):
        """
        Sample points from the surface of the shape.
        Args:
            num_points (int): Number of points to sample per environment.
            num_envs (int): Number of environments.
            params (Tensor): Shape parameters [num_envs, num_params].
        Returns:
            points (Tensor): [num_envs, num_points, 3]
            normals (Tensor): [num_envs, num_points, 3]
        """
        raise NotImplementedError

class Sphere(SurfaceShape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [radius]
        radius = params[:, 0].view(-1, 1, 1)
        
        # Sample unit vectors (normal)
        normal = torch.randn((num_envs, num_points, 3), device=self.device)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        
        # Points are on the surface
        points = radius * normal
        
        return points, normal

class Cuboid(SurfaceShape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [size_x, size_y, size_z]
        sx = params[:, 0]
        sy = params[:, 1]
        sz = params[:, 2]
        
        # Areas of faces
        area_x = sy * sz
        area_y = sx * sz
        area_z = sx * sy
        
        total_area = 2 * (area_x + area_y + area_z)
        
        # Probabilities for each pair of faces
        p_x = (2 * area_x) / total_area
        p_y = (2 * area_y) / total_area
        
        # Expand for broadcasting
        sx = sx.view(-1, 1, 1)
        sy = sy.view(-1, 1, 1)
        sz = sz.view(-1, 1, 1)
        p_x = p_x.view(-1, 1)
        p_y = p_y.view(-1, 1)
        
        # Random choice of face axis: 0=X, 1=Y, 2=Z
        rand_face = torch.rand((num_envs, num_points), device=self.device)
        
        # Masks
        mask_x = rand_face < p_x
        mask_y = (rand_face >= p_x) & (rand_face < (p_x + p_y))
        mask_z = rand_face >= (p_x + p_y)
        
        # Random sign +/- 1
        sign = torch.sign(torch.rand((num_envs, num_points), device=self.device) - 0.5)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign) # handle 0 case
        
        # Initialize points and normals
        points = torch.zeros((num_envs, num_points, 3), device=self.device)
        normals = torch.zeros((num_envs, num_points, 3), device=self.device)
        
        # X-faces
        if mask_x.any():
            k = mask_x.sum()
            vals = torch.zeros((k, 3), device=self.device)
            vals[:, 0] = sign[mask_x] * sx.expand(-1, num_points, -1)[mask_x].squeeze(-1) / 2
            vals[:, 1] = (torch.rand(k, device=self.device) - 0.5) * sy.expand(-1, num_points, -1)[mask_x].squeeze(-1)
            vals[:, 2] = (torch.rand(k, device=self.device) - 0.5) * sz.expand(-1, num_points, -1)[mask_x].squeeze(-1)
            points[mask_x] = vals
            
            norms = torch.zeros((k, 3), device=self.device)
            norms[:, 0] = sign[mask_x]
            normals[mask_x] = norms
        
        # Y-faces
        if mask_y.any():
            k = mask_y.sum()
            vals = torch.zeros((k, 3), device=self.device)
            vals[:, 0] = (torch.rand(k, device=self.device) - 0.5) * sx.expand(-1, num_points, -1)[mask_y].squeeze(-1)
            vals[:, 1] = sign[mask_y] * sy.expand(-1, num_points, -1)[mask_y].squeeze(-1) / 2
            vals[:, 2] = (torch.rand(k, device=self.device) - 0.5) * sz.expand(-1, num_points, -1)[mask_y].squeeze(-1)
            points[mask_y] = vals
            
            norms = torch.zeros((k, 3), device=self.device)
            norms[:, 1] = sign[mask_y]
            normals[mask_y] = norms
            
        # Z-faces
        if mask_z.any():
            k = mask_z.sum()
            vals = torch.zeros((k, 3), device=self.device)
            vals[:, 0] = (torch.rand(k, device=self.device) - 0.5) * sx.expand(-1, num_points, -1)[mask_z].squeeze(-1)
            vals[:, 1] = (torch.rand(k, device=self.device) - 0.5) * sy.expand(-1, num_points, -1)[mask_z].squeeze(-1)
            vals[:, 2] = sign[mask_z] * sz.expand(-1, num_points, -1)[mask_z].squeeze(-1) / 2
            points[mask_z] = vals
            
            norms = torch.zeros((k, 3), device=self.device)
            norms[:, 2] = sign[mask_z]
            normals[mask_z] = norms
            
        return points, normals

class Box(SurfaceShape):
    """ Hollow Box (Well) with 10 surfaces. Uses structured grid sampling to ensure full occlusion coverage at distance. """
    def sample_surface(self, num_points, num_envs, params):
        sx = params[:, 0]
        sy = params[:, 1]
        sz = params[:, 2]
        
        # We have 10 surfaces. Allocate points roughly equally.
        pts_per_surf = num_points // 10
        if pts_per_surf < 1: pts_per_surf = 1
        
        # We use a grid for each surface: sqrt(pts_per_surf) x sqrt(pts_per_surf)
        grid_size = int(np.sqrt(pts_per_surf))
        if grid_size < 1: grid_size = 1
        actual_pts_per_surf = grid_size * grid_size
        
        # Generate grid [-0.5, 0.5]
        lin_range = torch.linspace(-0.5, 0.5, grid_size, device=self.device)
        u_grid, v_grid = torch.meshgrid(lin_range, lin_range, indexing='ij')
        u_grid = u_grid.flatten() # [G*G]
        v_grid = v_grid.flatten()
        
        # Total points we will generate
        total_gen = actual_pts_per_surf * 10
        points = torch.zeros((num_envs, total_gen, 3), device=self.device)
        normals = torch.zeros((num_envs, total_gen, 3), device=self.device)
        
        # Helper to fill faces
        def fill_face(start_idx, p_base, n_base, sx_v, sy_v, sz_v, axes):
            # axes: e.g. [0, 1, 2] means point = [const, u, v]
            # p_base: [N, 1] constant value for the first axis
            # n_base: [N, 3] normal
            end_idx = start_idx + actual_pts_per_surf
            
            p_batch = torch.zeros((num_envs, actual_pts_per_surf, 3), device=self.device)
            # Add a tiny random jitter to the grid to avoid perfect alignment artifacts
            jitter = (torch.rand((num_envs, actual_pts_per_surf), device=self.device) - 0.5) / grid_size
            u_jittered = u_grid.unsqueeze(0) + jitter
            jitter2 = (torch.rand((num_envs, actual_pts_per_surf), device=self.device) - 0.5) / grid_size
            v_jittered = v_grid.unsqueeze(0) + jitter2
            
            p_batch[..., axes[0]] = p_base.unsqueeze(1)
            p_batch[..., axes[1]] = u_jittered * sx_v.unsqueeze(1)
            p_batch[..., axes[2]] = v_jittered * sy_v.unsqueeze(1)
            
            points[:, start_idx:end_idx, :] = p_batch
            normals[:, start_idx:end_idx, :] = n_base.unsqueeze(1)
            return end_idx

        # sx, sy, sz are [N, 1]
        # Front Out/In
        curr = fill_face(0, sx/2, torch.stack([torch.ones_like(sx), torch.zeros_like(sx), torch.zeros_like(sx)], dim=-1), sy, sz, None, [0, 1, 2])
        curr = fill_face(curr, sx/2, torch.stack([-torch.ones_like(sx), torch.zeros_like(sx), torch.zeros_like(sx)], dim=-1), sy, sz, None, [0, 1, 2])
        # Back Out/In
        curr = fill_face(curr, -sx/2, torch.stack([-torch.ones_like(sx), torch.zeros_like(sx), torch.zeros_like(sx)], dim=-1), sy, sz, None, [0, 1, 2])
        curr = fill_face(curr, -sx/2, torch.stack([torch.ones_like(sx), torch.zeros_like(sx), torch.zeros_like(sx)], dim=-1), sy, sz, None, [0, 1, 2])
        # Left Out/In
        curr = fill_face(curr, sy/2, torch.stack([torch.zeros_like(sy), torch.ones_like(sy), torch.zeros_like(sy)], dim=-1), sx, sz, None, [1, 0, 2])
        curr = fill_face(curr, sy/2, torch.stack([torch.zeros_like(sy), -torch.ones_like(sy), torch.zeros_like(sy)], dim=-1), sx, sz, None, [1, 0, 2])
        # Right Out/In
        curr = fill_face(curr, -sy/2, torch.stack([torch.zeros_like(sy), -torch.ones_like(sy), torch.zeros_like(sy)], dim=-1), sx, sz, None, [1, 0, 2])
        curr = fill_face(curr, -sy/2, torch.stack([torch.zeros_like(sy), torch.ones_like(sy), torch.zeros_like(sy)], dim=-1), sx, sz, None, [1, 0, 2])
        # Bot Out/In
        curr = fill_face(curr, -sz/2, torch.stack([torch.zeros_like(sz), torch.zeros_like(sz), -torch.ones_like(sz)], dim=-1), sx, sy, None, [2, 0, 1])
        curr = fill_face(curr, -sz/2, torch.stack([torch.zeros_like(sz), torch.zeros_like(sz), torch.ones_like(sz)], dim=-1), sx, sy, None, [2, 0, 1])

        # If we didn't generate exactly num_points, pad or clip
        if total_gen < num_points:
            padding = num_points - total_gen
            points = torch.cat([points, points[:, :padding, :]], dim=1)
            normals = torch.cat([normals, normals[:, :padding, :]], dim=1)
        else:
            points = points[:, :num_points, :]
            normals = normals[:, :num_points, :]

        return points, normals

class Ellipsoid(SurfaceShape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [radius_x, radius_y, radius_z] (semi-axes)
        # Note: input params might be diameters (dims), so we divide by 2 if they are interpreted as such.
        # In legged_robot_nav, we see for cuboid/box it uses x, y, z which are full lengths.
        # For sphere, it uses radius directly.
        # Let's assume input params are full diameters (x, y, z) to be consistent with cuboids/box logic, 
        # or semi-axes?
        # Looking at legged_robot_nav for cuboid: dims are size_x, size_y, size_z.
        # So it's best to interpret params as diameters (full extent).
        
        rx = params[:, 0].view(-1, 1, 1) / 2.0
        ry = params[:, 1].view(-1, 1, 1) / 2.0
        rz = params[:, 2].view(-1, 1, 1) / 2.0
        
        # Sample points on unit sphere
        # Use simple rejection sampling or normalizing Gaussian
        normal_pre = torch.randn((num_envs, num_points, 3), device=self.device)
        unit_sphere = normal_pre / torch.norm(normal_pre, dim=-1, keepdim=True)
        
        # Scale by semi-axes to get points on ellipsoid
        # P = (rx*x, ry*y, rz*z)
        x = unit_sphere[:, :, 0:1]
        y = unit_sphere[:, :, 1:2]
        z = unit_sphere[:, :, 2:3]
        
        points = torch.cat([x * rx, y * ry, z * rz], dim=-1)
        
        # Compute normals
        # For ellipsoid (x/a)^2 + ... = 1, gradient is (2x/a^2, 2y/b^2, 2z/c^2)
        # Normal is proportional to (x/a^2, y/b^2, z/c^2)
        # Here x, y, z are the coordinates on ellipsoid.
        nx = points[:, :, 0:1] / (rx**2)
        ny = points[:, :, 1:2] / (ry**2)
        nz = points[:, :, 2:3] / (rz**2)
        
        normal_unnormalized = torch.cat([nx, ny, nz], dim=-1)
        normals = normal_unnormalized / torch.norm(normal_unnormalized, dim=-1, keepdim=True)
        
        return points, normals

class Cylinder(SurfaceShape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [radius, height]
        radius = params[:, 0]
        height = params[:, 1]
        
        # Areas
        area_side = 2 * np.pi * radius * height
        area_caps = 2 * (np.pi * radius**2)
        total_area = area_side + area_caps
        p_side = area_side / total_area
        
        radius = radius.view(-1, 1, 1)
        height = height.view(-1, 1, 1)
        p_side = p_side.view(-1, 1)
        
        rand_face = torch.rand((num_envs, num_points), device=self.device)
        mask_side = rand_face < p_side
        mask_caps = ~mask_side
        
        points = torch.zeros((num_envs, num_points, 3), device=self.device)
        normals = torch.zeros((num_envs, num_points, 3), device=self.device)
        
        # Side
        if mask_side.any():
            k = mask_side.sum()
            theta = torch.rand(k, device=self.device) * 2 * np.pi
            z = (torch.rand(k, device=self.device) - 0.5) * height.expand(-1, num_points, -1)[mask_side].squeeze(-1)
            
            r = radius.expand(-1, num_points, -1)[mask_side].squeeze(-1)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            
            vals = torch.zeros((k, 3), device=self.device)
            vals[:, 0] = x
            vals[:, 1] = y
            vals[:, 2] = z
            points[mask_side] = vals
            
            # Normal is radial (x, y, 0) normalized
            norms = torch.zeros((k, 3), device=self.device)
            norms[:, 0] = torch.cos(theta)
            norms[:, 1] = torch.sin(theta)
            norms[:, 2] = 0.0
            normals[mask_side] = norms
            
        # Caps
        if mask_caps.any():
            k = mask_caps.sum()
            # Sample on disk
            theta = torch.rand(k, device=self.device) * 2 * np.pi
            u = torch.rand(k, device=self.device)
            r = radius.expand(-1, num_points, -1)[mask_caps].squeeze(-1) * torch.sqrt(u)
            
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            
            # Top or Bottom
            sign = torch.sign(torch.rand(k, device=self.device) - 0.5)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            z = sign * height.expand(-1, num_points, -1)[mask_caps].squeeze(-1) / 2
            
            vals = torch.zeros((k, 3), device=self.device)
            vals[:, 0] = x
            vals[:, 1] = y
            vals[:, 2] = z
            points[mask_caps] = vals
            
            norms = torch.zeros((k, 3), device=self.device)
            norms[:, 2] = sign
            normals[mask_caps] = norms
            
        return points, normals


class CylinderWell(SurfaceShape):
    """ Hollow Cylinder (Well, similar to Box) with 4 surfaces: Outer Side, Inner Side, Outer Bottom, Inner Bottom. """
    def sample_surface(self, num_points, num_envs, params):
        # params: [radius, height]
        radius = params[:, 0]
        height = params[:, 1]
        
        pts_per_surf = num_points // 4
        if pts_per_surf < 1: pts_per_surf = 1
        
        # We will generate 4 * pts_per_surf points
        total_gen = pts_per_surf * 4
        points = torch.zeros((num_envs, total_gen, 3), device=self.device)
        normals = torch.zeros((num_envs, total_gen, 3), device=self.device)
        
        curr = 0
        def fill_cyl_side(start_idx, r_val, h_val, normal_sign):
            end_idx = start_idx + pts_per_surf
            k = pts_per_surf
            
            theta = torch.rand((num_envs, k), device=self.device) * 2 * np.pi
            z = (torch.rand((num_envs, k), device=self.device) - 0.5) * h_val.view(-1, 1).expand(-1, k)
            
            r = r_val.view(-1, 1).expand(-1, k)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            
            p_batch = torch.stack([x, y, z], dim=-1)
            
            # Normal
            n_x = torch.cos(theta) * normal_sign
            n_y = torch.sin(theta) * normal_sign
            n_z = torch.zeros_like(n_x)
            n_batch = torch.stack([n_x, n_y, n_z], dim=-1)
            
            points[:, start_idx:end_idx] = p_batch
            normals[:, start_idx:end_idx] = n_batch
            return end_idx
        
        def fill_cyl_bottom(start_idx, r_val, h_val, normal_z):
            end_idx = start_idx + pts_per_surf
            k = pts_per_surf
            
            theta = torch.rand((num_envs, k), device=self.device) * 2 * np.pi
            u = torch.rand((num_envs, k), device=self.device)
            r = r_val.view(-1, 1).expand(-1, k) * torch.sqrt(u)
            
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = -0.5 * h_val.view(-1, 1).expand(-1, k)
            
            p_batch = torch.stack([x, y, z], dim=-1)
            
            n_x = torch.zeros_like(x)
            n_y = torch.zeros_like(y)
            n_z = torch.ones_like(z) * normal_z
            n_batch = torch.stack([n_x, n_y, n_z], dim=-1)

            points[:, start_idx:end_idx] = p_batch
            normals[:, start_idx:end_idx] = n_batch
            return end_idx

        curr = fill_cyl_side(curr, radius, height, 1.0) # Outer
        curr = fill_cyl_side(curr, radius, height, -1.0) # Inner
        curr = fill_cyl_bottom(curr, radius, height, -1.0) # Outer Bottom (facing down)
        curr = fill_cyl_bottom(curr, radius, height, 1.0) # Inner Bottom (facing up/inside)

        # Pad or Clip
        if total_gen < num_points:
            padding = num_points - total_gen
            points = torch.cat([points, points[:, :padding, :]], dim=1)
            normals = torch.cat([normals, normals[:, :padding, :]], dim=1)
        else:
            points = points[:, :num_points, :]
            normals = normals[:, :num_points, :]
            
        return points, normals
