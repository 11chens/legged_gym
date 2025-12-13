import torch
import numpy as np

class Shape:
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

class Sphere(Shape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [radius]
        radius = params[:, 0].view(-1, 1, 1)
        
        # Sample unit vectors (normal)
        normal = torch.randn((num_envs, num_points, 3), device=self.device)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        
        # Points are on the surface
        points = radius * normal
        
        return points, normal

class Cuboid(Shape):
    def sample_surface(self, num_points, num_envs, params):
        # params: [size_x, size_y, size_z]
        sx = params[:, 0]
        sy = params[:, 1]
        sz = params[:, 2]
        
        # Areas of faces
        # Front/Back (xy): sx*sy
        # Left/Right (yz): sy*sz
        # Top/Bottom (xz): sx*sz
        # Wait, standard cuboid usually aligned with axes.
        # Faces:
        # +/- X: size y * size z
        # +/- Y: size x * size z
        # +/- Z: size x * size y
        
        area_x = sy * sz
        area_y = sx * sz
        area_z = sx * sy
        
        total_area = 2 * (area_x + area_y + area_z)
        
        # Probabilities for each pair of faces
        p_x = (2 * area_x) / total_area
        p_y = (2 * area_y) / total_area
        p_z = (2 * area_z) / total_area
        
        # We need to sample faces for each point.
        # This is tricky to vectorize perfectly if we want exact counts, but random sampling is fine.
        # We can generate random numbers [0, 1] and threshold.
        
        # Expand for broadcasting
        sx = sx.view(-1, 1, 1)
        sy = sy.view(-1, 1, 1)
        sz = sz.view(-1, 1, 1)
        p_x = p_x.view(-1, 1)
        p_y = p_y.view(-1, 1)
        
        # Random choice of face axis: 0=X, 1=Y, 2=Z
        # We use cumulative probs
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
        # x = +/- sx/2
        # y = uniform(-sy/2, sy/2)
        # z = uniform(-sz/2, sz/2)
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

class Cylinder(Shape):
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
