import torch
from scipy import signal # Unused as I manually created the waveforms
import numpy as np
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Calc:
    @staticmethod
    def curl(vector_field):
        dFdy, dFdx, dFdz = torch.gradient(vector_field, axis=(1, 2, 3))
        curl_x = dFdz[1] - dFdy[2]
        curl_y = dFdx[2] - dFdz[0]
        curl_z = dFdy[0] - dFdx[1]
        return torch.stack([curl_x, curl_y, curl_z])

class Object:
    def __init__(self, position, size):
        self.position = position
        self.size = size

class MaterialObject(Object):
    def __init__(self, position, size, permittivity, permeability, reflection_coefficient):
        super().__init__(position, size)
        self.permittivity = permittivity
        self.permeability = permeability
        self.reflection_coefficient = reflection_coefficient

class Sphere(MaterialObject):
    def __init__(self, position, radius, permittivity, permeability, reflection_coefficient):
        super().__init__(position, size=(radius, radius, radius), permittivity=permittivity, permeability=permeability, reflection_coefficient=reflection_coefficient)

    def in_object(self, x, y, z):
        dx, dy, dz = x - self.position[0], y - self.position[1], z - self.position[2]
        return dx**2 + dy**2 + dz**2 <= self.size[0]**2



class Grid:
    def __init__(self, shape, min_wavelength, time_steps, permittivity, permeability, device):
        self.shape = shape
        self.time_steps = time_steps
        self.dx = min_wavelength / 20  # spatial resolution
        self.dt = self.dx / (2 * np.sqrt(3))  # temporal resolution based on CFL condition

        # Initialize fields
        self.E = torch.zeros((3,) + shape, device=device)  # Electric field
        self.H = torch.zeros((3,) + shape, device=device)  # Magnetic field

        # Material properties
        self.permittivity = torch.ones(shape, device=device) * permittivity
        self.permeability = torch.ones(shape, device=device) * permeability

    def apply_pec(self):
        # Set the electric field to zero at the boundaries
        self.E[:, 0, :, :] = 0
        self.E[:, -1, :, :] = 0
        self.E[:, :, 0, :] = 0
        self.E[:, :, -1, :] = 0
        self.E[:, :, :, 0] = 0
        self.E[:, :, :, -1] = 0

class Source:
    def __init__(self, position, frequency, amplitude, kind, planar=False, circular_motion=False, radius=0, speed=0, centre_y=0, centre_z=0):
        self.position = list(position)
        self.frequency = frequency
        self.amplitude = amplitude
        self.kind = kind  # kind of source (e.g "gaussian", "sine", "square")
        self.planar = planar  # whether the source is planar
        self.circular_motion = circular_motion  # whether the source moves in a circular path
        self.radius = radius  # radius of the circular path
        self.speed = speed  # speed of the circular motion
        self.centre_y = centre_y
        self.centre_z = centre_z


    def update_position(self, t):
        if self.circular_motion:
            # Update the y and z coordinates of the source to move it in a circular path in the yz-plane
            self.position[1] = self.centre_y + self.radius * np.cos(self.speed * t)
            self.position[2] = self.centre_z + self.radius * np.sin(self.speed * t)


    def waveform(self, t):
        if self.kind == "gaussian":
            return np.exp(-(t - 1 / self.frequency) ** 2) * self.amplitude
        elif self.kind == "sine":
            return np.sin(2 * np.pi * self.frequency * t) * self.amplitude*10**14
        elif self.kind == "square":
            return np.sign(np.sin(2 * np.pi * self.frequency * t)) * self.amplitude
        else:
            raise ValueError(f"Unknown source kind: {self.kind}")

    def inject(self, grid, t):
        if self.planar:
            # If the source is planar, add the waveform to all points along the yz-plane at the source's x-position
            grid.E[:-1, :-1, :-1, int(self.position[2])] += self.waveform(t)
        else:
            # If the source is not planar, add the waveform at the source's position
            grid.E[:, self.position[0], int(self.position[1]), int(self.position[2])] += self.waveform(t)


class Detector:
    def __init__(self, position):
        self.position = position
        self.data = []

    def record(self, grid):
        # Record the field values at the detector position
        self.data.append(grid.E[:, self.position[0], self.position[1], self.position[2]])

class Simulation:
    def __init__(self, grid, objects, sources, detectors):
        self.grid = grid
        self.objects = objects
        self.sources = sources
        self.detectors = detectors
        self.t = 0
        self.cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["blue", "white", "red"])

    def run(self):
        # Create directories if they don't exist
        if not os.path.exists('frames'):
            os.makedirs('frames')
        if not os.path.exists('gifs'):
            os.makedirs('gifs')

        for _ in range(self.grid.time_steps):
            self.step()
            
        # Create a GIF from the frames
        images = []
        for t in range(self.grid.time_steps):
            filename = f'frames/slab_{t+1}.png'
            images.append(imageio.imread(filename))
            os.remove(filename)  # Delete the frame
        imageio.mimsave(f'gifs/simulation_run{len(os.listdir("gifs")) + 1}.gif', images, duration=0.01)
    
    def step(self):
        # Update electric field
        curl_H = Calc.curl(self.grid.H)
        self.grid.E += self.grid.dt / self.grid.permittivity * curl_H

        # Apply PEC boundary conditions
        self.grid.apply_pec()

        # Make objects reflective
        for obj in self.objects:
            mask = obj.in_object(*torch.meshgrid([torch.arange(dim, device=self.grid.E.device) for dim in self.grid.E.shape[1:]]))
            self.grid.E[:, mask] *= obj.reflection_coefficient
        
        # Inject sources
        for source in self.sources:
            source.inject(self.grid, self.t)
            source.update_position(self.t)


        # Update magnetic field
        curl_E = Calc.curl(self.grid.E)
        self.grid.H -= self.grid.dt / self.grid.permeability * curl_E

        # Record detectors
        for detector in self.detectors:
            detector.record(self.grid)

        # Increment time
        self.t += 1
        print(self.t)
        # Visualize the slab
        slab = self.grid.E[:, self.grid.shape[0] // 2, :, :]
        plt.imshow(slab[0].cpu().numpy(), cmap=self.cmap, vmin=-1, vmax=1)  # Visualize the x-component of the electric field
        plt.title(f'Frame {self.t}')  # Add the frame step as the title
        plt.savefig(f'frames/slab_{self.t}.png')  # Save the figure in the frames subfolder
        plt.close()


# Set up the simulation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
grid = Grid(shape=(200, 200, 200), min_wavelength=20, time_steps=600, permittivity=1, permeability=1, device=device)
source1 = Source(position=(100, 50, 50), frequency=0.5, amplitude=10, kind="sine", planar=True, circular_motion=False, radius=40, speed=0.1, centre_y=100, centre_z=100)
#sphere1 = Sphere(position=(100, 50, 50), radius=20, permittivity=80, permeability=1, reflection_coefficient=0.9)
sphere2 = Sphere(position=(100, 100, 100), radius=40, permittivity=80, permeability=1, reflection_coefficient=0.9)


simulation = Simulation(grid=grid, objects=[sphere2], sources=[source1], detectors=[])

simulation.run()
