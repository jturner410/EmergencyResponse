


import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other): return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar): return Vec2(self.x * scalar, self.y * scalar)
    def __truediv__(self, scalar): return Vec2(self.x / scalar, self.y / scalar)

    def length(self): return math.sqrt(self.x ** 2 + self.y ** 2)
    def normalized(self):
        l = self.length()
        return self / l if l > 1e-6 else Vec2(0, 0)
    def distance_to(self, other): return (self - other).length()

class Obstacle:
    def __init__(self, center, radius, strength, sensing_range):
        self.center = center
        self.radius = radius
        self.strength = strength
        self.sensing_range = sensing_range

class Agent:

    def __init__(self, position):
        self.position = position
        self.velocity = Vec2(0, 0)
        self.num_nearby_agents = 0
   
    
    def distance_to_agent(self, other_agent):
        return self.position.distance_to(other_agent.position)

class SwarmSimulation:
    def __init__(self, num_agents, steps):
        self.num_agents = num_agents
        self.steps = steps
        self.goal = Vec2(1300, 1300)
        self.env_size = 1500.0
        self.dt = 1.0
        self.desired_distance = 40.0
        self.k_spring = 0.03
        self.k_repel = 400.0
        self.repel_cutoff = 60.0
        self.damping = 0.30
        self.k_goal = 0.01
        self.max_force = 4.0
        self.max_velocity = 2.5
        
        # Add these missing lattice formation parameters
        self.lattice_angle_weight = 0.02  # New parameter for angle enforcement
        self.formation_weight = 0.015     # Weight for formation maintenance
        self.neighbor_alignment_weight = 0.01  # Weight for velocity alignment
        self.ideal_lattice_angles = [60, 120, 180, 240, 300]  # Ideal angles for hexagonal lattice
        
        self.agents = []
        self.obstacles = []
        
        # Initialize heatmap for obstacle aggregation
        self.grid_resolution = 20
        self.grid_cells = int(self.env_size // self.grid_resolution)
        self.obstacle_heatmap = np.zeros((self.grid_cells, self.grid_cells))
        
        self.initialize_agents()
        
    def initialize_agents(self):
        origin = Vec2(50, 50)
        spacing = self.desired_distance
        
        # Create lattice positions for different numbers of agents
        if self.num_agents == 1:
            positions = [Vec2(0, 0)]
        elif self.num_agents == 2:
            positions = [Vec2(-spacing/2, 0), Vec2(spacing/2, 0)]
        elif self.num_agents == 3:
            # Triangle formation
            positions = [
                Vec2(0, spacing * 0.577),  # Top vertex
                Vec2(-spacing/2, -spacing * 0.289),  # Bottom left
                Vec2(spacing/2, -spacing * 0.289)   # Bottom right
            ]
        elif self.num_agents == 4:
            # Square formation
            positions = [
                Vec2(-spacing/2, -spacing/2),
                Vec2(spacing/2, -spacing/2),
                Vec2(-spacing/2, spacing/2),
                Vec2(spacing/2, spacing/2)
            ]
        elif self.num_agents == 6:
            # Hexagon formation (6 agents around center)
            positions = []
            for i in range(6):
                angle = i * math.pi / 3  # 60 degrees apart
                x = spacing * math.cos(angle)
                y = spacing * math.sin(angle)
                positions.append(Vec2(x, y))
        elif self.num_agents == 7:
            # Hexagon with center agent
            positions = [Vec2(0, 0)]  # Center agent
            for i in range(6):
                angle = i * math.pi / 3  # 60 degrees apart
                x = spacing * math.cos(angle)
                y = spacing * math.sin(angle)
                positions.append(Vec2(x, y))
        else:
            # For larger numbers, create a triangular lattice
            positions = []
            row = 0
            col = 0
            agents_placed = 0
            
            while agents_placed < self.num_agents:
                # Calculate position in triangular lattice
                x = col * spacing + (row % 2) * (spacing / 2)
                y = row * spacing * 0.866  # sqrt(3)/2 for equilateral triangles
                
                positions.append(Vec2(x, y))
                agents_placed += 1
                
                col += 1
                # Move to next row when we've filled current row
                if col > row:
                    row += 1
                    col = 0
        
        # Create agents at calculated positions, offset by origin
        self.agents = []
        for pos in positions:
            final_pos = Vec2(origin.x + pos.x, origin.y + pos.y)
            self.agents.append(Agent(final_pos))

    def add_obstacle(self, center, radius, strength, sensing_range):
        self.obstacles.append(Obstacle(center, radius, strength, sensing_range))
        
        # Update the heatmap with this obstacle
        self.update_obstacle_heatmap(center.x, center.y, radius)
    
    def update_obstacle_heatmap(self, x, y, radius):
        """Update the obstacle heatmap with a new obstacle"""
        grid_x = int(x // self.grid_resolution)
        grid_y = int(y // self.grid_resolution)
        
        # Calculate how many grid cells the radius spans
        radius_cells = int(radius // self.grid_resolution) + 1
        
        # Apply weight to all cells within the obstacle's actual radius
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.grid_cells and 0 <= ny < self.grid_cells:
                    # Calculate the distance from this cell to the obstacle center
                    cell_center_x = (nx + 0.5) * self.grid_resolution
                    cell_center_y = (ny + 0.5) * self.grid_resolution
                    dist = math.sqrt((cell_center_x - x)**2 + (cell_center_y - y)**2)
                    
                    # If this cell is within the obstacle's radius, add full weight
                    if dist <= radius:
                        # Use a higher base weight for better visibility
                        weight = 5.0
                        self.obstacle_heatmap[ny, nx] += weight
                    # Add some decay effect just outside the radius for smoother visualization
                    elif dist <= radius * 1.2:
                        decay = 1.0 - ((dist - radius) / (radius * 0.2))
                        weight = 5.0 * decay
                        self.obstacle_heatmap[ny, nx] += weight
    
    def plot_obstacle_heatmap(self, simulation_count):
        """Plot the current state of the obstacle heatmap"""
        plt.figure(figsize=(10, 8))
        
        
        plt.imshow(self.obstacle_heatmap, cmap='hot', interpolation='gaussian', origin='lower', 
                   extent=[0, self.env_size, 0, self.env_size], vmin=0)
        
        plt.colorbar(label='Obstacle Density')
        
        # Mark the goal and spawn positions
        plt.scatter(1300, 1300, color='green', s=100, marker='*', label='Goal')
        plt.scatter(50, 50, color='blue', s=100, marker='o', label='Spawn')
        
        # Add the fixed obstacle marker
        plt.scatter(500, 500, color='black', s=100, marker='x', label='Fixed Obstacle')
        
        plt.title(f'Obstacle Heatmap (Aggregated over {simulation_count} Simulations)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the heatmap
        plt.savefig('obstacle_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Heatmap updated after {simulation_count} simulations and saved as 'obstacle_heatmap.png'")
        
        plt.close()  # Close the figure to avoid displaying it during simulation runs

    def get_agent_distance(self, agent_index1, agent_index2):
        if 0 <= agent_index1 < len(self.agents) and 0 <= agent_index2 < len(self.agents):
            return self.agents[agent_index1].distance_to_agent(self.agents[agent_index2])
        else:
            raise IndexError("Agent index out of range")
    
    def count_nearby_agents(self):
        for i, agent_i in enumerate(self.agents):
            nearby_count = 0
            for j, agent_j in enumerate(self.agents):
                if i == j:
                    continue
                distance = agent_i.distance_to_agent(agent_j)
                if 30.0 <= distance <= 50.0:
                    nearby_count += 1
            agent_i.num_nearby_agents = nearby_count
            
           


    def count_agents_near_goal(self, distance_threshold=100.0):
        # Count how many agents are within the specified distance from the goal
        count = 0
        for agent in self.agents:
            if agent.position.distance_to(self.goal) <= distance_threshold:
                count += 1
        return count

    def run(self, filename=None):
        if filename:
            with open(filename, "w") as f:
                for obs in self.obstacles:
                    f.write(f"#OBSTACLE\t{obs.center.x}\t{obs.center.y}\t{obs.radius}\t{obs.sensing_range}\n")
                step_count = 0
                for _ in range(self.steps):
                    if step_count < 500:
                        self.max_velocity = self.max_velocity
                    else:
                        self.max_velocity = self.max_velocity
                    step_count += 1
                    self.update()
                    
                    for i, agent in enumerate(self.agents):
                        f.write(f"{agent.position.x}\t{agent.position.y}")
                        if i < self.num_agents - 1:
                            f.write("\t")
                    f.write("\n")
        else:
            # Just run the simulation without writing to a file
            step_count = 0
            for _ in range(self.steps):
                if step_count < 500:
                    self.max_velocity = self.max_velocity
                else:
                    self.max_velocity = self.max_velocity
                step_count += 1
                self.update()
                

    def get_lattice_neighbors(self, agent_index, max_neighbors=6):
        #Get the closest neighbors for lattice formation
        agent = self.agents[agent_index]
        distances = []
        
        for i, other_agent in enumerate(self.agents):
            if i != agent_index:
                dist = agent.distance_to_agent(other_agent)
                if dist <= self.desired_distance * 1.5:  # Within lattice range
                    distances.append((dist, i))
        
        # Sort by distance and return closest neighbors
        distances.sort()
        return [idx for _, idx in distances[:max_neighbors]]
    
    def calculate_lattice_force(self, agent_index):
        # Calculate force to maintain proper lattice formation
        agent = self.agents[agent_index]
        lattice_force = Vec2(0, 0)
        neighbors = self.get_lattice_neighbors(agent_index)
        
        if len(neighbors) < 2:
            return lattice_force
        
        # Calculate angles between neighbors
        neighbor_angles = []
        for neighbor_idx in neighbors:
            neighbor = self.agents[neighbor_idx]
            delta = neighbor.position - agent.position
            angle = math.atan2(delta.y, delta.x) * 180 / math.pi
            if angle < 0:
                angle += 360
            neighbor_angles.append((angle, neighbor_idx))
        
        neighbor_angles.sort()
        
        # Apply forces to maintain ideal lattice angles
        for i in range(len(neighbor_angles)):
            current_angle, neighbor_idx = neighbor_angles[i]
            neighbor = self.agents[neighbor_idx]
            
            # Calculate ideal angle based on position in sorted list
            if len(neighbors) <= 6:
                ideal_angle = (360 / len(neighbors)) * i
            else:
                # For more than 6 neighbors, use hexagonal pattern
                ideal_angle = 60 * (i % 6)
            
            angle_diff = current_angle - ideal_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            # Apply corrective force perpendicular to neighbor direction
            if abs(angle_diff) > 5:  # Only apply if significant deviation
                to_neighbor = neighbor.position - agent.position
                perpendicular = Vec2(-to_neighbor.y, to_neighbor.x).normalized()
                correction_strength = self.lattice_angle_weight * angle_diff / 180.0
                lattice_force += perpendicular * correction_strength
        
        return lattice_force
    
    def calculate_formation_maintenance_force(self, agent_index):
        # Calculate force to maintain overall formation during movement
        agent = self.agents[agent_index]
        formation_force = Vec2(0, 0)
        neighbors = self.get_lattice_neighbors(agent_index)
        
        if len(neighbors) == 0:
            return formation_force
        
        # Calculate center of mass of neighbors
        neighbor_center = Vec2(0, 0)
        for neighbor_idx in neighbors:
            neighbor_center += self.agents[neighbor_idx].position
        neighbor_center /= len(neighbors)
        
        # Calculate ideal position relative to neighbor center
        to_center = neighbor_center - agent.position
        ideal_distance = self.desired_distance * 0.8  # Slightly closer to center
        
        if to_center.length() > ideal_distance:
            # Pull towards formation center if too far
            formation_force += to_center.normalized() * self.formation_weight
        elif to_center.length() < ideal_distance * 0.5:
            # Push away if too close to center
            formation_force -= to_center.normalized() * self.formation_weight
        
        return formation_force
    
    def calculate_neighbor_alignment_force(self, agent_index):
        # Calculate force to align velocity with nearby agents
        agent = self.agents[agent_index]
        alignment_force = Vec2(0, 0)
        neighbors = self.get_lattice_neighbors(agent_index)
        
        if len(neighbors) == 0:
            return alignment_force
        
        # Calculate average velocity of neighbors
        avg_velocity = Vec2(0, 0)
        for neighbor_idx in neighbors:
            avg_velocity += self.agents[neighbor_idx].velocity
        avg_velocity /= len(neighbors)
        
        # Apply alignment force
        velocity_diff = avg_velocity - agent.velocity
        alignment_force = velocity_diff * self.neighbor_alignment_weight
        
        return alignment_force

    def update(self):
        forces = [Vec2(0, 0) for _ in self.agents]
        for i, agent_i in enumerate(self.agents):
            pos_i = agent_i.position
            
            # Existing forces (spring and repulsion)
            for j, agent_j in enumerate(self.agents):
                if i == j: continue
                delta = agent_j.position - pos_i
                dist = delta.length()
                if dist < 1e-6: continue
                direction = delta.normalized()
                
                if dist < self.repel_cutoff:
                    repel_force = self.k_repel / (dist * dist)
                    forces[i] -= direction * repel_force
                
                spring_force = self.k_spring * (dist - self.desired_distance)
                forces[i] += direction * spring_force

            # Enhanced lattice formation forces
            lattice_force = self.calculate_lattice_force(i)
            formation_force = self.calculate_formation_maintenance_force(i)
            alignment_force = self.calculate_neighbor_alignment_force(i)
            
            forces[i] += lattice_force + formation_force + alignment_force

            # Obstacle avoidance
            for obs in self.obstacles:
                to_obs = pos_i - obs.center
                dist = to_obs.length()
                if 1e-6 < dist < (obs.sensing_range + 200):
                    direction = to_obs.normalized()
                    repel_force = obs.strength / (dist * dist)
                    forces[i] += direction * repel_force


            # Goal attraction and damping
            to_goal = self.goal - pos_i
            forces[i] += to_goal * self.k_goal
            forces[i] -= agent_i.velocity * self.damping
            
            if forces[i].length() > self.max_force:
                forces[i] = forces[i].normalized() * self.max_force

        for i, agent in enumerate(self.agents):
            agent.velocity += forces[i] * self.dt
            if agent.velocity.length() > self.max_velocity:
                agent.velocity = agent.velocity.normalized() * self.max_velocity
            agent.position += agent.velocity * self.dt
            agent.position.x = max(0, min(self.env_size, agent.position.x))
            agent.position.y = max(0, min(self.env_size, agent.position.y))


if __name__ == "__main__":
    # Create a single simulation instance that will be reused and accumulate heatmap data

    num_agents = 7
    sim = SwarmSimulation(num_agents, steps=1000)
    
    # Run simulation x times, then aggregate percentage of agents that reached the goal
    num_simulations = 10
    
    success_counter = 0
    
    for i in range(num_simulations):
        print(f"\nRunning simulation {i+1}/{num_simulations}")
        
        # Reset the simulation for a new run, but keep the heatmap data
        sim.agents = []
        sim.obstacles = []
        sim.initialize_agents()
        
        # Add a single obstacle at a fixed position
        sim.add_obstacle(Vec2(500, 500), 60.0, 60000.0, 180.0)
        
        # Avoid obstacles near goal and spawn
        goal = Vec2(1300, 1300)
        spawn = Vec2(50, 50)
        min_distance_to_goal = 200.0
        min_distance_to_spawn = 200.0
        count = 0
        
        # Add random obstacles
        while count < 10:
            x = random.uniform(100, 1400)
            y = random.uniform(100, 1400)
            pos = Vec2(x, y)
            if pos.distance_to(goal) >= min_distance_to_goal and pos.distance_to(spawn) >= min_distance_to_spawn:
                radius = random.uniform(40, 80)
                sensing_range = random.uniform(400, 700)
                sim.add_obstacle(Vec2(x, y), radius, 60000.0, sensing_range)
                count += 1
        
        # Run the simulation
        # Use a single output file or none if you don't need the position data
        output_file = "swarm_output.txt" if i == num_simulations-1 else None
        sim.run(output_file)
        
        # Count agents near goal
        agents_near_goal = sim.count_agents_near_goal(100.0)
        success_counter += agents_near_goal
        
        # Update and save the heatmap after each simulation
        sim.plot_obstacle_heatmap(i+1)
    
    # Calculate and display success percentage
    success_percentage = (success_counter / (num_agents * num_simulations)) * 100
    print(f"\nSuccess percentage after {num_simulations} simulations: {success_percentage:.2f}%")
    
    # Final display of the heatmap (this will show the plot)
    plt.figure(figsize=(10, 8))
    plt.imshow(sim.obstacle_heatmap, cmap='hot', interpolation='gaussian', origin='lower', 
               extent=[0, sim.env_size, 0, sim.env_size], vmin=0)
    plt.colorbar(label='Obstacle Density')
    plt.scatter(1300, 1300, color='green', s=100, marker='*', label='Goal')
    plt.scatter(50, 50, color='blue', s=100, marker='o', label='Spawn')
    plt.scatter(500, 500, color='black', s=100, marker='x', label='Fixed Obstacle')
    plt.title(f'Final Obstacle Heatmap (Aggregated over {num_simulations} Simulations)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
