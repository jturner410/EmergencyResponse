import matplotlib.pyplot as plt
import numpy as np
import os
from swarm_simulation_v5_lattice import SwarmSimulation, Vec2

def generate_obstacle_heatmap(num_simulations=50, grid_size=1500, resolution=20):
    # Create a grid to store obstacle counts
    grid_cells = grid_size // resolution
    heatmap = np.zeros((grid_cells, grid_cells))
    
    # Directory for temporary simulation output files
    temp_dir = "temp_sim_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run multiple simulations and aggregate obstacle positions
    for sim_num in range(num_simulations):
        print(f"Running simulation {sim_num+1}/{num_simulations}")
        
        # Create a new simulation
        sim = SwarmSimulation(num_agents=7, steps=10)  # Reduced steps since we only need obstacle positions
        
        # Add the fixed obstacle that's always at the same position
        sim.add_obstacle(Vec2(500, 500), 60.0, 60000.0, 180.0)
        
        # Add random obstacles as in the original simulation
        goal = Vec2(1300, 1300)
        spawn = Vec2(50, 50)
        min_distance_to_goal = 200.0
        min_distance_to_spawn = 200.0
        count = 0
        
        while count < 10:
            x = np.random.uniform(100, 1400)
            y = np.random.uniform(100, 1400)
            pos = Vec2(x, y)
            if pos.distance_to(goal) >= min_distance_to_goal and pos.distance_to(spawn) >= min_distance_to_spawn:
                radius = np.random.uniform(40, 80)
                sensing_range = np.random.uniform(400, 700)
                sim.add_obstacle(Vec2(x, y), radius, 60000.0, sensing_range)
                count += 1
        
        # Output file for this simulation
        output_file = os.path.join(temp_dir, f"sim_{sim_num}.txt")
        
        # Run the simulation - this will generate obstacles according to the logic in the simulation file
        sim.run(output_file)
        
        # Read the obstacle data from the output file
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Extract obstacle information
        for line in lines:
            if line.startswith('#OBSTACLE'):
                parts = line.strip().split('\t')
                if len(parts) >= 5:  # Make sure we have enough parts
                    _, x, y, radius, sensing_range = parts
                    x, y, radius = float(x), float(y), float(radius)
                    
                    # Add this obstacle to our heatmap
                    grid_x = int(x // resolution)
                    grid_y = int(y // resolution)
                    if 0 <= grid_x < grid_cells and 0 <= grid_y < grid_cells:
                        # Add weight based on obstacle radius (larger obstacles have more impact)
                        weight = radius / 40.0  # Normalize by minimum radius
                        heatmap[grid_y, grid_x] += weight
                        
                        # Also add some weight to surrounding cells based on radius
                        radius_cells = int(radius // resolution) + 1
                        for dx in range(-radius_cells, radius_cells + 1):
                            for dy in range(-radius_cells, radius_cells + 1):
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < grid_cells and 0 <= ny < grid_cells:
                                    dist = np.sqrt(dx**2 + dy**2) * resolution
                                    if dist <= radius:
                                        decay = 1.0 - (dist / radius)
                                        heatmap[ny, nx] += weight * decay * 0.5
    
    # Clean up temporary files
    for sim_num in range(num_simulations):
        temp_file = os.path.join(temp_dir, f"sim_{sim_num}.txt")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    
    # Flip the heatmap vertically to match the coordinate system
    # In matplotlib, origin='lower' means (0,0) is at the bottom-left
    plt.imshow(heatmap, cmap='hot', interpolation='gaussian', origin='lower', 
               extent=[0, grid_size, 0, grid_size], vmin=0)
    
    plt.colorbar(label='Obstacle Density')
    
    # Mark the goal and spawn positions
    plt.scatter(1300, 1300, color='green', s=100, marker='*', label='Goal')
    plt.scatter(50, 50, color='blue', s=100, marker='o', label='Spawn')
    
    # Add the fixed obstacle marker
    plt.scatter(500, 500, color='black', s=100, marker='x', label='Fixed Obstacle')
    
    plt.title(f'Obstacle Heatmap (Aggregated over {num_simulations} Simulations)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the heatmap
    plt.savefig('obstacle_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as 'obstacle_heatmap.png'")
    
    plt.show()

if __name__ == "__main__":
    # You can adjust these parameters as needed
    generate_obstacle_heatmap(num_simulations=20, resolution=20)