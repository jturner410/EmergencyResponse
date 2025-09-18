
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_swarm(filename, steps_to_plot=1000):
    with open(filename, 'r') as f:
        lines = f.readlines()

    obstacle_lines = [line for line in lines if line.startswith('#OBSTACLE')]
    data_lines = [line for line in lines if not line.startswith('#')]

    obstacles = []
    for line in obstacle_lines:
        _, x, y, r, _ = line.strip().split('\t')
        obstacles.append((float(x), float(y), float(r)))

    positions = []
    for line in data_lines[:steps_to_plot]:
        tokens = line.strip().split('\t')
        step_positions = [(float(tokens[i]), float(tokens[i+1])) for i in range(0, len(tokens), 2)]
        positions.append(step_positions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1500)

    for (x, y, r) in obstacles:
        circle = patches.Circle((x, y), r, color='red', alpha=0.3)
        ax.add_patch(circle)

    goal = (1300, 1300)
    ax.add_patch(patches.Rectangle((goal[0]-7.5, goal[1]-7.5), 15, 15, color='green'))

    for step in positions:
        xs, ys = zip(*step)
        ax.clear()
        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1500)
        for (x, y, r) in obstacles:
            circle = patches.Circle((x, y), r, color='red', alpha=0.3)
            ax.add_patch(circle)
        ax.add_patch(patches.Rectangle((goal[0]-7.5, goal[1]-7.5), 15, 15, color='green'))
        ax.scatter(xs, ys, color='blue', s=10)
        plt.pause(0.01)

    # Create final labeled image
    plt.figure(figsize=(10, 8))
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    
    # Draw obstacles
    for (x, y, r) in obstacles:
        circle = patches.Circle((x, y), r, color='red', alpha=0.3, label='Obstacles' if obstacles.index((x, y, r)) == 0 else "")
        plt.gca().add_patch(circle)
    
    # Draw goal
    plt.gca().add_patch(patches.Rectangle((goal[0]-7.5, goal[1]-7.5), 15, 15, color='green', label='Goal'))
    
    # Draw final positions with labels
'''    if positions:
        final_positions = positions[-1]  # Get last step positions
        for i, (x, y) in enumerate(final_positions):
            plt.scatter(x, y, color='blue', s=100, zorder=5)
            plt.annotate(f'Agent {i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.title('Final Swarm Formation with Labeled Agents', fontsize=14, fontweight='bold')
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the final image
    plt.savefig('final_swarm_formation.png', dpi=300, bbox_inches='tight')
    print("Final labeled image saved as 'final_swarm_formation.png'")
    
    plt.show()'''

if __name__ == "__main__":
    visualize_swarm("swarm_output.txt", steps_to_plot=1000)
