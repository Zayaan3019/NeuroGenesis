import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import logging
from typing import Dict, List, Tuple
import random

from neural_fabric import NeuralFabric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralFabric-Example")

class PatternRecognitionDemo:
    """
    Demonstrates NeuralFabric's pattern recognition capabilities.
    """
    
    def __init__(self, input_size: int = 100, num_patterns: int = 5):
        """
        Initialize the pattern recognition demo.
        
        Args:
            input_size: Size of the input patterns
            num_patterns: Number of patterns to generate
        """
        self.input_size = input_size
        self.num_patterns = num_patterns
        
        # Create the neural fabric
        self.fabric = NeuralFabric("PatternRecognizer")
        
        # Create regions
        self.input_region_id = self.fabric.create_sensory_region("Input", size=input_size)
        self.hidden_region_id = self.fabric.create_processing_region("Hidden", size=200)
        self.output_region_id = self.fabric.create_output_region("Output", size=num_patterns)
        
        # Connect regions with random connections
        self.fabric.connect_regions(self.input_region_id, self.hidden_region_id, connection_density=0.2)
        self.fabric.connect_regions(self.hidden_region_id, self.output_region_id, connection_density=0.3)
        
        # Generate patterns
        self.patterns = self._generate_patterns(num_patterns, input_size)
        
        # Tracking
        self.current_pattern_idx = 0
        self.recognition_history = []
        
        logger.info(f"Created pattern recognition demo with {num_patterns} patterns of size {input_size}")
    
    def _generate_patterns(self, num_patterns: int, pattern_size: int) -> List[Dict[int, float]]:
        """
        Generate random patterns for recognition.
        
        Args:
            num_patterns: Number of patterns to generate
            pattern_size: Size of each pattern
            
        Returns:
            List of patterns (dict mapping indices to values)
        """
        patterns = []
        
        for i in range(num_patterns):
            # Generate a random pattern with 20% active neurons
            active_indices = random.sample(range(pattern_size), int(pattern_size * 0.2))
            
            pattern = {}
            for idx in active_indices:
                pattern[idx] = 0.8 + 0.2 * random.random()  # Values between 0.8 and 1.0
            
            patterns.append(pattern)
            
        return patterns
    
    def start(self) -> None:
        """Start the neural fabric and pattern recognition."""
        self.fabric.start()
        
        # Start pattern presentation thread
        self.is_running = True
        self.pattern_thread = threading.Thread(target=self._present_patterns)
        self.pattern_thread.daemon = True
        self.pattern_thread.start()
        
        logger.info("Started pattern recognition demo")
    
    def stop(self) -> None:
        """Stop the pattern recognition and neural fabric."""
        self.is_running = False
        
        if hasattr(self, 'pattern_thread'):
            self.pattern_thread.join(timeout=1.0)
        
        self.fabric.stop()
        
        logger.info("Stopped pattern recognition demo")
    
    def _present_patterns(self) -> None:
        """Present patterns to the neural fabric."""
        # Initial learning phase
        logger.info("Starting learning phase")
        
        learning_cycles = 20
        for cycle in range(learning_cycles):
            for i, pattern in enumerate(self.patterns):
                if not self.is_running:
                    return
                
                # Present pattern
                self.fabric.inject_pattern(self.input_region_id, pattern, strength=1.0)
                
                # Wait for processing
                time.sleep(2.0)
                
                # Get output activity
                output_activity = self.fabric.get_output_activity(self.output_region_id)
                
                # Log progress
                if cycle == 0 or cycle == learning_cycles - 1:
                    logger.info(f"Cycle {cycle+1}/{learning_cycles}, Pattern {i+1}, "
                               f"Output: {len(output_activity)} neurons active")
        
        logger.info("Learning phase complete, starting testing phase")
        
        # Testing phase
        while self.is_running:
            # Select a random pattern
            pattern_idx = random.randint(0, self.num_patterns - 1)
            pattern = self.patterns[pattern_idx]
            
            # Add noise to the pattern
            noisy_pattern = self._add_noise(pattern, noise_level=0.2)
            
            # Present pattern
            self.fabric.inject_pattern(self.input_region_id, noisy_pattern, strength=1.0)
            
            # Wait for processing
            time.sleep(3.0)
            
            # Get output activity
            output_activity = self.fabric.get_output_activity(self.output_region_id)
            
            # Analyze response
            recognized_pattern = self._analyze_output(output_activity)
            
            # Track recognition accuracy
            is_correct = recognized_pattern == pattern_idx
            self.recognition_history.append(is_correct)
            
            # Log result
            logger.info(f"Presented noisy pattern {pattern_idx}, "
                       f"Recognized as pattern {recognized_pattern}, "
                       f"Correct: {is_correct}")
            
            # Calculate recognition rate
            if len(self.recognition_history) > 10:
                recent_rate = sum(self.recognition_history[-10:]) / 10
                logger.info(f"Recent recognition rate: {recent_rate:.2f}")
            
            # Wait before next pattern
            time.sleep(2.0)
    
    def _add_noise(self, pattern: Dict[int, float], noise_level: float) -> Dict[int, float]:
        """
        Add noise to a pattern.
        
        Args:
            pattern: Original pattern
            noise_level: Amount of noise to add (0-1)
            
        Returns:
            Noisy pattern
        """
        noisy_pattern = pattern.copy()
        
        # Remove some active neurons
        neurons_to_remove = int(len(pattern) * noise_level)
        if neurons_to_remove > 0:
            for idx in random.sample(list(pattern.keys()), neurons_to_remove):
                del noisy_pattern[idx]
        
        # Add some random activations
        neurons_to_add = int(len(pattern) * noise_level)
        available_indices = [i for i in range(self.input_size) if i not in noisy_pattern]
        
        if neurons_to_add > 0 and available_indices:
            for idx in random.sample(available_indices, min(neurons_to_add, len(available_indices))):
                noisy_pattern[idx] = 0.7 + 0.3 * random.random()
        
        return noisy_pattern
    
    def _analyze_output(self, output_activity: Dict[str, float]) -> int:
        """
        Analyze output activity to determine recognized pattern.
        
        Args:
            output_activity: Output neuron activity levels
            
        Returns:
            Index of the recognized pattern
        """
        if not output_activity:
            return random.randint(0, self.num_patterns - 1)
        
        # Group neurons by their position in the output layer
        regions = self.fabric.regions
        output_region = regions[self.output_region_id]
        output_neurons = list(output_region.neurons.values())
        
        # Calculate activation for each pattern group
        pattern_activations = [0.0] * self.num_patterns
        
        # Simple approach: divide neurons equally among patterns
        neurons_per_pattern = len(output_neurons) // self.num_patterns
        
        for i, neuron_id in enumerate(output_activity.keys()):
            pattern_idx = min(i // neurons_per_pattern, self.num_patterns - 1)
            pattern_activations[pattern_idx] += output_activity[neuron_id]
        
        # Normalize by number of neurons
        for i in range(self.num_patterns):
            pattern_activations[i] = pattern_activations[i] / (neurons_per_pattern or 1)
        
        # Return the pattern with highest activation
        return np.argmax(pattern_activations)
    
    def visualize_recognition_performance(self) -> None:
        """Visualize the pattern recognition performance."""
        if not self.recognition_history:
            logger.warning("No recognition history available")
            return
        
        # Calculate moving average
        window_size = min(10, len(self.recognition_history))
        moving_avg = []
        
        for i in range(len(self.recognition_history)):
            if i < window_size - 1:
                continue
            
            window_avg = sum(self.recognition_history[i - window_size + 1:i + 1]) / window_size
            moving_avg.append(window_avg)
        
        # Plot raw results and moving average
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.recognition_history, 'o', alpha=0.5, label="Individual Results")
        plt.plot(range(window_size - 1, len(self.recognition_history)), 
                moving_avg, 'r-', linewidth=2, label=f"{window_size}-point Moving Average")
        
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label="Perfect Recognition")
        plt.axhline(y=1.0/self.num_patterns, color='r', linestyle='--', alpha=0.7, 
                   label="Random Chance")
        
        plt.title("Pattern Recognition Performance")
        plt.xlabel("Trial Number")
        plt.ylabel("Recognition Accuracy")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_patterns(self) -> None:
        """Visualize the patterns used in the demo."""
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(self.num_patterns)))
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        for i, pattern in enumerate(self.patterns):
            if i >= grid_size * grid_size:
                break
                
            # Create a grid representation
            grid = np.zeros((int(np.sqrt(self.input_size)), int(np.sqrt(self.input_size))))
            
            for idx, value in pattern.items():
                row = idx // int(np.sqrt(self.input_size))
                col = idx % int(np.sqrt(self.input_size))
                grid[row, col] = value
            
            # Plot
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(grid, cmap='viridis')
            plt.title(f"Pattern {i + 1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Run demo when script is executed directly
if __name__ == "__main__":
    # Create demo with 10x10 input grid and 5 patterns
    demo = PatternRecognitionDemo(input_size=100, num_patterns=5)
    
    # Visualize the patterns
    demo.visualize_patterns()
    
    # Start the demo
    demo.start()
    
    try:
        # Run for a while
        print("\nRunning pattern recognition. Press Ctrl+C to stop...\n")
        time.sleep(60)  # Run for 60 seconds
        
    except KeyboardInterrupt:
        print("\nStopping demo...")
    
    finally:
        # Stop the demo
        demo.stop()
        
        # Visualize performance
        demo.visualize_recognition_performance()
        
        # Visualize the neural fabric
        demo.fabric.visualize_network()
        demo.fabric.visualize_regions()
