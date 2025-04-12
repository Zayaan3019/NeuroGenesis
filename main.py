import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

from neural_fabric import NeuralFabric
from pattern_recognition import PatternRecognitionDemo
from continual_learning import ContinualLearningDemo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralFabric-Main")

def main():
    """Main entry point for the NeuralFabric demonstration."""
    parser = argparse.ArgumentParser(description="NeuralFabric - Self-Evolving Distributed Intelligence System")
    
    # Add command-line arguments
    parser.add_argument('--demo', type=str, choices=['pattern', 'continual', 'custom'], 
                      default='pattern', help='Demo type to run')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration to run the demo (seconds)')
    parser.add_argument('--input-size', type=int, default=100,
                      help='Size of input patterns')
    parser.add_argument('--visualize', action='store_true', 
                      help='Visualize the neural network at the end')
    
    args = parser.parse_args()
    
    logger.info(f"Starting NeuralFabric with {args.demo} demo")
    
    try:
        if args.demo == 'pattern':
            run_pattern_recognition_demo(args)
        elif args.demo == 'continual':
            run_continual_learning_demo(args)
        elif args.demo == 'custom':
            run_custom_demo(args)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    logger.info("Demo completed")

def run_pattern_recognition_demo(args):
    """Run the pattern recognition demo."""
    logger.info("Starting Pattern Recognition Demo")
    
    # Create the demo
    demo = PatternRecognitionDemo(input_size=args.input_size, num_patterns=5)
    
    # Visualize the patterns
    demo.visualize_patterns()
    
    # Start the demo
    demo.start()
    
    try:
        logger.info(f"Running demo for {args.duration} seconds...")
        time.sleep(args.duration)
        
    finally:
        # Stop the demo
        demo.stop()
        
        # Visualize results
        demo.visualize_recognition_performance()
        
        if args.visualize:
            demo.fabric.visualize_network()
            demo.fabric.visualize_regions()

def run_continual_learning_demo(args):
    """Run the continual learning demo."""
    logger.info("Starting Continual Learning Demo")
    
    # Create the demo
    demo = ContinualLearningDemo(num_tasks=3, input_size=args.input_size)
    
    # Start the demo
    demo.start()
    
    try:
        logger.info(f"Running demo for {args.duration} seconds...")
        time.sleep(args.duration)
        
    finally:
        # Stop the demo
        demo.stop()
        
        # Visualize results
        demo.visualize_performance()
        demo.visualize_catastrophic_forgetting()
        
        if args.visualize:
            demo.fabric.visualize_network()
            demo.fabric.visualize_regions()

def run_custom_demo(args):
    """Run a custom demo showcasing various NeuralFabric capabilities."""
    logger.info("Starting Custom Demo")
    
    # Create a new neural fabric
    fabric = NeuralFabric("CustomDemo")
    
    # Create regions
    sensory_region = fabric.create_sensory_region("Visual", size=args.input_size)
    process_region1 = fabric.create_processing_region("Process1", size=150)
    process_region2 = fabric.create_processing_region("Process2", size=150)
    output_region = fabric.create_output_region("Motor", size=50)
    
    # Connect regions with different densities
    fabric.connect_regions(sensory_region, process_region1, connection_density=0.3)
    fabric.connect_regions(sensory_region, process_region2, connection_density=0.2)
    fabric.connect_regions(process_region1, process_region2, connection_density=0.4)
    fabric.connect_regions(process_region1, output_region, connection_density=0.3)
    fabric.connect_regions(process_region2, output_region, connection_density=0.3)
    
    # Start the fabric
    fabric.start()
    
    try:
        logger.info("Running custom demo with stimulus patterns...")
        
        # Run for the specified duration
        end_time = time.time() + args.duration
        
        while time.time() < end_time:
            # Generate a random input pattern
            pattern_size = int(args.input_size * 0.2)  # 20% active
            pattern = {}
            
            for idx in np.random.choice(args.input_size, pattern_size, replace=False):
                pattern[idx] = 0.8 + 0.2 * np.random.random()
            
            # Inject the pattern
            fabric.inject_pattern(sensory_region, pattern, strength=1.0)
            
            # Wait before next pattern
            time.sleep(2.0)
        
    finally:
        # Stop the fabric
        fabric.stop()
        
        # Visualize the network
        if args.visualize:
            fabric.visualize_network()
            fabric.visualize_regions()
        
        # Save the state
        fabric.save_state("neural_fabric_state.json")
        logger.info("Saved neural fabric state to neural_fabric_state.json")

if __name__ == "__main__":
    main()
