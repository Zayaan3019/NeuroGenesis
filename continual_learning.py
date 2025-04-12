import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import logging
from typing import Dict, List, Tuple, Any
import random

from neural_fabric import NeuralFabric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContinualLearning")

class Task:
    """Represents a learning task for continual learning."""
    
    def __init__(self, name: str, input_size: int, output_size: int):
        """
        Initialize a learning task.
        
        Args:
            name: Name of the task
            input_size: Size of input patterns
            output_size: Size of output patterns
        """
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.examples = []
        
    def add_example(self, input_pattern: Dict[int, float], output_pattern: Dict[int, float]) -> None:
        """
        Add a training example to the task.
        
        Args:
            input_pattern: Input pattern (indices -> values)
            output_pattern: Expected output pattern (indices -> values)
        """
        self.examples.append((input_pattern, output_pattern))
    
    def get_random_example(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Get a random example from the task.
        
        Returns:
            Tuple of (input_pattern, output_pattern)
        """
        if not self.examples:
            # Create empty patterns if no examples
            return {}, {}
        
        return random.choice(self.examples)

class ContinualLearningDemo:
    """
    Demonstrates NeuralFabric's continual learning capabilities.
    """
    
    def __init__(self, num_tasks: int = 3, input_size: int = 100, task_outputs: int = 10):
        """
        Initialize the continual learning demo.
        
        Args:
            num_tasks: Number of tasks to learn
            input_size: Size of the input patterns
            task_outputs: Number of outputs per task
        """
        self.num_tasks = num_tasks
        self.input_size = input_size
        self.task_outputs = task_outputs
        
        # Create the neural fabric
        self.fabric = NeuralFabric("ContinualLearner")
        
        # Create regions
        self.input_region_id = self.fabric.create_sensory_region("Input", size=input_size)
        self.hidden_region_id = self.fabric.create_processing_region("Hidden", size=200)
        self.output_region_id = self.fabric.create_output_region("Output", size=num_tasks * task_outputs)
        
        # Connect regions
        self.fabric.connect_regions(self.input_region_id, self.hidden_region_id, connection_density=0.2)
        self.fabric.connect_regions(self.hidden_region_id, self.output_region_id, connection_density=0.3)
        
        # Generate tasks
        self.tasks = self._generate_tasks()
        
        # Tracking
        self.task_performances = {task.name: [] for task in self.tasks}
        self.current_task_idx = 0
        
        logger.info(f"Created continual learning demo with {num_tasks} tasks")
    
    def _generate_tasks(self) -> List[Task]:
        """
        Generate learning tasks.
        
        Returns:
            List of tasks
        """
        tasks = []
        
        for i in range(self.num_tasks):
            task_name = f"Task{i+1}"
            task = Task(task_name, self.input_size, self.task_outputs)
            
            # Generate 5 examples for this task
            for j in range(5):
                # Generate random input pattern with 20% active neurons
                input_indices = random.sample(range(self.input_size), int(self.input_size * 0.2))
                input_pattern = {idx: 0.8 + 0.2 * random.random() for idx in input_indices}
                
                # Generate target output pattern
                output_start = i * self.task_outputs
                output_pattern = {}
                
                # Set one neuron with high activity, others with low
                target_idx = random.randint(output_start, output_start + self.task_outputs - 1)
                
                for idx in range(output_start, output_start + self.task_outputs):
                    if idx == target_idx:
                        output_pattern[idx - output_start] = 0.9  # Target neuron
                    else:
                        output_pattern[idx - output_start] = 0.1  # Background
                
                task.add_example(input_pattern, output_pattern)
            
            tasks.append(task)
            logger.info(f"Generated task {task_name} with 5 examples")
        
        return tasks
    
    def start(self) -> None:
        """Start the neural fabric and continual learning."""
        self.fabric.start()
        
        # Start learning thread
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        logger.info("Started continual learning demo")
    
    def stop(self) -> None:
        """Stop the learning and neural fabric."""
        self.is_running = False
        
        if hasattr(self, 'learning_thread'):
            self.learning_thread.join(timeout=1.0)
        
        self.fabric.stop()
        
        logger.info("Stopped continual learning demo")
    
    def _learning_loop(self) -> None:
        """Main learning loop."""
        # Track which task we're currently learning
        current_task_idx = 0
        task_epochs = 10
        
        while self.is_running:
            # Get current task
            current_task = self.tasks[current_task_idx]
            
            logger.info(f"Learning {current_task.name} for {task_epochs} epochs")
            
            # Learn the current task
            for epoch in range(task_epochs):
                if not self.is_running:
                    return
                
                # Train on all examples
                for _ in range(len(current_task.examples)):
                    input_pattern, output_pattern = current_task.get_random_example()
                    
                    # Present input pattern
                    self.fabric.inject_pattern(self.input_region_id, input_pattern, strength=1.0)
                    
                    # Wait for processing
                    time.sleep(1.0)
                
                # Evaluate on all tasks
                self._evaluate_all_tasks()
                
                # Log progress
                for task in self.tasks:
                    if self.task_performances[task.name]:
                        latest_perf = self.task_performances[task.name][-1]
                        logger.info(f"Epoch {epoch+1}/{task_epochs}, {task.name}: accuracy={latest_perf:.2f}")
            
            # Move to next task
            current_task_idx = (current_task_idx + 1) % self.num_tasks
            self.current_task_idx = current_task_idx
            
            # Pause between tasks
            logger.info(f"Completed learning {current_task.name}, moving to next task")
            time.sleep(2.0)
    
    def _evaluate_all_tasks(self) -> None:
        """Evaluate performance on all tasks."""
        for task in self.tasks:
            accuracy = self._evaluate_task(task)
            self.task_performances[task.name].append(accuracy)
    
    def _evaluate_task(self, task: Task) -> float:
        """
        Evaluate performance on a specific task.
        
        Args:
            task: The task to evaluate
            
        Returns:
            Accuracy between 0 and 1
        """
        if not task.examples:
            return 0.0
        
        correct = 0
        total = len(task.examples)
        
        for input_pattern, expected_output in task.examples:
            # Present input pattern
            self.fabric.inject_pattern(self.input_region_id, input_pattern, strength=1.0)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Get output activity
            output_activity = self.fabric.get_output_activity(self.output_region_id)
            
            # Convert to task-specific output
            task_output = self._get_task_output(task, output_activity)
            
            # Check if the output matches expected
            is_correct = self._is_output_correct(task_output, expected_output)
            
            if is_correct:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _get_task_output(self, task: Task, output_activity: Dict[str, float]) -> Dict[int, float]:
        """
        Extract task-specific output from the global output activity.
        
        Args:
            task: The current task
            output_activity: Overall output neuron activity
            
        Returns:
            Task-specific output pattern
        """
        # Get task index
        task_idx = [i for i, t in enumerate(self.tasks) if t.name == task.name][0]
        task_start = task_idx * self.task_outputs
        
        # Get output neurons for this task
        regions = self.fabric.regions
        output_region = regions[self.output_region_id]
        output_neurons = list(output_region.neurons.values())
        
        # Group by task
        task_output = {}
        
        for i, neuron_id in enumerate(output_activity.keys()):
            neuron_idx = output_neurons.index(self.fabric.neurons[neuron_id])
            
            if task_start <= neuron_idx < task_start + self.task_outputs:
                task_output[neuron_idx - task_start] = output_activity[neuron_id]
        
        return task_output
    
    def _is_output_correct(self, actual_output: Dict[int, float], 
                         expected_output: Dict[int, float]) -> bool:
        """
        Check if the actual output matches the expected output.
        
        Args:
            actual_output: Actual output pattern
            expected_output: Expected output pattern
            
        Returns:
            True if output is correct, False otherwise
        """
        if not actual_output or not expected_output:
            return False
        
        # Find the most active neuron in actual and expected
        actual_max_idx = max(actual_output.keys(), key=lambda k: actual_output[k]) if actual_output else -1
        expected_max_idx = max(expected_output.keys(), key=lambda k: expected_output[k]) if expected_output else -1
        
        return actual_max_idx == expected_max_idx
    
    def visualize_performance(self) -> None:
        """Visualize the learning performance over time."""
        plt.figure(figsize=(12, 6))
        
        # Plot performance for each task
        for task_name, performance in self.task_performances.items():
            if performance:
                plt.plot(performance, label=task_name)
        
        plt.title("Continual Learning Performance")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_catastrophic_forgetting(self) -> None:
        """
        Visualize the effect of catastrophic forgetting 
        (or its mitigation in NeuralFabric).
        """
        plt.figure(figsize=(12, 6))
        
        # Plot average performance before and after tasks
        task_names = [task.name for task in self.tasks]
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        for i, task_name in enumerate(task_names):
            performance = self.task_performances[task_name]
            
            if len(performance) < 2:
                continue
            
            # Get task indices
            task_indices = []
            points_per_task = len(performance) // self.num_tasks
            
            for t in range(self.num_tasks):
                task_indices.append(t * points_per_task)
            
            # Highlight where this task was being learned
            start_idx = i * points_per_task
            end_idx = (i + 1) * points_per_task - 1
            
            plt.axvspan(start_idx, end_idx, alpha=0.2, color=colors[i % len(colors)])
            plt.plot(performance, 'o-', label=task_name, color=colors[i % len(colors)])
        
        # Add vertical lines between tasks
        for i in range(1, self.num_tasks):
            x = i * (len(next(iter(self.task_performances.values()), [])) // self.num_tasks)
            plt.axvline(x=x, color='k', linestyle='--', alpha=0.5)
        
        plt.title("Catastrophic Forgetting Analysis")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
