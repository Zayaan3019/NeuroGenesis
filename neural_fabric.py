import numpy as np
import time
import threading
import uuid
import heapq
import random
import logging
import json
import hashlib
import queue
import math
import os
from typing import Dict, List, Tuple, Set, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralFabric")

# Constants for the system
MAX_NEURONS = 10000  # Maximum neurons per region
MAX_CONNECTIONS_PER_NEURON = 1000
SPIKE_THRESHOLD = 0.65  # Threshold for neuron firing
RESTING_POTENTIAL = 0.0
REFRACTORY_PERIOD = 3  # Time steps a neuron is refractory after firing
LEARNING_RATE = 0.01
DECAY_RATE = 0.98  # Decay rate for neuron potentials
MAX_REGIONS = 16  # Maximum number of regions in the fabric
PRUNING_THRESHOLD = 0.01  # Threshold for connection pruning
GROWTH_PROBABILITY = 0.02  # Probability of new connection formation

@dataclass
class Spike:
    """Represents a neuronal spike in the neural fabric."""
    source_id: str  # Neuron ID that generated the spike
    target_id: str  # Target neuron ID
    timestamp: float  # When the spike was generated
    strength: float  # Strength of the spike
    features: Dict[str, float] = field(default_factory=dict)  # Feature vector

    def __lt__(self, other):
        # For priority queue based on timestamp
        return self.timestamp < other.timestamp

@dataclass
class Neuron:
    """Represents a single neuron in the neural fabric."""
    id: str
    region_id: str
    neuron_type: str  # 'sensory', 'processing', 'motor'
    potential: float = RESTING_POTENTIAL
    last_spike_time: float = 0.0
    refractory_until: float = 0.0
    created_at: float = field(default_factory=time.time)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Plasticity parameters
    plasticity: float = 0.5  # How easily this neuron forms new connections
    activation_count: int = 0  # How many times this neuron has fired
    
    # Connections - target_id -> weight
    connections: Dict[str, float] = field(default_factory=dict)
    
    # Reverse connections for backpropagation and learning
    reverse_connections: Set[str] = field(default_factory=set)
    
    # Queue of incoming spikes
    incoming_spikes: List[Spike] = field(default_factory=list)
    
    def add_connection(self, target_id: str, weight: float = None) -> None:
        """Add or update a connection to another neuron."""
        if weight is None:
            # Initialize with small random weight
            weight = np.random.normal(0.5, 0.1)
        
        self.connections[target_id] = weight
    
    def remove_connection(self, target_id: str) -> None:
        """Remove a connection to another neuron."""
        if target_id in self.connections:
            del self.connections[target_id]
    
    def update_potential(self, current_time: float) -> bool:
        """
        Update neuron potential based on incoming spikes and decay.
        Returns True if the neuron fires.
        """
        # Skip if in refractory period
        if current_time < self.refractory_until:
            return False
        
        # Apply decay to current potential
        self.potential *= DECAY_RATE
        
        # Process incoming spikes
        while self.incoming_spikes and self.incoming_spikes[0].timestamp <= current_time:
            spike = self.incoming_spikes.pop(0)
            self.potential += spike.strength
        
        # Check if potential is above threshold
        if self.potential >= SPIKE_THRESHOLD:
            # Neuron fires
            self.activation_count += 1
            self.last_spike_time = current_time
            self.refractory_until = current_time + REFRACTORY_PERIOD
            self.potential = RESTING_POTENTIAL
            return True
        
        return False
    
    def get_connection_strength(self, target_id: str) -> float:
        """Get the strength of a connection to another neuron."""
        return self.connections.get(target_id, 0.0)

@dataclass
class Region:
    """Represents a region of neurons in the neural fabric."""
    id: str
    name: str
    region_type: str  # 'sensory', 'associative', 'motor'
    neurons: Dict[str, Neuron] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    # Regional properties
    excitatory_ratio: float = 0.8  # Ratio of excitatory to inhibitory neurons
    plasticity: float = 0.5  # Overall plasticity of the region
    
    # Connectivity statistics
    internal_connectivity: Dict[str, int] = field(default_factory=lambda: {"total": 0, "active": 0})
    external_connectivity: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Activity metrics
    activity_level: float = 0.0  # Current activity level (0-1)
    activity_history: List[float] = field(default_factory=list)
    
    def add_neuron(self, neuron: Neuron) -> None:
        """Add a neuron to the region."""
        self.neurons[neuron.id] = neuron
    
    def remove_neuron(self, neuron_id: str) -> None:
        """Remove a neuron from the region."""
        if neuron_id in self.neurons:
            del self.neurons[neuron_id]
    
    def update_activity(self, current_time: float, window_size: int = 100) -> None:
        """Update the activity level of the region."""
        active_count = sum(1 for n in self.neurons.values() 
                          if current_time - n.last_spike_time < window_size)
        
        if self.neurons:
            self.activity_level = active_count / len(self.neurons)
        else:
            self.activity_level = 0.0
        
        self.activity_history.append(self.activity_level)
        if len(self.activity_history) > window_size:
            self.activity_history.pop(0)
    
    def get_active_neurons(self, current_time: float, time_window: float = 10.0) -> List[Neuron]:
        """Get neurons that have fired within the given time window."""
        return [n for n in self.neurons.values() 
                if current_time - n.last_spike_time <= time_window]

class NeuralFabric:
    """
    The main class implementing the NeuralFabric system.
    
    This system creates a self-organizing, adaptive neural network that:
    1. Dynamically grows and prunes connections based on activity
    2. Uses spike-timing-dependent plasticity for learning
    3. Implements homeostatic mechanisms to maintain stability
    4. Features hierarchical organization through regions
    """
    
    def __init__(self, name: str = "NeuralFabric"):
        """Initialize the NeuralFabric system."""
        self.name = name
        self.id = str(uuid.uuid4())
        self.current_time = 0.0
        
        # Core neural structures
        self.regions: Dict[str, Region] = {}
        self.neurons: Dict[str, Neuron] = {}  # All neurons indexed by ID
        
        # Spike propagation
        self.global_spike_queue: List[Spike] = []
        self.is_running = False
        self.processing_thread = None
        
        # For visualization
        self.activity_history: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            "spikes_processed": 0,
            "connections_created": 0,
            "connections_pruned": 0,
            "neurons_created": 0
        }
        
        logger.info(f"Initialized NeuralFabric system '{name}' with ID {self.id}")
    
    def create_region(self, name: str, region_type: str) -> str:
        """
        Create a new region in the neural fabric.
        
        Args:
            name: The name of the region
            region_type: The type of region ('sensory', 'associative', 'motor')
            
        Returns:
            The ID of the newly created region
        """
        if len(self.regions) >= MAX_REGIONS:
            logger.warning(f"Maximum number of regions ({MAX_REGIONS}) reached")
            return None
        
        region_id = str(uuid.uuid4())
        region = Region(
            id=region_id,
            name=name,
            region_type=region_type
        )
        
        self.regions[region_id] = region
        logger.info(f"Created {region_type} region '{name}' with ID {region_id}")
        
        return region_id
    
    def create_neuron(self, region_id: str, neuron_type: str,
                     features: Dict[str, float] = None) -> str:
        """
        Create a new neuron in the specified region.
        
        Args:
            region_id: The ID of the region to add the neuron to
            neuron_type: The type of neuron ('sensory', 'processing', 'motor')
            features: Initial feature vector for the neuron
            
        Returns:
            The ID of the newly created neuron
        """
        if region_id not in self.regions:
            logger.error(f"Region with ID {region_id} does not exist")
            return None
        
        region = self.regions[region_id]
        
        if len(region.neurons) >= MAX_NEURONS:
            logger.warning(f"Maximum number of neurons in region {region.name} reached")
            return None
        
        neuron_id = str(uuid.uuid4())
        
        neuron = Neuron(
            id=neuron_id,
            region_id=region_id,
            neuron_type=neuron_type,
            features=features or {},
            plasticity=region.plasticity * (0.8 + 0.4 * random.random())  # Add some variability
        )
        
        region.add_neuron(neuron)
        self.neurons[neuron_id] = neuron
        
        self.performance_metrics["neurons_created"] += 1
        
        return neuron_id
    
    def connect_neurons(self, source_id: str, target_id: str, weight: float = None) -> bool:
        """
        Create a connection between two neurons.
        
        Args:
            source_id: The ID of the source neuron
            target_id: The ID of the target neuron
            weight: The initial weight of the connection (random if None)
            
        Returns:
            True if the connection was created, False otherwise
        """
        if source_id not in self.neurons or target_id not in self.neurons:
            logger.error(f"Cannot connect: one or both neurons do not exist")
            return False
        
        source = self.neurons[source_id]
        target = self.neurons[target_id]
        
        # Check for maximum connections
        if len(source.connections) >= MAX_CONNECTIONS_PER_NEURON:
            logger.debug(f"Neuron {source_id} has reached maximum connections")
            return False
        
        # Create the connection
        source.add_connection(target_id, weight)
        target.reverse_connections.add(source_id)
        
        # Update regional connectivity statistics
        source_region = self.regions[source.region_id]
        target_region = self.regions[target.region_id]
        
        if source.region_id == target.region_id:
            # Internal connection
            source_region.internal_connectivity["total"] += 1
        else:
            # External connection
            if target.region_id not in source_region.external_connectivity:
                source_region.external_connectivity[target.region_id] = {"total": 0, "active": 0}
            
            source_region.external_connectivity[target.region_id]["total"] += 1
        
        self.performance_metrics["connections_created"] += 1
        
        return True
    
    def inject_spike(self, neuron_id: str, strength: float = 1.0, 
                    features: Dict[str, float] = None) -> bool:
        """
        Inject a spike directly into a neuron (typically used for sensory input).
        
        Args:
            neuron_id: The ID of the target neuron
            strength: The strength of the spike
            features: Feature vector associated with the spike
            
        Returns:
            True if the spike was injected, False otherwise
        """
        if neuron_id not in self.neurons:
            logger.error(f"Cannot inject spike: neuron {neuron_id} does not exist")
            return False
        
        neuron = self.neurons[neuron_id]
        
        # Create the spike
        spike = Spike(
            source_id="external",
            target_id=neuron_id,
            timestamp=self.current_time,
            strength=strength,
            features=features or {}
        )
        
        # Add to neuron's incoming spikes
        neuron.incoming_spikes.append(spike)
        neuron.incoming_spikes.sort(key=lambda s: s.timestamp)
        
        return True
    
    def start(self) -> None:
        """Start the NeuralFabric processing."""
        if self.is_running:
            logger.warning("NeuralFabric is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Started NeuralFabric processing")
    
    def stop(self) -> None:
        """Stop the NeuralFabric processing."""
        if not self.is_running:
            logger.warning("NeuralFabric is not running")
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        logger.info("Stopped NeuralFabric processing")
    
    def _process_loop(self) -> None:
        """Main processing loop for the neural fabric."""
        time_step = 0.1  # Time step for simulation
        
        while self.is_running:
            self.current_time += time_step
            
            # Process all neurons
            self._process_neurons()
            
            # Process all spikes
            self._process_spikes()
            
            # Apply plasticity and structural changes
            if int(self.current_time * 10) % 10 == 0:  # Every ~1 time unit
                self._apply_plasticity()
            
            # Update region activity levels
            for region in self.regions.values():
                region.update_activity(self.current_time)
                self.activity_history[region.id].append(region.activity_level)
            
            # Slow down simulation to real-time
            time.sleep(time_step * 0.1)  # Run 10x faster than real-time
    
    def _process_neurons(self) -> None:
        """Process all neurons, updating potentials and generating spikes."""
        neurons_to_process = list(self.neurons.values())
        random.shuffle(neurons_to_process)  # Add some randomness to the order
        
        for neuron in neurons_to_process:
            # Update neuron potential and check if it fires
            if neuron.update_potential(self.current_time):
                # Neuron fired, create spikes to all connected neurons
                for target_id, weight in neuron.connections.items():
                    if target_id in self.neurons:
                        # Create a spike
                        spike = Spike(
                            source_id=neuron.id,
                            target_id=target_id,
                            timestamp=self.current_time + 0.1 * random.random(),  # Add jitter
                            strength=weight,
                            features=neuron.features.copy()
                        )
                        
                        # Add to global spike queue
                        self.global_spike_queue.append(spike)
    
    def _process_spikes(self) -> None:
        """Process all pending spikes in the system."""
        # Sort spikes by timestamp
        self.global_spike_queue.sort(key=lambda s: s.timestamp)
        
        # Process spikes that are due
        spikes_to_process = []
        current_idx = 0
        
        while (current_idx < len(self.global_spike_queue) and 
              self.global_spike_queue[current_idx].timestamp <= self.current_time):
            spikes_to_process.append(self.global_spike_queue[current_idx])
            current_idx += 1
        
        # Remove processed spikes from the queue
        self.global_spike_queue = self.global_spike_queue[current_idx:]
        
        # Deliver spikes to target neurons
        for spike in spikes_to_process:
            if spike.target_id in self.neurons:
                target = self.neurons[spike.target_id]
                target.incoming_spikes.append(spike)
                target.incoming_spikes.sort(key=lambda s: s.timestamp)
                
                self.performance_metrics["spikes_processed"] += 1
    
    def _apply_plasticity(self) -> None:
        """Apply plasticity rules to modify connection weights."""
        # For each neuron, adjust weights based on spike timing
        for neuron_id, neuron in self.neurons.items():
            # Skip neurons that haven't fired
            if neuron.last_spike_time <= 0:
                continue
            
            # Get all neurons that connect to this neuron
            for source_id in neuron.reverse_connections:
                if source_id in self.neurons:
                    source = self.neurons[source_id]
                    
                    # Skip if source hasn't fired
                    if source.last_spike_time <= 0:
                        continue
                    
                    # Calculate time difference between spikes
                    time_diff = neuron.last_spike_time - source.last_spike_time
                    
                    # Apply spike-timing-dependent plasticity (STDP)
                    if time_diff > 0 and time_diff < 20.0:
                        # Post-synaptic neuron fired after pre-synaptic neuron (LTP)
                        stdp_factor = math.exp(-time_diff / 10.0)
                        weight_change = LEARNING_RATE * stdp_factor
                        
                        # Strengthen the connection
                        current_weight = source.get_connection_strength(neuron_id)
                        new_weight = min(1.0, current_weight + weight_change)
                        source.connections[neuron_id] = new_weight
                    
                    elif time_diff < 0 and time_diff > -20.0:
                        # Post-synaptic neuron fired before pre-synaptic neuron (LTD)
                        stdp_factor = math.exp(time_diff / 10.0)
                        weight_change = LEARNING_RATE * stdp_factor
                        
                        # Weaken the connection
                        current_weight = source.get_connection_strength(neuron_id)
                        new_weight = max(0.0, current_weight - weight_change)
                        source.connections[neuron_id] = new_weight
            
            # Prune weak connections
            connections_to_remove = []
            for target_id, weight in neuron.connections.items():
                if weight < PRUNING_THRESHOLD:
                    connections_to_remove.append(target_id)
            
            for target_id in connections_to_remove:
                neuron.remove_connection(target_id)
                if target_id in self.neurons:
                    self.neurons[target_id].reverse_connections.discard(neuron_id)
                
                self.performance_metrics["connections_pruned"] += 1
            
            # Form new connections with probability based on neuron plasticity
            # This models the biological process of synaptogenesis
            if random.random() < neuron.plasticity * GROWTH_PROBABILITY:
                # Choose a target region with preference for regions with high activity
                target_regions = []
                weights = []
                
                for region_id, region in self.regions.items():
                    target_regions.append(region_id)
                    
                    # Prefer regions with higher activity
                    if region.activity_history:
                        activity = sum(region.activity_history) / len(region.activity_history)
                        weights.append(0.1 + activity)
                    else:
                        weights.append(0.1)
                
                if target_regions:
                    # Normalize weights
                    weights = [w / sum(weights) for w in weights]
                    
                    # Select a target region
                    target_region_id = random.choices(target_regions, weights=weights, k=1)[0]
                    target_region = self.regions[target_region_id]
                    
                    # Select a random neuron in the target region
                    if target_region.neurons:
                        potential_targets = list(target_region.neurons.keys())
                        target_id = random.choice(potential_targets)
                        
                        # Create connection if it doesn't already exist
                        if (target_id != neuron_id and 
                            target_id not in neuron.connections and
                            len(neuron.connections) < MAX_CONNECTIONS_PER_NEURON):
                            
                            # Create the connection
                            self.connect_neurons(neuron_id, target_id)
    
    def inject_pattern(self, region_id: str, pattern: Dict[int, float],
                     duration: float = 1.0, strength: float = 1.0) -> None:
        """
        Inject a pattern of activity into a region.
        
        Args:
            region_id: The ID of the target region
            pattern: Mapping of neuron indices to activation values
            duration: Duration of the pattern in time units
            strength: Base strength of the injected spikes
        """
        if region_id not in self.regions:
            logger.error(f"Region with ID {region_id} does not exist")
            return
        
        region = self.regions[region_id]
        neurons = list(region.neurons.values())
        
        if not neurons:
            logger.warning(f"Region {region.name} has no neurons")
            return
        
        # Inject spikes according to the pattern
        for idx, value in pattern.items():
            if idx < len(neurons):
                neuron = neurons[idx]
                
                # Scale spike strength by pattern value
                spike_strength = strength * value
                
                # Inject the spike
                self.inject_spike(neuron.id, strength=spike_strength)
        
        logger.info(f"Injected pattern into region {region.name}")
    
    def create_sensory_region(self, name: str, size: int = 100) -> str:
        """
        Create a sensory region with the specified number of neurons.
        
        Args:
            name: The name of the region
            size: The number of neurons to create
            
        Returns:
            The ID of the created region
        """
        region_id = self.create_region(name, region_type="sensory")
        
        if not region_id:
            return None
        
        # Create neurons
        for _ in range(min(size, MAX_NEURONS)):
            self.create_neuron(region_id, neuron_type="sensory")
        
        logger.info(f"Created sensory region '{name}' with {size} neurons")
        return region_id
    
    def create_processing_region(self, name: str, size: int = 200) -> str:
        """
        Create a processing region with the specified number of neurons.
        
        Args:
            name: The name of the region
            size: The number of neurons to create
            
        Returns:
            The ID of the created region
        """
        region_id = self.create_region(name, region_type="associative")
        
        if not region_id:
            return None
        
        # Create neurons
        for _ in range(min(size, MAX_NEURONS)):
            self.create_neuron(region_id, neuron_type="processing")
        
        logger.info(f"Created processing region '{name}' with {size} neurons")
        return region_id
    
    def create_output_region(self, name: str, size: int = 50) -> str:
        """
        Create an output region with the specified number of neurons.
        
        Args:
            name: The name of the region
            size: The number of neurons to create
            
        Returns:
            The ID of the created region
        """
        region_id = self.create_region(name, region_type="motor")
        
        if not region_id:
            return None
        
        # Create neurons
        for _ in range(min(size, MAX_NEURONS)):
            self.create_neuron(region_id, neuron_type="motor")
        
        logger.info(f"Created output region '{name}' with {size} neurons")
        return region_id
    
    def connect_regions(self, source_id: str, target_id: str, 
                       connection_density: float = 0.1) -> int:
        """
        Create connections between two regions.
        
        Args:
            source_id: The ID of the source region
            target_id: The ID of the target region
            connection_density: Fraction of possible connections to create
            
        Returns:
            Number of connections created
        """
        if source_id not in self.regions or target_id not in self.regions:
            logger.error(f"Cannot connect regions: one or both regions do not exist")
            return 0
        
        source_region = self.regions[source_id]
        target_region = self.regions[target_id]
        
        source_neurons = list(source_region.neurons.values())
        target_neurons = list(target_region.neurons.values())
        
        if not source_neurons or not target_neurons:
            logger.warning(f"Cannot connect regions: one or both regions have no neurons")
            return 0
        
        # Calculate number of connections to create
        max_connections = len(source_neurons) * len(target_neurons)
        num_connections = int(max_connections * connection_density)
        
        # Create random connections
        connections_created = 0
        for _ in range(num_connections):
            source = random.choice(source_neurons)
            target = random.choice(target_neurons)
            
            if target.id not in source.connections:
                if self.connect_neurons(source.id, target.id):
                    connections_created += 1
        
        logger.info(f"Created {connections_created} connections from region {source_region.name} to {target_region.name}")
        return connections_created
    
    def get_region_activity(self, region_id: str) -> float:
        """
        Get the current activity level of a region.
        
        Args:
            region_id: The ID of the region
            
        Returns:
            Activity level between 0 and 1
        """
        if region_id not in self.regions:
            logger.error(f"Region with ID {region_id} does not exist")
            return 0.0
        
        return self.regions[region_id].activity_level
    
    def get_output_activity(self, region_id: str) -> Dict[str, float]:
        """
        Get the activity of neurons in an output region.
        
        Args:
            region_id: The ID of the output region
            
        Returns:
            Dictionary mapping neuron IDs to activity levels
        """
        if region_id not in self.regions:
            logger.error(f"Region with ID {region_id} does not exist")
            return {}
        
        region = self.regions[region_id]
        
        # Get activity levels based on recent firing
        output = {}
        for neuron_id, neuron in region.neurons.items():
            # Calculate activity based on time since last spike
            time_since_spike = self.current_time - neuron.last_spike_time
            
            if time_since_spike <= 5.0:  # Within last 5 time units
                # Exponential decay of activity
                activity = math.exp(-time_since_spike / 2.0)
                output[neuron_id] = activity
        
        return output
    
    def visualize_regions(self, figure_size: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize the regions and their activity.
        
        Args:
            figure_size: Size of the figure (width, height) in inches
        """
        plt.figure(figsize=figure_size)
        
        # Plot activity history for each region
        for region_id, region in self.regions.items():
            activity_hist = self.activity_history.get(region_id, [])
            if activity_hist:
                plt.plot(activity_hist[-100:], label=region.name)
        
        plt.title("Region Activity")
        plt.xlabel("Time Steps")
        plt.ylabel("Activity Level")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_network(self, max_neurons: int = 100) -> None:
        """
        Visualize the neural network structure.
        
        Args:
            max_neurons: Maximum number of neurons to display per region
        """
        G = nx.DiGraph()
        
        # Add regions as super-nodes
        for region_id, region in self.regions.items():
            G.add_node(f"Region:{region.name}", 
                     type="region", 
                     color={"sensory": "blue", "associative": "green", "motor": "red"}[region.region_type],
                     size=20)
        
        # Sample neurons from each region
        selected_neurons = {}
        for region_id, region in self.regions.items():
            neurons = list(region.neurons.items())
            if len(neurons) > max_neurons:
                neurons = random.sample(neurons, max_neurons)
            
            # Add neuron nodes
            for neuron_id, neuron in neurons:
                G.add_node(neuron_id, 
                         type="neuron", 
                         region=region.name,
                         color={"sensory": "lightblue", "processing": "lightgreen", "motor": "salmon"}[neuron.neuron_type],
                         size=10)
                
                # Connect neuron to its region
                G.add_edge(f"Region:{region.name}", neuron_id, weight=0.5, color="gray", type="belongs_to")
                
                selected_neurons[neuron_id] = neuron
        
        # Add connections between neurons
        for neuron_id, neuron in selected_neurons.items():
            for target_id, weight in neuron.connections.items():
                if target_id in selected_neurons:
                    G.add_edge(neuron_id, target_id, weight=weight, color="black", type="neural")
        
        # Draw the network
        plt.figure(figsize=(12, 12))
        
        # Get positions for nodes
        pos = nx.spring_layout(G, k=0.5)
        
        # Draw region nodes
        region_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "region"]
        region_colors = [G.nodes[n]["color"] for n in region_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=region_nodes, node_color=region_colors, 
                             node_size=[G.nodes[n]["size"] * 200 for n in region_nodes], alpha=0.7)
        
        # Draw neuron nodes
        neuron_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "neuron"]
        neuron_colors = [G.nodes[n]["color"] for n in neuron_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_nodes, node_color=neuron_colors, 
                             node_size=[G.nodes[n]["size"] * 20 for n in neuron_nodes], alpha=0.7)
        
        # Draw edges
        neural_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "neural"]
        belongs_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "belongs_to"]
        
        nx.draw_networkx_edges(G, pos, edgelist=neural_edges, width=0.5, alpha=0.3, arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=belongs_edges, width=0.5, alpha=0.2, 
                             style="dashed", arrows=False)
        
        # Draw labels for regions only
        region_labels = {n: n.split(":")[1] for n in region_nodes}
        nx.draw_networkx_labels(G, pos, labels=region_labels, font_size=10)
        
        plt.title(f"Neural Fabric Network Structure (showing {len(neuron_nodes)} neurons)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current state of the neural fabric to a file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a simplified state dictionary without complex objects
            state = {
                "id": self.id,
                "name": self.name,
                "current_time": self.current_time,
                "regions": {},
                "performance_metrics": self.performance_metrics.copy()
            }
            
            # Save region information
            for region_id, region in self.regions.items():
                state["regions"][region_id] = {
                    "id": region.id,
                    "name": region.name,
                    "region_type": region.region_type,
                    "neuron_count": len(region.neurons),
                    "activity_level": region.activity_level,
                    "neurons": {}
                }
                
                # Save neuron information
                for neuron_id, neuron in region.neurons.items():
                    state["regions"][region_id]["neurons"][neuron_id] = {
                        "id": neuron.id,
                        "type": neuron.neuron_type,
                        "potential": neuron.potential,
                        "last_spike_time": neuron.last_spike_time,
                        "activation_count": neuron.activation_count,
                        "connections": neuron.connections.copy()
                    }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved neural fabric state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load the state of the neural fabric from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from file
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Reset current state
            self.regions = {}
            self.neurons = {}
            self.global_spike_queue = []
            
            # Set basic properties
            self.id = state["id"]
            self.name = state["name"]
            self.current_time = state["current_time"]
            self.performance_metrics = state["performance_metrics"]
            
            # Recreate regions and neurons
            for region_id, region_data in state["regions"].items():
                # Create region
                region = Region(
                    id=region_data["id"],
                    name=region_data["name"],
                    region_type=region_data["region_type"]
                )
                
                self.regions[region_id] = region
                
                # Recreate neurons
                for neuron_id, neuron_data in region_data["neurons"].items():
                    neuron = Neuron(
                        id=neuron_data["id"],
                        region_id=region_id,
                        neuron_type=neuron_data["type"],
                        potential=neuron_data["potential"],
                        last_spike_time=neuron_data["last_spike_time"],
                        activation_count=neuron_data["activation_count"]
                    )
                    
                    # Add connections
                    for target_id, weight in neuron_data["connections"].items():
                        neuron.connections[target_id] = weight
                    
                    # Add to region and global neurons
                    region.neurons[neuron_id] = neuron
                    self.neurons[neuron_id] = neuron
            
            # Restore reverse connections
            for neuron_id, neuron in self.neurons.items():
                for target_id in neuron.connections:
                    if target_id in self.neurons:
                        self.neurons[target_id].reverse_connections.add(neuron_id)
            
            logger.info(f"Loaded neural fabric state from {filepath} with {len(self.neurons)} neurons")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
